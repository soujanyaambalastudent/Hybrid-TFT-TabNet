# ============================================================
#  preprocessing.py
#  ECG signal preprocessing pipeline, matching the manuscript's
#  description exactly:
#    - Baseline wander removal: median filter, 600 ms window
#    - High-frequency noise: 5th-order Butterworth low-pass, 40 Hz cutoff
#    - Powerline noise: 50 Hz notch filter
#    - Amplitude normalization: per-lead training-set mean/std
#    - Zero-padding / truncation to a fixed length of 5,000 samples
#  Output tensor shape: (N, 12, 5000), matching Section III.
#
#  Also builds the 25-dimensional structured clinical feature
#  vector for the TabNet branch, per Section III.C:
#    age (norm.), sex, recording-device one-hot, avg heart rate,
#    PR interval, QRS duration, QT/QTc interval, frontal axis,
#    Sokolow-Lyon voltage index, 6 binary form-label flags
#    (ST-elevation, T-abnormality, conduction defect, LVH, RBBB,
#    LBBB), a co-morbidity severity score, and 6 binary rhythm
#    flags (sinus, AF, SVT, junctional, pacemaker, other).
#
#  Run as a single Colab cell, or import its functions from
#  train.py / evaluate.py. Requires: numpy, scipy, wfdb, pandas.
# ============================================================

import numpy as np
import pandas as pd
from scipy.signal import medfilt, butter, filtfilt, iirnotch

FS = 500          # PTB-XL sampling rate (Hz)
TARGET_LEN = 5000  # 10 s at 500 Hz
N_LEADS = 12

BASELINE_WANDER_WINDOW_MS = 600
LOWPASS_CUTOFF_HZ = 40
LOWPASS_ORDER = 5
NOTCH_FREQ_HZ = 50
NOTCH_Q = 30  # quality factor for the notch filter


def remove_baseline_wander(signal_1d, fs=FS, window_ms=BASELINE_WANDER_WINDOW_MS):
    """Median-filter baseline wander removal. Window length is rounded
    to the nearest odd sample count, as scipy.medfilt requires odd kernel size."""
    window_samples = int(round(window_ms / 1000 * fs))
    if window_samples % 2 == 0:
        window_samples += 1
    baseline = medfilt(signal_1d, kernel_size=window_samples)
    return signal_1d - baseline


def lowpass_filter(signal_1d, fs=FS, cutoff=LOWPASS_CUTOFF_HZ, order=LOWPASS_ORDER):
    nyq = fs / 2
    b, a = butter(order, cutoff / nyq, btype="low")
    return filtfilt(b, a, signal_1d)


def notch_filter(signal_1d, fs=FS, freq=NOTCH_FREQ_HZ, q=NOTCH_Q):
    b, a = iirnotch(freq / (fs / 2), q)
    return filtfilt(b, a, signal_1d)


def pad_or_truncate(signal_1d, target_len=TARGET_LEN):
    n = len(signal_1d)
    if n == target_len:
        return signal_1d
    if n > target_len:
        return signal_1d[:target_len]
    return np.pad(signal_1d, (0, target_len - n), mode="constant")


def preprocess_single_lead(signal_1d, fs=FS):
    x = remove_baseline_wander(signal_1d, fs)
    x = lowpass_filter(x, fs)
    x = notch_filter(x, fs)
    x = pad_or_truncate(x)
    return x


def preprocess_ecg_record(raw_signal_12xN, fs=FS):
    """raw_signal_12xN: array of shape (12, N) or (N, 12) from WFDB.
    Returns array of shape (12, TARGET_LEN), filtered but NOT yet
    amplitude-normalized (normalization needs training-set statistics,
    applied separately by normalize_with_training_stats)."""
    raw = np.asarray(raw_signal_12xN)
    if raw.shape[0] != N_LEADS and raw.shape[1] == N_LEADS:
        raw = raw.T  # ensure (12, N)
    processed = np.stack([preprocess_single_lead(raw[lead], fs) for lead in range(N_LEADS)])
    return processed  # (12, TARGET_LEN)


def compute_training_lead_stats(processed_signals_stack):
    """processed_signals_stack: (N_train, 12, TARGET_LEN).
    Returns per-lead mean and std computed ONLY from the training
    partition, to be applied to val/test as well (no leakage)."""
    mean = processed_signals_stack.mean(axis=(0, 2), keepdims=True)   # (1,12,1)
    std = processed_signals_stack.std(axis=(0, 2), keepdims=True) + 1e-8
    return mean, std


def normalize_with_training_stats(processed_signals_stack, mean, std):
    return (processed_signals_stack - mean) / std


# ---------------------------------------------------------------
# Clinical feature vector (25-dim) for the TabNet branch
# ---------------------------------------------------------------

FORM_LABEL_FLAGS = ["ST-elevation", "T-abnormality", "conduction-defect",
                     "LVH", "RBBB", "LBBB"]
RHYTHM_FLAGS = ["sinus-rhythm", "atrial-fibrillation", "SVT",
                 "junctional-rhythm", "pacemaker", "other-rhythm"]


def build_clinical_feature_vector(record_meta, device_categories, train_continuous_stats=None):
    """record_meta: dict-like with the raw fields for one record
    (age, sex, recording device id, heart-rate, PR, QRS, QTc,
    frontal_axis, sokolow_lyon, comorbidity_score, and the 12
    binary form/rhythm flags already derived from SCP form/rhythm
    codes -- see the leakage-analysis note in Section III.D: these
    must come ONLY from form/rhythm layer codes, never from the
    diagnostic layer used for the label itself).

    device_categories: fixed, sorted list of device ids seen in the
    TRAINING set, used to build a consistent one-hot encoding across
    train/val/test.

    train_continuous_stats: optional dict of {feature_name: (mean, std)}
    computed from the training set only, for continuous-feature
    normalization; pass None to build a raw (un-normalized) vector,
    e.g. when first computing the stats themselves.

    Returns a 1D numpy array. Exact length = 1 (age) + 1 (sex)
    + len(device_categories) (one-hot) + 6 (continuous measurements:
    heart rate, PR, QRS, QTc, frontal axis, Sokolow-Lyon) + 6 (form
    flags) + 1 (comorbidity score) + 6 (rhythm flags).
    Total should equal 25 for the device one-hot dimensionality
    actually present in the training set -- print and check this
    when first building the vectors; it is a property of the data,
    not a fixed constant, since it depends on how many distinct
    recording devices appear in PTB-XL's metadata.
    """
    continuous = {
        "age": record_meta["age"],
        "heart_rate": record_meta["heart_rate"],
        "pr_interval": record_meta["pr_interval"],
        "qrs_duration": record_meta["qrs_duration"],
        "qtc_interval": record_meta["qtc_interval"],
        "frontal_axis": record_meta["frontal_axis"],
        "sokolow_lyon": record_meta["sokolow_lyon"],
        "comorbidity_score": record_meta["comorbidity_score"],
    }
    if train_continuous_stats is not None:
        for k in continuous:
            mean, std = train_continuous_stats[k]
            continuous[k] = (continuous[k] - mean) / (std + 1e-8)

    sex = float(record_meta["sex"])  # 0/1, already binary

    device_onehot = [1.0 if record_meta["device"] == d else 0.0 for d in device_categories]

    form_flags = [float(record_meta[f]) for f in FORM_LABEL_FLAGS]
    rhythm_flags = [float(record_meta[f]) for f in RHYTHM_FLAGS]

    vector = (
        [continuous["age"], sex]
        + device_onehot
        + [continuous["heart_rate"], continuous["pr_interval"], continuous["qrs_duration"],
           continuous["qtc_interval"], continuous["frontal_axis"], continuous["sokolow_lyon"]]
        + form_flags
        + [continuous["comorbidity_score"]]
        + rhythm_flags
    )
    return np.array(vector, dtype=np.float32)


if __name__ == "__main__":
    print("preprocessing.py loaded. Import preprocess_ecg_record(), "
          "compute_training_lead_stats(), normalize_with_training_stats(), "
          "and build_clinical_feature_vector() from train.py / evaluate.py.")
    print(f"Target ECG tensor shape per record: ({N_LEADS}, {TARGET_LEN})")
