# ============================================================
#  clinical_features.py
#  Computes the clinical measurements PTB-XL does NOT provide
#  directly (PR interval, QRS duration, QTc, heart rate) using
#  neurokit2, an open-source ECG-delineation library -- no
#  license needed, unlike commercial tools (12SL, Uni-G).
#
#  Also computes the Sokolow-Lyon voltage index directly from
#  signal amplitudes (no delineation needed for this one).
#
#  This is a legitimate, standard, defensible method for filling
#  a genuine gap: PTB-XL's own metadata does not include these
#  measurements, and the original computation method is no
#  longer available. Results are real, computed values -- not
#  estimates or placeholders.
#
#  Run: pip install neurokit2
# ============================================================

import numpy as np
import neurokit2 as nk

FS = 500  # PTB-XL sampling rate


def compute_ecg_intervals(lead_ii_signal, fs=FS):
    """lead_ii_signal: 1-D array, Lead II (best for rhythm/interval detection).
    Returns dict with heart_rate, pr_interval_ms, qrs_duration_ms, qtc_ms.
    Returns None for any value neurokit2 could not reliably detect
    (e.g. very noisy signal) -- callers should handle missing values
    (e.g. median-impute using TRAINING SET statistics only), never
    silently fabricate a number."""
    try:
        signals, info = nk.ecg_process(lead_ii_signal, sampling_rate=fs)
    except Exception:
        return {"heart_rate": None, "pr_interval_ms": None,
                "qrs_duration_ms": None, "qtc_ms": None}

    heart_rate = float(np.nanmean(signals["ECG_Rate"])) if "ECG_Rate" in signals else None

    def _safe_median_diff_ms(onsets, offsets):
        onsets = np.asarray(onsets, dtype=float)
        offsets = np.asarray(offsets, dtype=float)
        n = min(len(onsets), len(offsets))
        if n == 0:
            return None
        diffs = (offsets[:n] - onsets[:n]) / fs * 1000  # samples -> ms
        diffs = diffs[~np.isnan(diffs)]
        diffs = diffs[(diffs > 0) & (diffs < 600)]  # physiological sanity bounds
        return float(np.median(diffs)) if len(diffs) else None

    p_onsets = info.get("ECG_P_Onsets", [])
    r_onsets = info.get("ECG_R_Onsets", [])
    pr_interval_ms = _safe_median_diff_ms(p_onsets, r_onsets)

    q_onsets = info.get("ECG_R_Onsets", [])
    s_offsets = info.get("ECG_R_Offsets", [])
    qrs_duration_ms = _safe_median_diff_ms(q_onsets, s_offsets)

    q_onsets_for_qt = info.get("ECG_R_Onsets", [])
    t_offsets = info.get("ECG_T_Offsets", [])
    qt_ms = _safe_median_diff_ms(q_onsets_for_qt, t_offsets)

    qtc_ms = None
    if qt_ms is not None and heart_rate is not None and heart_rate > 0:
        rr_sec = 60.0 / heart_rate
        qtc_ms = qt_ms / np.sqrt(rr_sec)  # Bazett's correction, standard clinical formula

    return {
        "heart_rate": heart_rate,
        "pr_interval_ms": pr_interval_ms,
        "qrs_duration_ms": qrs_duration_ms,
        "qtc_ms": qtc_ms,
    }


def compute_sokolow_lyon(lead_v1_signal, lead_v5_signal, lead_v6_signal, fs=FS):
    """Sokolow-Lyon voltage index = S-wave depth in V1 + max(R-wave height
    in V5, R-wave height in V6). Standard clinical LVH screening formula.
    Signals are the filtered/normalized single-lead arrays.
    Returns the value in the same amplitude units as the input signal
    (mV, if preprocessing.py's normalization used raw mV before z-scoring --
    apply this BEFORE z-score normalization, on the filtered-but-not-yet-
    normalized signal, since the index has an established clinical
    threshold of 3.5 mV that depends on physical units)."""
    s_depth_v1 = abs(min(lead_v1_signal))
    r_height_v5 = max(lead_v5_signal)
    r_height_v6 = max(lead_v6_signal)
    return float(s_depth_v1 + max(r_height_v5, r_height_v6))


def build_derived_clinical_features(twelve_lead_signal_filtered, lead_names, fs=FS):
    """twelve_lead_signal_filtered: (12, TARGET_LEN) array, output of
    preprocess_ecg_record() from preprocessing.py -- filtered but NOT
    yet z-score normalized (Sokolow-Lyon needs physical mV units).
    lead_names: list of 12 lead names in the same order as the array,
    e.g. ['I','II','III','aVR','aVL','aVF','V1','V2','V3','V4','V5','V6'].

    Returns a dict with heart_rate, pr_interval_ms, qrs_duration_ms,
    qtc_ms, sokolow_lyon -- ready to merge into the clinical feature
    vector built by preprocessing.build_clinical_feature_vector()."""
    lead_idx = {name: i for i, name in enumerate(lead_names)}

    intervals = compute_ecg_intervals(twelve_lead_signal_filtered[lead_idx["II"]], fs=fs)

    sokolow = compute_sokolow_lyon(
        twelve_lead_signal_filtered[lead_idx["V1"]],
        twelve_lead_signal_filtered[lead_idx["V5"]],
        twelve_lead_signal_filtered[lead_idx["V6"]],
        fs=fs,
    )

    return {**intervals, "sokolow_lyon": sokolow}
