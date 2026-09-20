# ============================================================
#  data_cache.py
#  Provides get_ecg_tensor(record_id) and get_clinical_vector
#  (record_id) -- the two functions train.py / evaluate.py need.
#  Loads from the cache built by build_cache.py, applies the
#  normalization computed by compute_normalization_stats.py.
#
#  This closes the gap that used to raise NotImplementedError
#  in train.py / evaluate.py.
# ============================================================

import json
import os

import numpy as np

from preprocessing import FORM_LABEL_FLAGS, RHYTHM_FLAGS

PTBXL_META_DIR = os.environ.get("PTBXL_META_DIR") or (
    "/content/ptbxl_meta" if os.path.exists("/content") else
    ("/kaggle/working/ptbxl_meta" if os.path.exists("/kaggle") else "./ptbxl_meta"))
CACHE_DIR = os.environ.get("PTBXL_CACHE_DIR") or (
    "/content/ptbxl_cache" if os.path.exists("/content") else
    ("/kaggle/working/ptbxl_cache" if os.path.exists("/kaggle") else "./ptbxl_cache"))

with open(os.path.join(PTBXL_META_DIR, "normalization_stats.json")) as f:
    _STATS = json.load(f)

_ECG_MEAN = np.array(_STATS["ecg_lead_mean"]).reshape(12, 1)
_ECG_STD = np.array(_STATS["ecg_lead_std"]).reshape(12, 1)
_DEVICE_CATEGORIES = _STATS["device_categories"]
_CONTINUOUS_STATS = _STATS["continuous_features"]

_CONTINUOUS_ORDER = ["heart_rate", "pr_interval", "qrs_duration",
                      "qtc_interval", "frontal_axis", "sokolow_lyon"]


def get_ecg_tensor(record_id):
    path = os.path.join(CACHE_DIR, f"{record_id}.npz")
    data = np.load(path, allow_pickle=True)
    ecg = data["ecg_filtered"]  # (12, 5000), filtered but not yet normalized
    return (ecg - _ECG_MEAN) / _ECG_STD


def _impute_or_normalize(value, field_name):
    stats = _CONTINUOUS_STATS.get(field_name)
    if stats is None:
        return 0.0
    if value is None or (isinstance(value, float) and np.isnan(value)):
        value = stats["mean"]  # median-impute using TRAINING stats, never a guess per-record
    return (value - stats["mean"]) / stats["std"]


def get_clinical_vector(record_id):
    path = os.path.join(CACHE_DIR, f"{record_id}.npz")
    data = np.load(path, allow_pickle=True)
    meta = data["clinical_raw"][0]

    age_norm = _impute_or_normalize(meta.get("age"), "age") if "age" in _CONTINUOUS_STATS else \
        (meta.get("age", 0.0) or 0.0)
    sex = float(meta.get("sex", 0.0) or 0.0)

    device_onehot = [1.0 if meta.get("device") == d else 0.0 for d in _DEVICE_CATEGORIES]

    continuous = [_impute_or_normalize(meta.get(f), f) for f in _CONTINUOUS_ORDER]

    form_flags = [float(meta.get(f, 0.0) or 0.0) for f in FORM_LABEL_FLAGS]
    comorbidity = float(meta.get("comorbidity_score", 0.0) or 0.0)
    rhythm_flags = [float(meta.get(f, 0.0) or 0.0) for f in RHYTHM_FLAGS]

    vector = [age_norm, sex] + device_onehot + continuous + form_flags + [comorbidity] + rhythm_flags
    return np.array(vector, dtype=np.float32)
