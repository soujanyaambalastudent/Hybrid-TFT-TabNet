# ============================================================
#  compute_normalization_stats.py
#  Computes the per-lead ECG mean/std and per-feature clinical
#  mean/std, using ONLY the training pool (strat_fold 1-9) --
#  never the test set, to avoid leakage. Run this once, after
#  build_cache.py finishes.
# ============================================================

import json
import os

import numpy as np
import pandas as pd

PTBXL_META_DIR = os.environ.get("PTBXL_META_DIR") or (
    "/content/ptbxl_meta" if os.path.exists("/content") else
    ("/kaggle/working/ptbxl_meta" if os.path.exists("/kaggle") else "./ptbxl_meta"))
CACHE_DIR = os.environ.get("PTBXL_CACHE_DIR") or (
    "/content/ptbxl_cache" if os.path.exists("/content") else
    ("/kaggle/working/ptbxl_cache" if os.path.exists("/kaggle") else "./ptbxl_cache"))

CONTINUOUS_FIELDS = ["age", "heart_rate", "pr_interval", "qrs_duration",
                      "qtc_interval", "frontal_axis", "sokolow_lyon"]


def main():
    labels_df = pd.read_csv(os.path.join(PTBXL_META_DIR, "ischemic_labels.csv"))
    train_pool_ids = labels_df[labels_df["strat_fold"] != 10]["record_id"].tolist()

    print(f"Computing stats from {len(train_pool_ids)} training-pool records...")

    ecg_sum = None
    ecg_sq_sum = None
    n_ecg = 0
    continuous_values = {f: [] for f in CONTINUOUS_FIELDS}
    devices_seen = set()
    missing_continuous = {f: 0 for f in CONTINUOUS_FIELDS}

    for i, record_id in enumerate(train_pool_ids, start=1):
        path = os.path.join(CACHE_DIR, f"{record_id}.npz")
        if not os.path.exists(path):
            continue
        data = np.load(path, allow_pickle=True)
        ecg = data["ecg_filtered"]  # (12, 5000)

        if ecg_sum is None:
            ecg_sum = np.zeros_like(ecg, dtype=np.float64)
            ecg_sq_sum = np.zeros_like(ecg, dtype=np.float64)
        ecg_sum += ecg
        ecg_sq_sum += ecg ** 2
        n_ecg += 1

        clinical_raw = data["clinical_raw"][0]
        devices_seen.add(clinical_raw.get("device", "UNKNOWN"))
        for f in CONTINUOUS_FIELDS:
            v = clinical_raw.get(f)
            if v is None or (isinstance(v, float) and np.isnan(v)):
                missing_continuous[f] += 1
            else:
                continuous_values[f].append(v)

        if i % 2000 == 0:
            print(f"  {i}/{len(train_pool_ids)} processed...")

    ecg_mean = (ecg_sum / n_ecg)
    ecg_var = (ecg_sq_sum / n_ecg) - ecg_mean ** 2
    ecg_std = np.sqrt(np.clip(ecg_var, 1e-12, None))

    stats = {
        "n_training_records_used": n_ecg,
        "ecg_lead_mean": ecg_mean.mean(axis=1).tolist(),  # per-lead scalar mean
        "ecg_lead_std": ecg_std.mean(axis=1).tolist(),
        "continuous_features": {
            f: {"mean": float(np.mean(vals)), "std": float(np.std(vals) + 1e-8),
                "n_missing": missing_continuous[f]}
            for f, vals in continuous_values.items() if len(vals) > 0
        },
        "device_categories": sorted(devices_seen),
    }

    for f, info in stats["continuous_features"].items():
        if info["n_missing"] > 0:
            print(f"  NOTE: {info['n_missing']} training records missing '{f}' "
                  f"(likely neurokit2 delineation failures on noisy signals) -- "
                  f"these will be median-imputed at load time.")

    out_path = os.path.join(PTBXL_META_DIR, "normalization_stats.json")
    with open(out_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
