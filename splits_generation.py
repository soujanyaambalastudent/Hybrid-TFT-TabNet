# ============================================================
#  splits_generation.py
#  Generates the exact, patient-level record-ID splits for the
#  5-fold CV and held-out test set, matching Section 6 of the
#  manuscript and the standard PTB-XL benchmark protocol
#  (Wagner et al., 2020):
#     - strat_fold == 10  -> held-out TEST partition
#     - strat_fold in 1-9 -> pool for the 5-fold stratified CV
#
#  Within the 1-9 pool, a fresh patient-grouped stratified 5-fold
#  split (StratifiedGroupKFold, grouped by patient_id) is run with
#  a fixed seed, so no patient appears in more than one CV fold --
#  this is enforced explicitly here even though PTB-XL's own
#  strat_fold assignment is already patient-disjoint, as a second,
#  independent safeguard against leakage.
#
#  Input:  ischemic_labels.csv (from label_derivation.py)
#  Output: splits.json, containing record IDs for:
#            test
#            fold{1..5}_train / fold{1..5}_val   (within the CV pool)
#
#  Run as a single Colab cell after label_derivation.py.
#  Requires: pandas, scikit-learn, and seed_utils.py in the same directory.
# ============================================================

import json
import os

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from seed_utils import fix_all_seeds, SEED

# PTBXL_META_DIR overrides all of these, so this script finds the exact
# same working directory label_derivation.py just wrote ischemic_labels.csv
# to (see that file's PTBXL_META_DIR env var).
_META_DIR_OVERRIDE = os.environ.get("PTBXL_META_DIR")
LABELS_CSV_CANDIDATES = [
    os.path.join(_META_DIR_OVERRIDE, "ischemic_labels.csv") if _META_DIR_OVERRIDE else None,
    "/content/ptbxl_meta/ischemic_labels.csv",
    "/kaggle/working/ptbxl_meta/ischemic_labels.csv",
    "./ptbxl_meta/ischemic_labels.csv",
]
LABELS_CSV_CANDIDATES = [p for p in LABELS_CSV_CANDIDATES if p]
OUTPUT_PATH_CANDIDATES = [
    _META_DIR_OVERRIDE,
    "/content/ptbxl_meta",
    "/kaggle/working/ptbxl_meta",
    "./ptbxl_meta",
]
OUTPUT_PATH_CANDIDATES = [p for p in OUTPUT_PATH_CANDIDATES if p]
N_CV_FOLDS = 5


def _find_labels_csv():
    for path in LABELS_CSV_CANDIDATES:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        "ischemic_labels.csv not found. Run label_derivation.py first, "
        "in the same session / working directory (or set PTBXL_META_DIR "
        "to the directory it wrote to)."
    )


def _output_dir():
    for d in OUTPUT_PATH_CANDIDATES:
        if os.path.isdir(d):
            return d
    d = OUTPUT_PATH_CANDIDATES[-1]
    os.makedirs(d, exist_ok=True)
    return d


def main():
    fix_all_seeds(SEED)

    labels_path = _find_labels_csv()
    df = pd.read_csv(labels_path)
    print(f"Loaded {len(df)} labeled records from {labels_path}")

    test_df = df[df["strat_fold"] == 10].copy()
    pool_df = df[df["strat_fold"] != 10].copy()
    print(f"Test partition (strat_fold==10): {len(test_df)} records")
    print(f"CV pool (strat_fold in 1..9):    {len(pool_df)} records")

    sgkf = StratifiedGroupKFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=SEED)

    splits = {
        "seed": SEED,
        "test": sorted(test_df["record_id"].tolist()),
        "folds": {},
    }

    X = pool_df["record_id"].values
    y = pool_df["ischemic_label"].values
    groups = pool_df["patient_id"].values

    for fold_idx, (train_idx, val_idx) in enumerate(sgkf.split(X, y, groups), start=1):
        train_ids = sorted(X[train_idx].tolist())
        val_ids = sorted(X[val_idx].tolist())

        # Patient-leakage sanity check: no patient should appear in both
        train_patients = set(groups[train_idx])
        val_patients = set(groups[val_idx])
        overlap = train_patients & val_patients
        assert not overlap, f"Fold {fold_idx}: {len(overlap)} patients leak across train/val!"

        splits["folds"][f"fold{fold_idx}"] = {"train": train_ids, "val": val_ids}
        pos_rate_train = pool_df.set_index("record_id").loc[train_ids, "ischemic_label"].mean()
        pos_rate_val = pool_df.set_index("record_id").loc[val_ids, "ischemic_label"].mean()
        print(f"  Fold {fold_idx}: train={len(train_ids)} (pos rate {pos_rate_train:.3f})  "
              f"val={len(val_ids)} (pos rate {pos_rate_val:.3f})")

    out_dir = _output_dir()
    out_path = os.path.join(out_dir, "splits.json")
    with open(out_path, "w") as f:
        json.dump(splits, f, indent=2)
    print(f"\nSaved -> {out_path}")

    # Also write plain-text ID lists per fold, since reviewers asked for
    # split IDs explicitly and plain .txt files are the easiest format
    # for anyone to inspect or diff without parsing JSON.
    for fold_idx in range(1, N_CV_FOLDS + 1):
        fold = splits["folds"][f"fold{fold_idx}"]
        for split_name in ("train", "val"):
            txt_path = os.path.join(out_dir, f"fold{fold_idx}_{split_name}.txt")
            with open(txt_path, "w") as f:
                f.write("\n".join(str(r) for r in fold[split_name]))
    with open(os.path.join(out_dir, "test_ids.txt"), "w") as f:
        f.write("\n".join(str(r) for r in splits["test"]))
    print("Also wrote fold{1..5}_train.txt / fold{1..5}_val.txt / test_ids.txt")

    return splits


if __name__ == "__main__":
    splits = main()
