# ============================================================
#  label_derivation.py  (FINAL, locked-in version)
#  Derives Ischemic-positive / Ischemic-negative labels from
#  PTB-XL SCP-ECG diagnostic annotations.
#
#  Definition: full MI diagnostic_class (all infarction-related
#  leaf codes) UNION the ischemia-specific leaf codes within STTC
#  (ANEUR, EL, ISCAL, ISCAN, ISCAS, ISCIL, ISCIN, ISCLA, ISC_).
#  Excludes DIG, LNGQT (not ischemic conditions) and NDT, NST_
#  ("nonspecific" changes, not attributable to ischemia).
#
#  Uses PTB-XL's own pre-assigned patient-level fold column
#  (strat_fold, values 1-10), following the standard PTB-XL
#  benchmark protocol from Wagner et al. (2020): fold 10 reserved
#  as the held-out test partition, folds 1-9 as the pool for
#  5-fold CV.
#
#  This is an intentionally re-derived, from-scratch pipeline
#  (the original undocumented run's code no longer exists). Its
#  output counts are the new source of truth for the repo; the
#  manuscript will be updated to match if they differ from the
#  previously reported figures.
#
#  Run as a single Colab cell. Requires: pandas, ast (stdlib).
#  Downloads only the small PTB-XL metadata CSVs (not the
#  waveform signals, which the preprocessing step handles later).
# ============================================================

import os
import ast
import urllib.request
import glob
import pandas as pd

# ---- Config -------------------------------------------------
PTBXL_BASE_URL = "https://physionet.org/files/ptb-xl/1.0.3/"

# PTBXL_DATA_ROOT: point this at your local, already-downloaded copy of the
# PTB-XL release (the folder containing ptbxl_database.csv, scp_statements.csv,
# records100/, records500/) to run entirely offline, e.g.:
#   PTBXL_DATA_ROOT=/path/to/ptb-xl python label_derivation.py
# If unset, falls back to auto-detecting a Kaggle/Drive-attached copy, and
# only downloads from PhysioNet as a last resort.
PTBXL_DATA_ROOT = os.environ.get("PTBXL_DATA_ROOT")

DATA_DIR = os.environ.get("PTBXL_META_DIR") or (
    "/content/ptbxl_meta" if os.path.exists("/content") else "./ptbxl_meta"
)
os.makedirs(DATA_DIR, exist_ok=True)


def _find_local_metadata_file(filename):
    """Auto-detects a local copy of a PTB-XL metadata file: first under
    PTBXL_DATA_ROOT (an explicit local copy of the release, if set), then
    under a Kaggle attached dataset or a mounted Drive folder, searching a
    few levels deep, so we don't need to download it if it's already
    available locally. Returns None if not found locally."""
    search_roots = [PTBXL_DATA_ROOT, "/kaggle/input", "/content/drive/MyDrive"]
    for root in search_roots:
        if not root or not os.path.isdir(root):
            continue
        matches = glob.glob(os.path.join(root, "**", filename), recursive=True)
        if matches:
            return matches[0]
    return None

# ISCHEMISCH superclass codes -- FINAL, LOCKED-IN definition.
# Clinically scoped to genuinely ischemic/infarction-related SCP-ECG codes:
# the full MI diagnostic_class, plus the ischemia-specific leaf codes within
# STTC. Deliberately EXCLUDES DIG (digitalis effect) and LNGQT (long QT
# syndrome) -- neither is an ischemic condition -- and EXCLUDES NDT/NST_
# ("nonspecific" ST/T changes), since "nonspecific" means not attributable
# to ischemia by definition. This will not necessarily reproduce the exact
# record counts from an earlier, undocumented run -- it is the definition
# being adopted going forward, and the manuscript's reported counts will be
# updated to match this run's verified output.
ISCHEMIC_POSITIVE_CODES = {
    # Full MI diagnostic_class (all infarction-related leaf codes)
    "ALMI", "AMI", "ASMI", "ILMI", "IMI", "INJAL", "INJAS", "INJIL",
    "INJIN", "INJLA", "IPLMI", "IPMI", "LMI", "PMI",
    # Ischemia-specific STTC leaf codes only (excludes DIG, LNGQT, NDT, NST_)
    "ANEUR", "EL", "ISCAL", "ISCAN", "ISCAS", "ISCIL", "ISCIN", "ISCLA", "ISC_",
}

# Manuscript's PREVIOUSLY reported counts (from an earlier, undocumented
# run whose original code no longer exists). Kept here for reference and
# comparison only -- NOT used to force this run's output. The counts from
# THIS run become the new source of truth; the manuscript will be updated
# to match if they differ.
PRIOR_REPORTED_TOTAL = 21837
PRIOR_REPORTED_POSITIVE = 10505
PRIOR_REPORTED_NEGATIVE = 11332
PRIOR_REPORTED_TEST_N = 4061          # strat_fold == 10 (old, unverified figure)
PRIOR_REPORTED_TRAIN_POOL_N = 17776   # strat_fold in 1..9 (old, unverified figure)


def _download_if_missing(filename):
    local_path = os.path.join(DATA_DIR, filename)
    if os.path.exists(local_path):
        return local_path

    found = _find_local_metadata_file(filename)
    if found:
        print(f"Found {filename} locally at {found} -- using it directly, no download needed.")
        import shutil
        shutil.copy(found, local_path)
        return local_path

    url = PTBXL_BASE_URL + filename
    print(f"Downloading {filename} ...")
    urllib.request.urlretrieve(url, local_path)
    return local_path


def load_ptbxl_metadata():
    """Loads ptbxl_database.csv (record-level metadata + scp_codes)
    and scp_statements.csv (code -> diagnostic_class / diagnostic_subclass)."""
    db_path = _download_if_missing("ptbxl_database.csv")
    scp_path = _download_if_missing("scp_statements.csv")

    df = pd.read_csv(db_path, index_col="ecg_id")
    df["scp_codes"] = df["scp_codes"].apply(ast.literal_eval)  # stored as stringified dict

    scp_df = pd.read_csv(scp_path, index_col=0)
    return df, scp_df


def derive_ischemic_label(scp_codes_dict):
    """scp_codes_dict: {'AMI': 100.0, 'NORM': 0.0, ...} for one record.
    Returns 1 (Ischemic-positive) if any key is in the ISCHEMISCH superclass
    codes listed above, else 0 (Ischemic-negative)."""
    codes_present = set(scp_codes_dict.keys())
    return int(len(codes_present & ISCHEMIC_POSITIVE_CODES) > 0)


def main():
    df, scp_df = load_ptbxl_metadata()

    df["ischemic_label"] = df["scp_codes"].apply(derive_ischemic_label)

    # Keep the columns needed downstream: record id, patient id, the
    # official PTB-XL fold assignment (per Wagner et al. 2020: fold 10 =
    # test, folds 1-9 = train/CV pool), and the derived label.
    out = df.reset_index()[["ecg_id", "patient_id", "strat_fold", "ischemic_label"]]
    out = out.rename(columns={"ecg_id": "record_id"})

    n_total = len(out)
    n_pos = int(out["ischemic_label"].sum())
    n_neg = n_total - n_pos
    n_test = int((out["strat_fold"] == 10).sum())
    n_train_pool = n_total - n_test

    print("=" * 58)
    print("  LABEL DERIVATION SUMMARY (final, locked-in definition)")
    print("=" * 58)
    print(f"  Total records:              {n_total}")
    print(f"  Ischemic-positive:          {n_pos}")
    print(f"  Ischemic-negative:          {n_neg}")
    print(f"  strat_fold==10 (test pool): {n_test}")
    print(f"  strat_fold 1-9 (CV pool):   {n_train_pool}")
    print()
    print("  For reference, the manuscript's PRIOR (unverified) figures were:")
    for name, actual, prior in [
        ("total records", n_total, PRIOR_REPORTED_TOTAL),
        ("Ischemic-positive", n_pos, PRIOR_REPORTED_POSITIVE),
        ("Ischemic-negative", n_neg, PRIOR_REPORTED_NEGATIVE),
        ("test partition", n_test, PRIOR_REPORTED_TEST_N),
        ("train/CV pool", n_train_pool, PRIOR_REPORTED_TRAIN_POOL_N),
    ]:
        print(f"    {name}: this run={actual}  |  prior manuscript={prior}")
    print()
    print("  This run's numbers are the ones going into the repo and, if they")
    print("  differ from the prior manuscript figures above, the manuscript's")
    print("  Table 9/10 counts and class-ratio text should be updated to match.")

    out_path = os.path.join(DATA_DIR, "ischemic_labels.csv")
    out.to_csv(out_path, index=False)
    print(f"\nSaved labeled record list -> {out_path}")
    print("Columns: record_id, patient_id, strat_fold, ischemic_label")

    return out


if __name__ == "__main__":
    labels_df = main()
