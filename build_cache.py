# ============================================================
#  build_cache.py
#  ONE script that does everything needed before training:
#    1. Downloads each record's raw ECG waveform from PTB-XL
#    2. Filters it (preprocessing.py)
#    3. Computes the derived clinical measurements neurokit2
#       (clinical_features.py) plus the demographic/form/rhythm
#       fields already in ptbxl_database.csv
#    4. Saves everything to disk as one compact file per record
#
#  You do not need to understand every line of this file.
#  Just run it (see bottom of this file for the single command)
#  and let it work through all ~21,799 records. This will take
#  a while (expect at least an hour, likely more, since it
#  downloads + processes every single ECG) -- that is normal,
#  not a sign anything is broken. Progress prints every 200
#  records so you can see it moving.
# ============================================================

import ast
import os
import re
import time

import numpy as np
import pandas as pd
import wfdb

from preprocessing import preprocess_ecg_record, FORM_LABEL_FLAGS, RHYTHM_FLAGS
from clinical_features import build_derived_clinical_features

# PTBXL_DATA_ROOT: point this at your local, already-downloaded copy of the
# PTB-XL release (containing records500/, ptbxl_database.csv, etc.) to run
# entirely offline -- see label_derivation.py for the same convention.
PTBXL_DATA_ROOT = os.environ.get("PTBXL_DATA_ROOT")

PTBXL_META_DIR = os.environ.get("PTBXL_META_DIR") or (
    "/content/ptbxl_meta" if os.path.exists("/content") else
    ("/kaggle/working/ptbxl_meta" if os.path.exists("/kaggle") else "./ptbxl_meta"))
CACHE_DIR = os.environ.get("PTBXL_CACHE_DIR") or (
    "/content/ptbxl_cache" if os.path.exists("/content") else
    ("/kaggle/working/ptbxl_cache" if os.path.exists("/kaggle") else "./ptbxl_cache"))
WAVEFORM_LOCAL_DIR = os.environ.get("PTBXL_WAVEFORM_DIR") or (
    "/content/ptbxl_raw" if os.path.exists("/content") else
    ("/kaggle/working/ptbxl_raw" if os.path.exists("/kaggle") else "./ptbxl_raw"))
PHYSIONET_RECORDS500_URL = "https://physionet.org/files/ptb-xl/1.0.3/records500/"
LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]


def _autodetect_records500():
    """Searches common locations (an explicit PTBXL_DATA_ROOT, Kaggle
    attached datasets, Drive) for an existing, already-complete records500
    folder, checking it actually has enough files before trusting it (a
    folder existing is not enough -- the Drive copy earlier looked fine by
    name but was only 17% complete).
    Returns the path if found and apparently complete, else None."""
    EXPECTED_MIN_TOTAL_FILES = 40000
    search_roots = [PTBXL_DATA_ROOT, "/kaggle/input", "/content/drive/MyDrive"]
    for root in search_roots:
        if not root:
            continue
        if not os.path.isdir(root):
            continue
        import glob
        for candidate in glob.glob(os.path.join(root, "**", "records500"), recursive=True):
            total = sum(len(files) for _, _, files in os.walk(candidate))
            if total >= EXPECTED_MIN_TOTAL_FILES:
                print(f"Found a complete records500/ at {candidate} ({total} files) -- using it directly.")
                return candidate
            else:
                print(f"Found records500/ at {candidate} but it's incomplete ({total} files) -- skipping it.")
    return None


def bulk_download_waveforms():
    """First tries to find an already-complete records500/ locally
    (Kaggle attached dataset, or Drive), verified by file count -- not
    just by folder name. Only downloads from PhysioNet as a last resort,
    and verifies THAT download is complete too before proceeding."""
    global WAVEFORM_LOCAL_DIR

    detected = _autodetect_records500()
    if detected is not None:
        WAVEFORM_LOCAL_DIR = os.path.dirname(detected)
        return

    import subprocess

    dest_records500 = os.path.join(WAVEFORM_LOCAL_DIR, "records500")
    EXPECTED_MIN_TOTAL_FILES = 40000

    def _count_files():
        if not os.path.isdir(dest_records500):
            return 0
        total = 0
        for _, _, files in os.walk(dest_records500):
            total += len(files)
        return total

    existing = _count_files()
    if existing >= EXPECTED_MIN_TOTAL_FILES:
        print(f"records500/ already complete ({existing} files), skipping download.")
        return

    if existing > 0:
        print(f"Found an incomplete previous download ({existing} files) -- "
              f"resuming/completing it.")

    os.makedirs(WAVEFORM_LOCAL_DIR, exist_ok=True)
    print("No local dataset found -- downloading records500/ directly from "
          "PhysioNet (one-time, ~2GB). This will take a while -- that's expected.")
    cmd = [
        "wget", "-r", "-N", "-c", "-np", "-nH", "--cut-dirs=3", "-q",
        "-P", WAVEFORM_LOCAL_DIR,
        PHYSIONET_RECORDS500_URL,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("wget stderr (last 2000 chars):", result.stderr[-2000:])
        raise RuntimeError("Download failed -- see stderr above.")

    final_count = _count_files()
    print(f"Download finished: {final_count} files present.")
    if final_count < EXPECTED_MIN_TOTAL_FILES:
        raise RuntimeError(
            f"Download appears INCOMPLETE: only {final_count} files found, "
            f"expected at least {EXPECTED_MIN_TOTAL_FILES}. Re-run this "
            f"script to resume (wget -c resumes rather than restarting)."
        )
    print("Download verified complete.")


_DEVICE_FAMILY_RE = re.compile(r"^([A-Za-z]+-?\d+)")


def canonicalize_device(raw_device):
    """PTB-XL's raw `device` field has firmware/revision-suffixed variants
    of the same physical recorder (e.g. 'AT-6 C 5.0', 'AT-6 C 5.3', 'AT-6
    C 5.6' are all the same AT-6 unit at different firmware versions) --
    verified against the actual metadata: 11 distinct raw strings collapse
    to exactly 4 canonical device families (AT-6, AT-60, CS-12, CS100).

    This matters for correctness, not just tidiness: Section 3.3 specifies
    a 25-dimensional clinical feature vector, and 2(age,sex) + n_devices +
    6(continuous) + 6(form) + 1(comorbidity) + 6(rhythm) only equals 25
    when n_devices=4. One-hot encoding the 11 RAW device strings (as an
    earlier version of this function did) would silently produce a
    32-dimensional vector instead, inconsistent with Table 3/Section 3.3
    and with feature_importance.py's expected vector length."""
    if raw_device is None or (isinstance(raw_device, float) and np.isnan(raw_device)):
        return "UNKNOWN"
    match = _DEVICE_FAMILY_RE.match(str(raw_device).strip())
    return match.group(1) if match else str(raw_device).strip()


# PTB-XL's own `heart_axis` field is CATEGORICAL (verified against the real
# metadata: 8 distinct string values -- 'LAD','ALAD','RAD','AXR','MID',
# 'ARAD','AXL','SAG' -- there is no numeric axis column anywhere in
# ptbxl_database.csv). Section 3.3 lists "frontal axis" as one of six
# CONTINUOUS measurements, so it must be converted to a number to occupy
# that slot -- feeding the raw string through would crash the very first
# attempt to compute mean/std on it (confirmed: this crashed
# compute_normalization_stats.py after successfully processing 18,000+
# records, wasting the time already spent on those, until fixed here).
#
# Mapped to the midpoint degree of each category's standard frontal QRS-
# axis range (normal axis range -30..+90 deg, left/right axis deviation
# beyond that, per standard cardiology convention). This specific mapping
# is a disclosed, practical convention -- PTB-XL's own codebook was not
# available to verify the exact intended degree ranges for the four
# non-standard labels (ALAD/ARAD/AXL/AXR, which read as intermediate/
# extreme gradations beyond the textbook LAD/RAD/normal split) -- so this
# should be treated as provisional and verified before being reported as
# an authoritative clinical measurement.
HEART_AXIS_TO_DEGREES = {
    "MID": 30,      # normal/mid axis, textbook range -30..+90, using midpoint
    "AXL": -15,     # borderline left, between normal and LAD
    "LAD": -60,     # left axis deviation, textbook range -30..-90
    "ALAD": -135,   # abnormal/extreme left axis deviation
    "AXR": 105,     # borderline right, between normal and RAD
    "RAD": 135,     # right axis deviation, textbook range +90..+180
    "ARAD": 165,    # abnormal/extreme right axis deviation
    "SAG": 180,     # sagittal / indeterminate axis
}


def heart_axis_to_degrees(raw_axis):
    if raw_axis is None or (isinstance(raw_axis, float) and np.isnan(raw_axis)):
        return np.nan
    return float(HEART_AXIS_TO_DEGREES.get(str(raw_axis).strip(), np.nan))


def _resolve_metadata_csv(filename):
    """Prefers PTBXL_DATA_ROOT (the raw release) for PTB-XL's own metadata
    files, falling back to PTBXL_META_DIR -- this repository intentionally
    does NOT keep its own copies of ptbxl_database.csv/scp_statements.csv
    committed under artifacts/ (only OUR derived outputs, e.g.
    ischemic_labels.csv, live there), so build_cache.py must be able to
    read PTB-XL's own files straight from the local release, the same way
    label_derivation.py already does."""
    if PTBXL_DATA_ROOT:
        candidate = os.path.join(PTBXL_DATA_ROOT, filename)
        if os.path.exists(candidate):
            return candidate
    return os.path.join(PTBXL_META_DIR, filename)


def load_ptbxl_full_metadata():
    path = _resolve_metadata_csv("ptbxl_database.csv")
    df = pd.read_csv(path, index_col="ecg_id")
    df["scp_codes"] = df["scp_codes"].apply(ast.literal_eval)
    return df


def load_labels():
    path = os.path.join(PTBXL_META_DIR, "ischemic_labels.csv")
    return pd.read_csv(path).set_index("record_id")


def fetch_raw_waveform(filename_hr):
    """Reads the .hea/.dat pair from your Google Drive copy of PTB-XL.
    filename_hr looks like 'records500/00000/00001_hr' in ptbxl_database.csv,
    which matches the folder structure directly under WAVEFORM_LOCAL_DIR."""
    record_path = os.path.join(WAVEFORM_LOCAL_DIR, filename_hr)
    record = wfdb.rdrecord(record_path)
    return record.p_signal.T  # (12, N), in the order record.sig_name


def derive_form_and_rhythm_flags(scp_codes_dict, scp_statements_df):
    """Builds the 6 binary form-label flags and 6 binary rhythm flags
    from the record's SCP codes, using ONLY the form/rhythm layer
    (never the diagnostic layer, per Section III.D leakage analysis)."""
    codes_present = set(scp_codes_dict.keys())

    form_map = {
        "ST-elevation": {"STE"},          # adjust to your exact form-code set if different
        "T-abnormality": {"NT_", "ANTAB", "INVT"},
        "conduction-defect": {"IVCD", "LAFB", "LPFB"},
        "LVH": {"LVH"},
        "RBBB": {"CRBBB", "IRBBB"},
        "LBBB": {"CLBBB", "ILBBB"},
    }
    rhythm_map = {
        "sinus-rhythm": {"SR"},
        "atrial-fibrillation": {"AFIB"},
        "SVT": {"SVTAC", "PSVT"},
        "junctional-rhythm": {"SVARR", "AJR"},
        "pacemaker": {"PACE"},
        "other-rhythm": set(),  # true if none of the above matched
    }

    flags = {}
    any_matched = False
    for name, codes in form_map.items():
        flags[name] = int(bool(codes_present & codes))
    for name, codes in rhythm_map.items():
        if name == "other-rhythm":
            continue
        matched = bool(codes_present & codes)
        flags[name] = int(matched)
        any_matched = any_matched or matched
    flags["other-rhythm"] = int(not any_matched)
    return flags


def main():
    os.makedirs(CACHE_DIR, exist_ok=True)

    bulk_download_waveforms()  # downloads directly from PhysioNet, verified complete

    print("Loading metadata...")
    meta_df = load_ptbxl_full_metadata()
    labels_df = load_labels()
    scp_path = _resolve_metadata_csv("scp_statements.csv")
    scp_statements_df = pd.read_csv(scp_path, index_col=0)

    record_ids = labels_df.index.tolist()
    total = len(record_ids)
    print(f"Processing {total} records. This will take a while -- that's expected.")

    device_categories = sorted(set(canonicalize_device(d) for d in meta_df["device"].dropna()))
    print(f"Found {len(device_categories)} distinct recording device FAMILIES "
          f"(after canonicalizing firmware/revision suffixes): {device_categories}")

    failures = []
    start = time.time()

    for i, record_id in enumerate(record_ids, start=1):
        out_path = os.path.join(CACHE_DIR, f"{record_id}.npz")
        if os.path.exists(out_path):
            continue  # already cached from a previous (possibly interrupted) run

        try:
            row = meta_df.loc[record_id]
            raw = fetch_raw_waveform(row["filename_hr"])
            filtered = preprocess_ecg_record(raw, fs=500)  # (12, 5000), NOT z-score normalized yet

            derived = build_derived_clinical_features(filtered, LEAD_NAMES, fs=500)
            flags = derive_form_and_rhythm_flags(row["scp_codes"], scp_statements_df)

            clinical_raw = {
                "age": row.get("age", np.nan),
                "sex": row.get("sex", np.nan),
                "device": canonicalize_device(row.get("device")),
                "heart_rate": derived["heart_rate"],
                "pr_interval": derived["pr_interval_ms"],
                "qrs_duration": derived["qrs_duration_ms"],
                "qtc_interval": derived["qtc_ms"],
                "frontal_axis": heart_axis_to_degrees(row.get("heart_axis")),
                "sokolow_lyon": derived["sokolow_lyon"],
                "comorbidity_score": np.nan,  # not in PTB-XL directly; see README note
                **{f: flags[f] for f in FORM_LABEL_FLAGS},
                **{f: flags[f] for f in RHYTHM_FLAGS},
            }

            np.savez_compressed(
                out_path,
                ecg_filtered=filtered.astype(np.float32),
                clinical_raw=np.array([clinical_raw], dtype=object),
                label=labels_df.loc[record_id, "ischemic_label"],
            )
        except Exception as e:
            failures.append((record_id, str(e)))

        if i % 200 == 0 or i == total:
            elapsed = time.time() - start
            rate = i / elapsed
            remaining = (total - i) / rate if rate > 0 else 0
            print(f"  {i}/{total} done  ({rate:.1f} records/sec, "
                  f"~{remaining/60:.0f} min remaining, {len(failures)} failures so far)")

    print(f"\nFinished. {total - len(failures)} succeeded, {len(failures)} failed.")
    if failures:
        fail_path = os.path.join(CACHE_DIR, "_failures.csv")
        pd.DataFrame(failures, columns=["record_id", "error"]).to_csv(fail_path, index=False)
        print(f"Failure details saved to {fail_path} -- share this file if you need help "
              f"debugging a batch of failures.")

    print(f"\nCached files are in: {CACHE_DIR}")
    print("Next: run compute_normalization_stats.py, then training can begin.")


if __name__ == "__main__":
    main()
