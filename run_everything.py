# ============================================================
#  run_everything.py
#  ONE script that runs the whole DATA-PREPARATION pipeline in order:
#    install packages -> label derivation -> splits -> data caching
#    -> normalization stats
#  Paste this into ONE Colab cell and run it. No other steps needed
#  except making sure the repo/ folder (from the zip) is already
#  unzipped and you've cd'd into it first.
#
#  NOTE: this covers steps 1-4 of the README's pipeline (everything
#  needed BEFORE training). It deliberately does NOT run train.py /
#  evaluate.py / hyperparameter_sweep.py / feature_importance.py /
#  significance_testing.py, since those require a GPU and hours-to-
#  days of compute (50 epochs x 5 folds x 9 models) -- run those
#  individually, per the exact commands in README.md's
#  "Reproducing Tables 4-9" section.
#
#  If you already have a local copy of the PTB-XL release, set
#  PTBXL_DATA_ROOT before running this (see label_derivation.py):
#    PTBXL_DATA_ROOT=/path/to/ptb-xl python run_everything.py
#  Without it, this will auto-detect a Kaggle/Drive-attached copy, or
#  download the (small) metadata CSVs and then the full ~2GB waveform
#  archive from PhysioNet as a last resort.
# ============================================================

import subprocess
import sys

print("=" * 60)
print("STEP 0: Installing required packages")
print("=" * 60)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"])
print("Packages installed.\n")

print("=" * 60)
print("STEP 1: Label derivation")
print("=" * 60)
subprocess.run([sys.executable, "label_derivation.py"], check=True)
print()

print("=" * 60)
print("STEP 2: Split generation")
print("=" * 60)
subprocess.run([sys.executable, "splits_generation.py"], check=True)
print()

print("=" * 60)
print("STEP 3: Data caching (download + preprocess + feature extraction)")
print("This is the slow step. It will keep printing progress.")
print("=" * 60)
subprocess.run([sys.executable, "build_cache.py"], check=True)
print()

print("=" * 60)
print("STEP 4: Normalization statistics (training-pool only, no leakage)")
print("=" * 60)
subprocess.run([sys.executable, "compute_normalization_stats.py"], check=True)
print()

print("=" * 60)
print("DATA PREPARATION DONE. Next: train.py (see README.md).")
print("=" * 60)
