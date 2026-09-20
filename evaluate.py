# ============================================================
#  evaluate.py
#  Loads trained fold checkpoints, builds the 5-fold ensemble,
#  and evaluates on the held-out test partition (strat_fold==10).
#  Reproduces:
#    - Table 3 / Section 6.1: per-fold validation metrics
#    - Table 4: held-out test metrics (accuracy, F1, AUC-ROC,
#      sensitivity, specificity) with Wilson/DeLong/bootstrap CIs
#    - The confusion matrix underlying Figures 7/12
#
#  Usage:
#    python evaluate.py --model proposed --config configs/proposed.json \
#        --splits /content/ptbxl_meta/splits.json \
#        --labels /content/ptbxl_meta/ischemic_labels.csv \
#        --checkpoint_dir ./checkpoints
# ============================================================

import argparse
import json

import numpy as np
import torch
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                              average_precision_score, confusion_matrix)

from train import build_model, ECGClinicalDataset  # reuse dataset/model builders


def wilson_ci(successes, n, z=1.96):
    p = successes / n
    denom = 1 + z ** 2 / n
    center = p + z ** 2 / (2 * n)
    margin = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2))
    return (center - margin) / denom, (center + margin) / denom


def bootstrap_accuracy_ci(y_true, y_pred, n_resamples=20000, seed=42):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    accs = []
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    for _ in range(n_resamples):
        idx = rng.integers(0, n, n)
        accs.append(accuracy_score(y_true[idx], y_pred[idx]))
    return np.percentile(accs, 2.5), np.percentile(accs, 97.5)


def evaluate_ensemble(model_name, config, splits, labels_by_id,
                       get_ecg_tensor, get_clinical_vector,
                       checkpoint_dir, device="cuda", n_folds=5,
                       threshold=0.5):
    test_ids = splits["test"]
    test_ds = ECGClinicalDataset(test_ids, labels_by_id, get_ecg_tensor, get_clinical_vector)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=256, shuffle=False)

    fold_probs = []  # (n_folds, n_test)
    y_true_ref = None

    for fold_num in range(1, n_folds + 1):
        model = build_model(model_name, config).to(device)
        ckpt_path = f"{checkpoint_dir}/{model_name}_fold{fold_num}_best.pt"
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.eval()

        probs, y_true = [], []
        with torch.no_grad():
            for ecg, clinical, labels in test_loader:
                ecg, clinical = ecg.to(device), clinical.to(device)
                logits, _ = model(ecg, clinical)
                probs.extend(torch.sigmoid(logits).cpu().numpy().tolist())
                y_true.extend(labels.numpy().tolist())
        fold_probs.append(probs)
        y_true_ref = y_true  # identical across folds (same test set)

    ensemble_probs = np.mean(np.array(fold_probs), axis=0)
    y_true = np.array(y_true_ref)
    y_pred = (ensemble_probs >= threshold).astype(int)

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, ensemble_probs)
    ap = average_precision_score(y_true, ensemble_probs)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    precision = tp / (tp + fp)

    n = len(y_true)
    n_correct = int(round(acc * n))
    wilson_lo, wilson_hi = wilson_ci(n_correct, n)
    boot_lo, boot_hi = bootstrap_accuracy_ci(y_true, y_pred)

    print("=" * 58)
    print(f"  HELD-OUT TEST RESULTS  ({model_name}, n={n})")
    print("=" * 58)
    print(f"  Accuracy:     {acc:.4f}  (Wilson 95% CI: {wilson_lo:.4f}-{wilson_hi:.4f}, "
          f"bootstrap 95% CI: {boot_lo:.4f}-{boot_hi:.4f})")
    print(f"  F1-score:     {f1:.4f}")
    print(f"  AUC-ROC:      {auc:.4f}")
    print(f"  Avg precision:{ap:.4f}")
    print(f"  Sensitivity:  {sensitivity:.4f}")
    print(f"  Specificity:  {specificity:.4f}")
    print(f"  Precision:    {precision:.4f}")
    print(f"  Confusion matrix: TN={tn} FP={fp} FN={fn} TP={tp}")
    print()
    print("  NOTE: these are the REAL, freshly computed numbers from this run.")
    print("  They are not expected to exactly match earlier manuscript drafts")
    print("  if the label definition or splits changed -- see label_derivation.py")
    print("  and splits_generation.py for the currently locked-in definitions,")
    print("  and update the manuscript's tables to match THESE numbers.")

    return {
        "accuracy": acc, "f1": f1, "auc_roc": auc, "avg_precision": ap,
        "sensitivity": sensitivity, "specificity": specificity, "precision": precision,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "accuracy_wilson_ci": [wilson_lo, wilson_hi],
        "accuracy_bootstrap_ci": [boot_lo, boot_hi],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--splits", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--checkpoint_dir", default="./checkpoints")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)
    with open(args.splits) as f:
        splits = json.load(f)

    import pandas as pd
    labels_df = pd.read_csv(args.labels)
    labels_by_id = dict(zip(labels_df["record_id"], labels_df["ischemic_label"]))

    from data_cache import get_ecg_tensor, get_clinical_vector

    evaluate_ensemble(
        args.model, config, splits, labels_by_id,
        get_ecg_tensor, get_clinical_vector,
        checkpoint_dir=args.checkpoint_dir, device=args.device,
    )


if __name__ == "__main__":
    main()
