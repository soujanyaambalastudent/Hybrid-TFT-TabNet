# ============================================================
#  feature_importance.py
#  Reproduces Table 7 (Section 6.6): TabNet attention masks, averaged
#  over all N_steps=5 decision steps and all held-out test-set records,
#  further averaged across the 5 CV fold models, then normalized so
#  all 25 feature-importance scores sum to 1.0.
#
#  Until now, models/tabnet_branch.py's get_attention_masks() existed
#  as a primitive hook but no script anywhere in this repository
#  actually called it, aggregated across folds/records, or normalized
#  the result -- so Table 7 had no reproduction path at all.
#
#  Usage (after training all 5 folds of the proposed model):
#    python feature_importance.py --config configs/proposed.json \
#        --splits ./artifacts/splits.json --labels ./artifacts/ischemic_labels.csv \
#        --checkpoint_dir ./checkpoints
# ============================================================

import argparse
import json

import numpy as np
import torch

from preprocessing import FORM_LABEL_FLAGS, RHYTHM_FLAGS
from train import build_model, ECGClinicalDataset

N_FOLDS = 5

CONTINUOUS_ORDER = ["heart_rate", "pr_interval", "qrs_duration",
                     "qtc_interval", "frontal_axis", "sokolow_lyon"]
CONTINUOUS_CATEGORY = {
    "heart_rate": "Rate", "pr_interval": "ECG Interval", "qrs_duration": "ECG Interval",
    "qtc_interval": "ECG Interval", "frontal_axis": "ECG Morphology", "sokolow_lyon": "Voltage",
}


def build_feature_names(device_categories):
    """Reconstructs the ordered (name, category) list matching
    preprocessing.build_clinical_feature_vector()'s exact concatenation
    order: [age, sex] + device_onehot + continuous(6) + form_flags(6)
    + [comorbidity_score] + rhythm_flags(6)."""
    names = [("Patient age (normalized)", "Demographic"),
             ("Biological sex", "Demographic")]
    names += [(f"Recording device: {d}", "Device") for d in device_categories]
    names += [(f, CONTINUOUS_CATEGORY[f]) for f in CONTINUOUS_ORDER]
    names += [(f"{f} indicator", "Form label") for f in FORM_LABEL_FLAGS]
    names += [("Co-morbidity severity score", "Clinical score")]
    names += [(f"{f} flag", "Rhythm") for f in RHYTHM_FLAGS]
    return names


def compute_fold_importance(model_name, config, fold_num, splits, labels_by_id,
                             get_ecg_tensor, get_clinical_vector, checkpoint_dir,
                             device="cpu"):
    """Loads one fold's trained checkpoint and returns the (25,)
    per-feature importance vector, averaged over N_steps decision
    steps and over every record in the held-out test partition."""
    model = build_model(model_name, config).to(device)
    ckpt_path = f"{checkpoint_dir}/{model_name}_fold{fold_num}_best.pt"
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    test_ids = splits["test"]
    test_ds = ECGClinicalDataset(test_ids, labels_by_id, get_ecg_tensor, get_clinical_vector)
    loader = torch.utils.data.DataLoader(test_ds, batch_size=256, shuffle=False)

    per_record_importance = []
    with torch.no_grad():
        for _, clinical, _ in loader:
            clinical = clinical.to(device)
            masks = model.tabnet.get_attention_masks(clinical)  # list of (B, d) per decision step
            stacked = torch.stack(masks, dim=0)                  # (N_steps, B, d)
            step_avg = stacked.mean(dim=0)                        # (B, d) -- Eq. (6) in the manuscript
            per_record_importance.append(step_avg.cpu().numpy())

    all_records = np.concatenate(per_record_importance, axis=0)  # (n_test, d)
    return all_records.mean(axis=0)  # (d,)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="proposed")
    parser.add_argument("--config", required=True)
    parser.add_argument("--splits", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--checkpoint_dir", default="./checkpoints")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", default="./artifacts/table7_feature_importance.csv")
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)
    with open(args.splits) as f:
        splits = json.load(f)

    import pandas as pd
    labels_df = pd.read_csv(args.labels)
    labels_by_id = dict(zip(labels_df["record_id"], labels_df["ischemic_label"]))

    from data_cache import get_ecg_tensor, get_clinical_vector, _DEVICE_CATEGORIES

    fold_importances = []
    for fold_num in range(1, N_FOLDS + 1):
        imp = compute_fold_importance(
            args.model, config, fold_num, splits, labels_by_id,
            get_ecg_tensor, get_clinical_vector, args.checkpoint_dir, device=args.device,
        )
        fold_importances.append(imp)
        print(f"Fold {fold_num}: importance vector computed over "
              f"{len(splits['test'])} test records.")

    mean_importance = np.mean(fold_importances, axis=0)
    normalized = mean_importance / mean_importance.sum()  # sum to 1.000, per Section 6.6

    feature_names = build_feature_names(_DEVICE_CATEGORIES)
    if len(feature_names) != len(normalized):
        raise ValueError(
            f"Feature-name list length ({len(feature_names)}) doesn't match the "
            f"clinical vector's actual dimensionality ({len(normalized)}) -- "
            f"check device_categories in normalization_stats.json."
        )

    order = np.argsort(-normalized)
    rows = []
    for rank, idx in enumerate(order, start=1):
        name, category = feature_names[idx]
        rows.append({"rank": rank, "feature": name, "category": category,
                      "importance_score": float(normalized[idx])})

    result_df = pd.DataFrame(rows)
    print("\n" + "=" * 58)
    print("  TABLE 7 -- TabNet Clinical Feature Ranking (reproduced)")
    print("=" * 58)
    print(result_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nSum of all importance scores: {normalized.sum():.6f} (should be 1.000000)")

    import os
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    result_df.to_csv(args.out, index=False)
    print(f"\nSaved -> {args.out}")


if __name__ == "__main__":
    main()
