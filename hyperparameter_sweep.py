# ============================================================
#  hyperparameter_sweep.py
#  Reproduces Table 8 (Section 6.7): trains the proposed model with
#  each listed value of each of the four swept hyperparameters (LSTM
#  encoder layers, attention heads, TabNet N_steps, fusion MLP
#  dimension), holding the other three at the manuscript's selected
#  configuration, and logs the resulting validation AUC-ROC for each.
#
#  This directly closes the gap the manuscript and Response to
#  Reviewers (3rd reviewer comment 16 / 1st reviewer comment 9) both
#  explicitly disclose: "individual per-value AUC scores from this
#  sweep were not retained in a form suitable for independent
#  verification." Running this script produces exactly that missing,
#  independently-verifiable, per-configuration log.
#
#  For compute-budget reasons this sweeps on a SINGLE CV fold (fold 1)
#  rather than the full 5-fold procedure used for Tables 4/6 -- Table
#  8's own purpose is relative comparison across configurations, not a
#  final reported metric, so a single-fold proxy is a standard and
#  defensible hyperparameter-search practice. State this explicitly
#  if/when Table 8 is regenerated from this script's output.
#
#  Usage:
#    python hyperparameter_sweep.py --splits ./artifacts/splits.json \
#        --labels ./artifacts/ischemic_labels.csv
# ============================================================

import argparse
import copy
import json

from train import build_model, train_one_fold, log_fold_metrics
from data_cache import get_ecg_tensor, get_clinical_vector

# The manuscript's selected configuration (Table 8's "Selected" column),
# used as the fixed baseline while one hyperparameter at a time is varied.
SELECTED = {
    "lstm_layers": 2,
    "attn_heads": 8,
    "tabnet_n_steps": 5,
    "fusion_mlp_dim": 128,
}

SWEEP_GRID = {
    "lstm_layers": [1, 2, 3],
    "attn_heads": [4, 8, 16],
    "tabnet_n_steps": [3, 5, 7],
    "fusion_mlp_dim": [64, 128, 256],
}


def build_config(base_config, lstm_layers, attn_heads, tabnet_n_steps, fusion_mlp_dim):
    cfg = copy.deepcopy(base_config)
    cfg["architecture"]["ecg_branch"]["lstm_layers"] = lstm_layers
    cfg["architecture"]["ecg_branch"]["attn_heads"] = attn_heads
    cfg["architecture"]["clinical_branch"]["n_steps"] = tabnet_n_steps
    cfg["architecture"]["fusion_hidden_dim"] = fusion_mlp_dim
    return cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_config", default="configs/proposed.json")
    parser.add_argument("--splits", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--fold", type=int, default=1,
                         help="Single CV fold used for the sweep (see module docstring).")
    parser.add_argument("--max_epochs", type=int, default=None,
                         help="Override max_epochs for the sweep runs (e.g. a smaller "
                              "budget than the full 50, to keep sweep cost manageable). "
                              "Defaults to the base config's own max_epochs.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--checkpoint_dir", default="./checkpoints/sweep")
    parser.add_argument("--metrics_log", default="./artifacts/table8_sweep.csv")
    args = parser.parse_args()

    with open(args.base_config) as f:
        base_config = json.load(f)
    if args.max_epochs is not None:
        base_config["training"]["max_epochs"] = args.max_epochs

    with open(args.splits) as f:
        splits = json.load(f)

    import pandas as pd
    labels_df = pd.read_csv(args.labels)
    labels_by_id = dict(zip(labels_df["record_id"], labels_df["ischemic_label"]))

    for hp_name, values in SWEEP_GRID.items():
        for value in values:
            kwargs = dict(SELECTED)
            kwargs[hp_name] = value
            config = build_config(base_config, **kwargs)

            run_name = f"proposed__{hp_name}={value}"
            print(f"\n=== Sweeping {hp_name}={value} "
                  f"(others held at {SELECTED}) ===")

            best_val_auc, best_val_metrics = train_one_fold(
                "proposed", config, args.fold, splits, labels_by_id,
                get_ecg_tensor, get_clinical_vector,
                device=args.device, checkpoint_dir=args.checkpoint_dir,
            )
            print(f"  {run_name}: val_auc={best_val_auc:.4f}")

            # Reuses train.py's CSV logger, tagging the "model" field with
            # the swept hyperparameter and value so rows are distinguishable
            # from the main 5-fold Table 4 runs sharing the same log file.
            log_fold_metrics(args.metrics_log, run_name, args.fold, best_val_metrics)

    print(f"\nAll sweep runs logged -> {args.metrics_log}")
    print("Table 8's 'Values Tested' column is SWEEP_GRID above; the 'Selected' "
          "column is SELECTED; per-value AUC is now in the metrics log by design, "
          "closing the gap disclosed in the manuscript / Response to Reviewers.")


if __name__ == "__main__":
    main()
