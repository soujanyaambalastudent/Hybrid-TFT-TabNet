# ============================================================
#  significance_testing.py
#  Reproduces Section 5.2's exploratory statistical comparison: an
#  exploratory Wilcoxon signed-rank test on per-fold AUC-ROC between
#  the proposed model and each of the seven baselines. Until now, NO
#  script anywhere in this repository actually computed this -- the
#  manuscript reported "p < 0.01" for all seven baselines with no
#  underlying code.
#
#  Reads its input from the CSV that train.py appends one row to
#  per {model, fold} run (see train.py's --metrics_log), so this is
#  computed from the SAME logged numbers Table 4/6/8 use -- not a
#  separate, potentially-inconsistent calculation.
#
#  Usage (after training the proposed model and all seven baselines
#  on all 5 folds, all logging to the same --metrics_log path):
#    python significance_testing.py --metrics_log ./artifacts/cv_metrics.csv
# ============================================================

import argparse

import pandas as pd

from stats_utils import wilcoxon_fold_comparison

PROPOSED_MODEL_NAME = "proposed"
N_EXPECTED_FOLDS = 5


def run_significance_tests(metrics_log_path, metric="auc"):
    df = pd.read_csv(metrics_log_path)

    # Only compare the full, unfrozen proposed model (freeze_tabnet=False)
    # against baselines -- ablation variants (A3-A5) are not part of this
    # comparison, matching Section 5.2's scope ("all seven baselines").
    proposed_rows = df[(df["model"] == PROPOSED_MODEL_NAME) & (~df["freeze_tabnet"])]
    proposed_rows = proposed_rows.sort_values("fold")

    if len(proposed_rows) != N_EXPECTED_FOLDS:
        raise ValueError(
            f"Expected {N_EXPECTED_FOLDS} logged folds for '{PROPOSED_MODEL_NAME}', "
            f"found {len(proposed_rows)}. Train all 5 folds before running this."
        )
    proposed_scores = proposed_rows[metric].to_numpy()

    results = []
    baseline_names = sorted(df.loc[df["model"] != PROPOSED_MODEL_NAME, "model"].unique())

    for baseline in baseline_names:
        baseline_rows = df[df["model"] == baseline].sort_values("fold")
        if len(baseline_rows) != N_EXPECTED_FOLDS:
            print(f"  Skipping '{baseline}': found {len(baseline_rows)}/{N_EXPECTED_FOLDS} "
                  f"logged folds (train all 5 before comparing).")
            continue
        baseline_scores = baseline_rows[metric].to_numpy()

        statistic, p_value = wilcoxon_fold_comparison(proposed_scores, baseline_scores)
        results.append({
            "baseline": baseline,
            "proposed_mean": proposed_scores.mean(),
            "baseline_mean": baseline_scores.mean(),
            "wilcoxon_statistic": statistic,
            "p_value": p_value,
            "significant_at_0.01": p_value < 0.01,
        })

    result_df = pd.DataFrame(results)
    return result_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics_log", default="./artifacts/cv_metrics.csv")
    parser.add_argument("--metric", default="auc", choices=["accuracy", "f1", "auc",
                                                              "sensitivity", "specificity"])
    parser.add_argument("--out", default="./artifacts/significance_results.csv")
    args = parser.parse_args()

    print("=" * 70)
    print("  EXPLORATORY WILCOXON SIGNED-RANK COMPARISON (Section 5.2)")
    print(f"  Metric: {args.metric}  |  n = 5 paired fold observations per comparison")
    print("  CAVEAT (stated explicitly in the manuscript and carried over here):")
    print("  five paired observations is a small sample and this test does not")
    print("  evaluate paired differences on held-out test-set predictions --")
    print("  treat results as suggestive, not confirmatory, evidence.")
    print("=" * 70)

    result_df = run_significance_tests(args.metrics_log, metric=args.metric)
    if result_df.empty:
        print("\nNo baseline has all 5 folds logged yet -- nothing to compare.")
        return

    print()
    print(result_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    import os
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    result_df.to_csv(args.out, index=False)
    print(f"\nSaved -> {args.out}")


if __name__ == "__main__":
    main()
