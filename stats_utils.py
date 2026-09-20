# ============================================================
#  stats_utils.py
#  Statistical-testing utilities referenced in the manuscript but,
#  until now, never actually implemented anywhere in this repository:
#    - DeLong's method for the variance/CI of a single AUC-ROC
#      (Section 5.2 / 6.3: "DeLong 95% CI: 0.9831-0.9893")
#    - An exploratory Wilcoxon signed-rank comparison of per-fold
#      AUC-ROC between the proposed model and a baseline
#      (Section 5.2: "a consistent fold-level advantage over all
#      seven baselines (p < 0.01)")
#
#  Requires: numpy, scipy.
# ============================================================

import numpy as np
from scipy import stats


def _compute_midrank(x):
    """Midranks, as used by DeLong's method (Sun & Xu, 2014 fast
    implementation). x must be a 1-D array."""
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    T2 = np.empty(N, dtype=float)
    T2[J] = T
    return T2


def _fast_delong(pos_scores, neg_scores):
    """Core DeLong covariance computation for ONE set of scores.
    Returns (auc, auc_variance)."""
    m = len(pos_scores)
    n = len(neg_scores)
    combined = np.concatenate([pos_scores, neg_scores])

    tx = _compute_midrank(pos_scores)
    ty = _compute_midrank(neg_scores)
    tz = _compute_midrank(combined)

    v01 = (tz[:m] - tx) / n
    v10 = 1.0 - (tz[m:] - ty) / m

    auc = v01.mean()

    s01 = np.cov(v01)
    s10 = np.cov(v10)
    auc_var = s01 / m + s10 / n
    return auc, float(auc_var)


def delong_auc_ci(y_true, y_scores, alpha=0.05):
    """95% (or 1-alpha) confidence interval for AUC-ROC via DeLong's
    method (DeLong, DeLong & Clarke-Pearson, 1988 -- cited as ref [23]
    in the manuscript). This is what Section 6.3/6.10's "DeLong 95% CI"
    figures are supposed to come from; no prior version of this repo
    actually computed it.

    y_true: array-like of {0,1} labels.
    y_scores: array-like of predicted probabilities / scores.
    Returns (auc, ci_lo, ci_hi).
    """
    y_true = np.asarray(y_true)
    y_scores = np.asarray(y_scores, dtype=float)

    pos_scores = y_scores[y_true == 1]
    neg_scores = y_scores[y_true == 0]
    if len(pos_scores) == 0 or len(neg_scores) == 0:
        raise ValueError("DeLong CI requires both classes to be present.")

    auc, auc_var = _fast_delong(pos_scores, neg_scores)
    se = np.sqrt(max(auc_var, 0.0))
    z = stats.norm.ppf(1 - alpha / 2)
    lo = max(0.0, auc - z * se)
    hi = min(1.0, auc + z * se)
    return auc, lo, hi


def wilcoxon_fold_comparison(proposed_fold_scores, baseline_fold_scores):
    """Exploratory paired Wilcoxon signed-rank test between the
    proposed model's and one baseline's per-fold metric (e.g. AUC-ROC),
    across the 5 CV folds, matching Section 5.2.

    Both arguments are length-5 sequences (one value per CV fold),
    in the SAME fold order. Returns (statistic, p_value).

    Caveat carried over verbatim from the manuscript's own framing
    (Section 5.2, and Response to Reviewers, 3rd reviewer comment 22):
    five paired observations is a small sample and this test does not
    evaluate paired differences on held-out test-set predictions: treat
    the result as suggestive, not confirmatory, evidence of superiority.
    """
    proposed_fold_scores = np.asarray(proposed_fold_scores, dtype=float)
    baseline_fold_scores = np.asarray(baseline_fold_scores, dtype=float)
    if len(proposed_fold_scores) != len(baseline_fold_scores):
        raise ValueError("Both fold-score arrays must be the same length.")
    if np.allclose(proposed_fold_scores, baseline_fold_scores):
        # scipy.stats.wilcoxon raises on all-zero differences; report
        # explicitly rather than letting that exception propagate.
        return 0.0, 1.0
    statistic, p_value = stats.wilcoxon(proposed_fold_scores, baseline_fold_scores)
    return float(statistic), float(p_value)


if __name__ == "__main__":
    # Sanity check against a case with a known closed-form-ish answer:
    # a small synthetic example with well-separated classes.
    rng = np.random.default_rng(0)
    y = np.concatenate([np.ones(50), np.zeros(50)])
    scores = np.concatenate([rng.normal(0.8, 0.15, 50), rng.normal(0.3, 0.15, 50)])
    auc, lo, hi = delong_auc_ci(y, scores)
    print(f"DeLong sanity check: AUC={auc:.4f}  95% CI=({lo:.4f}, {hi:.4f})")

    stat, p = wilcoxon_fold_comparison([0.986, 0.985, 0.987, 0.986, 0.988],
                                        [0.968, 0.965, 0.970, 0.969, 0.971])
    print(f"Wilcoxon sanity check: statistic={stat:.4f}  p={p:.4f}")
