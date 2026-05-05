"""Statistical helpers used across all experiments.

All effect sizes + CIs follow ROADMAP Part 3:
  - 95% CI on every reported number (bootstrap or analytic)
  - McNemar for paired binary comparisons
  - DeLong for AUC comparisons
  - Wilcoxon signed-rank for paired continuous
  - Bonferroni / FDR for multiple comparisons
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np
from scipy import stats


# ============================================================
# Confidence intervals
# ============================================================

def wilson_ci(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """Wilson score CI for a binomial proportion. (lo, hi)."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    halfw = (z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (max(0.0, centre - halfw), min(1.0, centre + halfw))


def bootstrap_ci(
    samples: Sequence[float],
    func=np.mean,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """Returns (point, lo, hi)."""
    arr = np.asarray(samples, dtype=float)
    if len(arr) == 0:
        return (0.0, 0.0, 0.0)
    rng = np.random.default_rng(seed)
    point = float(func(arr))
    boots = np.array([
        func(rng.choice(arr, size=len(arr), replace=True)) for _ in range(n_boot)
    ])
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return (point, float(lo), float(hi))


def bootstrap_paired_diff_ci(
    a: Sequence[float], b: Sequence[float],
    n_boot: int = 1000, alpha: float = 0.05, seed: int = 42,
) -> Tuple[float, float, float]:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    assert len(a) == len(b), "paired bootstrap requires equal-length arrays"
    rng = np.random.default_rng(seed)
    diffs = a - b
    point = float(np.mean(diffs))
    boots = np.array([
        np.mean(rng.choice(diffs, size=len(diffs), replace=True))
        for _ in range(n_boot)
    ])
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return (point, float(lo), float(hi))


# ============================================================
# Paired binary tests
# ============================================================

def mcnemar(a_correct: Sequence[bool], b_correct: Sequence[bool]) -> dict:
    """McNemar's test for paired binary data. a vs b on same examples.
    Returns {'b01': only-b-correct count, 'b10': only-a-correct count,
             'p': two-sided p (binomial exact),
             'odds_ratio': b10/b01,
             'a_acc', 'b_acc', 'diff_pp'}."""
    a = np.asarray(a_correct, dtype=bool)
    b = np.asarray(b_correct, dtype=bool)
    assert len(a) == len(b)
    n10 = int(np.sum(a & ~b))   # a correct, b wrong
    n01 = int(np.sum(~a & b))   # b correct, a wrong
    n = n10 + n01
    if n == 0:
        p = 1.0
    else:
        # exact binomial two-sided
        k = min(n10, n01)
        p = float(stats.binomtest(k, n, p=0.5).pvalue)
    return {
        "n": int(len(a)),
        "n_only_a_correct": n10,
        "n_only_b_correct": n01,
        "p_value": p,
        "odds_ratio_b_over_a": (n01 / n10) if n10 > 0 else float("inf"),
        "a_accuracy": float(np.mean(a)),
        "b_accuracy": float(np.mean(b)),
        "diff_pp": float(100.0 * (np.mean(b) - np.mean(a))),
    }


# ============================================================
# DeLong AUC test
# ============================================================

def _compute_midrank(x: np.ndarray) -> np.ndarray:
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


def _delong_components(scores_pos: np.ndarray, scores_neg: np.ndarray):
    m = len(scores_pos)
    n = len(scores_neg)
    all_scores = np.concatenate([scores_pos, scores_neg])
    Tall = _compute_midrank(all_scores)
    Tpos = Tall[:m]
    Tneg = Tall[m:]
    Tpos_only = _compute_midrank(scores_pos)
    Tneg_only = _compute_midrank(scores_neg)
    auc = (np.sum(Tpos) - m * (m + 1) / 2) / (m * n)
    v01 = (Tpos - Tpos_only) / n
    v10 = 1 - (Tneg - Tneg_only) / m
    s01 = float(np.var(v01, ddof=1)) if m > 1 else 0.0
    s10 = float(np.var(v10, ddof=1)) if n > 1 else 0.0
    var = s01 / m + s10 / n
    return float(auc), float(var)


def auc_with_ci(scores: Sequence[float], labels: Sequence[int],
                alpha: float = 0.05) -> dict:
    """AUC with DeLong variance CI."""
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=int)
    pos = s[y == 1]
    neg = s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return {"auc": 0.5, "ci_lo": 0.0, "ci_hi": 1.0,
                "n_pos": int(len(pos)), "n_neg": int(len(neg))}
    auc, var = _delong_components(pos, neg)
    se = float(np.sqrt(var))
    z = 1.959963984540054
    return {
        "auc": auc,
        "ci_lo": max(0.0, auc - z * se),
        "ci_hi": min(1.0, auc + z * se),
        "se": se,
        "n_pos": int(len(pos)),
        "n_neg": int(len(neg)),
    }


def delong_paired_test(
    scores_a: Sequence[float], scores_b: Sequence[float], labels: Sequence[int]
) -> dict:
    """Paired DeLong test for AUC_a == AUC_b on same labels."""
    sa = np.asarray(scores_a, dtype=float)
    sb = np.asarray(scores_b, dtype=float)
    y = np.asarray(labels, dtype=int)
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    m, n = len(pos_idx), len(neg_idx)
    if m < 2 or n < 2:
        return {"auc_a": 0.5, "auc_b": 0.5, "delta": 0.0, "p_value": 1.0}

    def comp(s):
        return _delong_components(s[pos_idx], s[neg_idx])

    auc_a, var_a = comp(sa)
    auc_b, var_b = comp(sb)
    # rough independent variance combination — paired DeLong covariance
    # would require recomputing v01/v10 jointly. This bound is conservative.
    se = float(np.sqrt(var_a + var_b))
    z = (auc_a - auc_b) / se if se > 0 else 0.0
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return {
        "auc_a": auc_a, "auc_b": auc_b,
        "delta": auc_a - auc_b, "z": z, "p_value": float(p),
    }


# ============================================================
# Other tests
# ============================================================

def wilcoxon_paired(a: Sequence[float], b: Sequence[float]) -> dict:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) < 2:
        return {"statistic": 0.0, "p_value": 1.0, "n": int(len(a))}
    try:
        res = stats.wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
        return {"statistic": float(res.statistic), "p_value": float(res.pvalue),
                "n": int(len(a)),
                "median_diff": float(np.median(a - b))}
    except Exception:
        return {"statistic": 0.0, "p_value": 1.0, "n": int(len(a))}


def mannwhitney_u(a: Sequence[float], b: Sequence[float]) -> dict:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) < 2 or len(b) < 2:
        return {"u": 0.0, "p_value": 1.0, "rb_corr": 0.0}
    res = stats.mannwhitneyu(a, b, alternative="two-sided")
    n1, n2 = len(a), len(b)
    rb = 1 - (2 * res.statistic) / (n1 * n2)  # rank-biserial
    return {"u": float(res.statistic), "p_value": float(res.pvalue),
            "rb_corr": float(rb), "n_a": n1, "n_b": n2}


def permutation_test_diff(
    a: Sequence[float], b: Sequence[float],
    func=np.mean, n_perm: int = 1000, seed: int = 42,
) -> dict:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    obs = float(func(a) - func(b))
    pool = np.concatenate([a, b])
    rng = np.random.default_rng(seed)
    n_a = len(a)
    cnt = 0
    for _ in range(n_perm):
        rng.shuffle(pool)
        diff = func(pool[:n_a]) - func(pool[n_a:])
        if abs(diff) >= abs(obs):
            cnt += 1
    return {"observed_diff": obs, "p_value": (cnt + 1) / (n_perm + 1)}


def cohens_d(a: Sequence[float], b: Sequence[float]) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) < 2 or len(b) < 2:
        return 0.0
    s_pool = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1))
                     / (len(a) + len(b) - 2))
    if s_pool == 0:
        return 0.0
    return float((a.mean() - b.mean()) / s_pool)


# ============================================================
# Multiple-comparisons correction
# ============================================================

def benjamini_hochberg(p_values: Sequence[float], alpha: float = 0.05) -> Tuple[list, list]:
    """Returns (rejected[bool], q_values)."""
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    if n == 0:
        return [], []
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * n / (np.arange(1, n + 1))
    q = np.minimum.accumulate(q[::-1])[::-1]
    q_unsorted = np.empty(n)
    q_unsorted[order] = q
    rejected = (q_unsorted < alpha).tolist()
    return rejected, q_unsorted.tolist()


# ============================================================
# CKA — used in Exps A and O
# ============================================================

def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Centered Kernel Alignment (linear). X: [n,p], Y: [n,q]."""
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)
    num = float(np.linalg.norm(X.T @ Y, "fro") ** 2)
    den = float(np.linalg.norm(X.T @ X, "fro") * np.linalg.norm(Y.T @ Y, "fro"))
    if den == 0:
        return 0.0
    return num / den
