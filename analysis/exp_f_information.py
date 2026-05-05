"""Exp F — information propagation across rounds.

F1. MI(h_t; h_{t+k}) for k=1,2,3 via Kraskov kNN estimator
F2. Split by correctness (correct vs incorrect MI half-life)
F3. LMAS vs TextMAS round-to-round MI
F4. Rate-distortion curve: PCA top-k vs answer-decodability AUC
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from analysis import common
from analysis.stats import bootstrap_ci, auc_with_ci


def kraskov_mi(X: np.ndarray, Y: np.ndarray, k: int = 3) -> float:
    """Kraskov–Stoegbauer–Grassberger I(X;Y), estimator 1.
    X: [N, dx], Y: [N, dy]. Returns nats."""
    from scipy.spatial import KDTree
    from scipy.special import digamma
    N = X.shape[0]
    if N < k + 5:
        return float("nan")
    Z = np.concatenate([X, Y], axis=1)
    tree_z = KDTree(Z)
    eps, _ = tree_z.query(Z, k=k + 1, p=np.inf)
    eps = eps[:, -1]
    tree_x = KDTree(X)
    tree_y = KDTree(Y)
    nx = np.array([len(tree_x.query_ball_point(X[i], r=eps[i] - 1e-12, p=np.inf)) - 1
                   for i in range(N)])
    ny = np.array([len(tree_y.query_ball_point(Y[i], r=eps[i] - 1e-12, p=np.inf)) - 1
                   for i in range(N)])
    nx = np.clip(nx, 1, N)
    ny = np.clip(ny, 1, N)
    mi = digamma(k) - np.mean(digamma(nx + 1) + digamma(ny + 1)) + digamma(N)
    return float(max(mi, 0.0))


def _round_pair_mi(X: np.ndarray, t: int, k: int, d_proj: int = 16,
                   n_neighbors: int = 3) -> float:
    """X: [N, A, M, D]. Mean MI between round t and round t+k across agents.
    PCA reduce to d_proj for tractability."""
    if t + k >= X.shape[2]:
        return float("nan")
    mis = []
    for a in range(X.shape[1]):
        Ht = X[:, a, t, :]
        Htk = X[:, a, t + k, :]
        if Ht.shape[1] > d_proj:
            pca = PCA(n_components=d_proj).fit(np.concatenate([Ht, Htk], 0))
            Ht = pca.transform(Ht); Htk = pca.transform(Htk)
        mi = kraskov_mi(Ht, Htk, k=n_neighbors)
        if not np.isnan(mi):
            mis.append(mi)
    return float(np.mean(mis)) if mis else float("nan")


def _rate_distortion(X: np.ndarray, y: np.ndarray) -> dict:
    """X: [N, D]. Project onto top-k PCA, train LR, report AUC vs k."""
    if X.shape[0] < 30 or y.sum() < 10 or (~y.astype(bool)).sum() < 10:
        return {}
    n_max = min(X.shape[0] - 1, X.shape[1], 512)
    pca = PCA(n_components=n_max).fit(X)
    Xp = pca.transform(X)
    out = []
    for k in [1, 2, 4, 8, 16, 32, 64, 128, 256, min(512, Xp.shape[1])]:
        if k > Xp.shape[1]:
            continue
        Xk = Xp[:, :k]
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        scores = np.zeros_like(y, dtype=float)
        for tr, te in skf.split(Xk, y):
            clf = LogisticRegression(C=1.0, max_iter=1000).fit(Xk[tr], y[tr])
            scores[te] = clf.predict_proba(Xk[te])[:, 1]
        auc = auc_with_ci(scores, y)
        out.append({"k": k, "auc": auc["auc"], "ci_lo": auc["ci_lo"], "ci_hi": auc["ci_hi"]})
    return {"curve": out}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--activations_dir", required=True, type=Path)
    args = p.parse_args()
    common.setup_logging()
    out = common.results_dir(args.activations_dir, "exp_f")
    results = {}

    # F1/F2/F3: MI between rounds
    for cond in ("latent_mas", "text_mas"):
        per_task = {}
        for task in common.TASKS:
            X, ifo = common.stack_post_aligned(args.activations_dir, cond, task,
                                                limit=200)
            if X.size == 0 or X.shape[2] < 2:
                continue
            y = np.array([1 if i["correct"] else 0 for i in ifo])
            per_k = {}
            for kk in (1, 2, 3):
                if kk >= X.shape[2]:
                    continue
                # all examples
                mi_all = _round_pair_mi(X, t=0, k=kk)
                # split by correctness
                if y.sum() > 10 and (~y.astype(bool)).sum() > 10:
                    mi_c = _round_pair_mi(X[y == 1], t=0, k=kk)
                    mi_i = _round_pair_mi(X[y == 0], t=0, k=kk)
                else:
                    mi_c = mi_i = float("nan")
                per_k[f"k={kk}"] = {"mi_all": mi_all, "mi_correct": mi_c, "mi_incorrect": mi_i}
            per_task[task] = per_k
        results[cond] = per_task

    # F4: rate-distortion (LMAS only)
    rd = {}
    for task in common.TASKS:
        X, ifo = common.stack_post_aligned(args.activations_dir, "latent_mas", task)
        if X.size == 0:
            continue
        H = X.mean(axis=(1, 2))
        y = np.array([1 if i["correct"] else 0 for i in ifo])
        rd[task] = _rate_distortion(H, y)
    results["F4_rate_distortion"] = rd

    (out / "exp_f.json").write_text(json.dumps(results, indent=2))
    print(f"[exp_f] wrote {out/'exp_f.json'}")


if __name__ == "__main__":
    main()
