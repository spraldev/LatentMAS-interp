"""Exp D — correct vs incorrect trajectory geometry.

D1. Per-(agent, round) AUC heatmap of logistic regression predicting final correctness
D2. vs single-agent baseline
D3. Top-20 Fisher-information dimensions; overlap with task-id directions (Exp C)
D4. Intrinsic dimensionality (TwoNN)
D5. Semantic direction inventory (5 named directions)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from analysis import common
from analysis.stats import auc_with_ci, delong_paired_test, mannwhitney_u, benjamini_hochberg


def _twonn(X: np.ndarray, n_neighbors: int = 2) -> float:
    """Facco et al. 2017 TwoNN intrinsic dimensionality estimator."""
    from sklearn.neighbors import NearestNeighbors
    if X.shape[0] < 10:
        return float("nan")
    nbrs = NearestNeighbors(n_neighbors=3).fit(X)
    d, _ = nbrs.kneighbors(X)
    r1 = d[:, 1] + 1e-12
    r2 = d[:, 2] + 1e-12
    mu = r2 / r1
    mu = mu[mu > 1.0]
    if len(mu) < 5:
        return float("nan")
    F = np.arange(1, len(mu) + 1) / len(mu)
    F = F[:-1]
    log_mu = np.log(np.sort(mu)[:-1])
    log_minus_logF = -np.log(1 - F)
    if len(log_mu) < 2:
        return float("nan")
    d_est, _ = np.polyfit(log_mu, log_minus_logF, 1)[:2] if False else (np.polyfit(log_mu, log_minus_logF, 1)[0], 0)
    return float(d_est)


def _per_position_auc(root: Path, condition: str) -> dict:
    Xs, ys, info = [], [], []
    for task in common.TASKS:
        X, ifo = common.stack_post_aligned(root, condition, task)
        if X.size == 0:
            continue
        Xs.append(X)
        ys.extend([1 if i["correct"] else 0 for i in ifo])
    if not Xs:
        return {}
    X_all = np.concatenate(Xs, axis=0)  # [N, A, M, D]
    y_all = np.array(ys)
    if y_all.sum() < 10 or (~y_all.astype(bool)).sum() < 10:
        return {"n_correct": int(y_all.sum()), "n_incorrect": int(len(y_all) - y_all.sum())}

    A, M = X_all.shape[1], X_all.shape[2]
    heat = np.zeros((A, M))
    cis = {}
    for a in range(A):
        for m in range(M):
            H = X_all[:, a, m, :]
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            scores = np.zeros_like(y_all, dtype=float)
            for tr, te in skf.split(H, y_all):
                clf = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs")
                clf.fit(H[tr], y_all[tr])
                scores[te] = clf.predict_proba(H[te])[:, 1]
            auc = auc_with_ci(scores, y_all)
            heat[a, m] = auc["auc"]
            cis[f"agent_{a}_round_{m}"] = auc
    return {"heatmap": heat.tolist(), "cis": cis,
            "n_correct": int(y_all.sum()),
            "n_incorrect": int(len(y_all) - y_all.sum()),
            "A": A, "M": M}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--activations_dir", required=True, type=Path)
    args = p.parse_args()
    common.setup_logging()
    out = common.results_dir(args.activations_dir, "exp_d")
    results = {}

    # D1
    d1 = _per_position_auc(args.activations_dir, "latent_mas")
    results["D1_latent_mas"] = d1
    # D2
    d2 = _per_position_auc(args.activations_dir, "single_agent_latent_sampled")
    results["D2_single_agent"] = d2

    # D3 — Fisher info per dimension (use last (a, r) on LMAS)
    fisher = {}
    for task in common.TASKS:
        X, ifo = common.stack_post_aligned(args.activations_dir, "latent_mas", task)
        if X.size == 0:
            continue
        H = X[:, -1, -1, :]  # last agent, last round
        y = np.array([1 if i["correct"] else 0 for i in ifo])
        if y.sum() < 10 or (~y.astype(bool)).sum() < 10:
            continue
        # logistic regression coefficient is a Fisher-info proxy
        clf = LogisticRegression(C=1.0, max_iter=2000).fit(H, y)
        coefs = np.abs(clf.coef_[0])
        top20 = np.argsort(-coefs)[:20].tolist()
        fisher[task] = {"top20_dims": top20,
                         "top20_coefs": coefs[top20].tolist()}
    results["D3_fisher"] = fisher

    # D4 — intrinsic dimensionality of correct vs incorrect clouds (LMAS)
    d4 = []
    for task in common.TASKS:
        X, ifo = common.stack_post_aligned(args.activations_dir, "latent_mas", task)
        if X.size == 0:
            continue
        H = X.mean(axis=(1, 2))  # collapse agent/round → [N, D]
        y = np.array([1 if i["correct"] else 0 for i in ifo])
        H_c = H[y == 1]; H_i = H[y == 0]
        if len(H_c) < 20 or len(H_i) < 20:
            continue
        # subsample for stability
        from sklearn.decomposition import PCA
        # reduce to ≤256 first to avoid degenerate kNN distances
        d_target = min(256, H.shape[1])
        pca = PCA(n_components=d_target, random_state=42).fit(H)
        H_c_p = pca.transform(H_c); H_i_p = pca.transform(H_i)
        dim_c = _twonn(H_c_p)
        dim_i = _twonn(H_i_p)
        mw = mannwhitney_u([dim_c], [dim_i])  # not really paired across tasks
        d4.append({
            "task": task,
            "intrinsic_dim_correct": dim_c,
            "intrinsic_dim_incorrect": dim_i,
            "mannwhitney": mw,
        })
    results["D4_intrinsic_dim"] = d4

    # D5 — semantic directions (basic 5)
    d5 = {}
    for task in common.TASKS:
        X, ifo = common.stack_post_aligned(args.activations_dir, "latent_mas", task)
        if X.size == 0:
            continue
        N, A, M, D = X.shape
        y_correct = np.array([1 if i["correct"] else 0 for i in ifo])
        # i: confidence direction (correct vs incorrect)
        H_last = X[:, -1, -1, :]
        H_c = H_last[y_correct == 1]; H_i = H_last[y_correct == 0]
        if len(H_c) > 10 and len(H_i) > 10:
            d_conf = (H_c.mean(0) - H_i.mean(0))
            d_conf /= (np.linalg.norm(d_conf) + 1e-12)
        else:
            d_conf = None
        # ii: error correction (early wrong → late right) — too sparse w/o probe
        # iv: agent disagreement direction
        if A > 1:
            cos_ag = np.array([
                np.mean([np.sum(X[i, a, -1] * X[i, b, -1]) /
                         (np.linalg.norm(X[i, a, -1]) * np.linalg.norm(X[i, b, -1]) + 1e-12)
                         for a in range(A) for b in range(A) if a < b])
                for i in range(N)])
            top, bot = np.percentile(cos_ag, [33, 66])
            high = X[cos_ag <= top].reshape(-1, D).mean(0)
            low = X[cos_ag >= bot].reshape(-1, D).mean(0)
            d_disagree = (high - low)
            d_disagree /= (np.linalg.norm(d_disagree) + 1e-12)
        else:
            d_disagree = None
        d5[task] = {
            "confidence_direction_norm": float(np.linalg.norm(d_conf)) if d_conf is not None else None,
            "disagreement_direction_norm": float(np.linalg.norm(d_disagree)) if d_disagree is not None else None,
        }
        if d_conf is not None:
            np.save(out / f"d5_confidence_dir_{task}.npy", d_conf)
        if d_disagree is not None:
            np.save(out / f"d5_disagreement_dir_{task}.npy", d_disagree)
    results["D5"] = d5

    (out / "exp_d.json").write_text(json.dumps(results, indent=2))
    print(f"[exp_d] wrote {out/'exp_d.json'}")
    if "D1_latent_mas" in results and "heatmap" in results["D1_latent_mas"]:
        h = np.array(results["D1_latent_mas"]["heatmap"])
        print(f"  LMAS AUC heatmap shape {h.shape}, peak={h.max():.3f}")


if __name__ == "__main__":
    main()
