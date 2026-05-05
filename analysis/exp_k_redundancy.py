"""Exp K — redundancy and error correction.

K1. Pairwise R² (ridge regression) between agents
K2. mean R² vs final correctness (Spearman)
K3. Complementarity score = 1 - mean R²
K4. LMAS vs TextMAS redundancy
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge

from analysis import common
from analysis.stats import bootstrap_ci, permutation_test_diff


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    return 1 - ss_res / max(ss_tot, 1e-12)


def _pairwise_r2(X: np.ndarray) -> np.ndarray:
    """X: [N, A, D] (last round). Returns [A, A] matrix of R²(j → i)."""
    N, A, D = X.shape
    R = np.zeros((A, A))
    for i in range(A):
        for j in range(A):
            if i == j:
                R[i, j] = 1.0
                continue
            try:
                clf = Ridge(alpha=1.0).fit(X[:, j, :], X[:, i, :])
                pred = clf.predict(X[:, j, :])
                R[i, j] = _r2(X[:, i, :], pred)
            except Exception:
                R[i, j] = 0.0
    return R


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--activations_dir", required=True, type=Path)
    args = p.parse_args()
    common.setup_logging()
    out = common.results_dir(args.activations_dir, "exp_k")
    results = {}

    for cond in ("latent_mas", "text_mas"):
        per_task = {}
        for task in common.TASKS:
            X, ifo = common.stack_post_aligned(args.activations_dir, cond, task)
            if X.size == 0 or X.shape[1] < 2:
                continue
            y = np.array([1 if i["correct"] else 0 for i in ifo])
            Xl = X[:, :, -1, :]  # last round
            R = _pairwise_r2(Xl)
            mean_r2 = float(np.mean([R[i, j] for i in range(R.shape[0]) for j in range(R.shape[0]) if i != j]))

            # K2: mean R² per example vs correctness
            per_ex_r2 = []
            for n in range(Xl.shape[0]):
                # per-example pairwise R² is ill-defined; use cosine instead as proxy
                Xn = Xl[n]
                Xn = Xn / (np.linalg.norm(Xn, axis=1, keepdims=True) + 1e-12)
                ms = []
                for i in range(Xn.shape[0]):
                    for j in range(Xn.shape[0]):
                        if i < j:
                            ms.append(float(np.dot(Xn[i], Xn[j])))
                per_ex_r2.append(np.mean(ms) if ms else 0)
            per_ex_r2 = np.array(per_ex_r2)
            if y.sum() > 5 and (~y.astype(bool)).sum() > 5:
                rho, pv = spearmanr(per_ex_r2, y)
            else:
                rho, pv = float("nan"), float("nan")

            per_task[task] = {
                "K1_R2_matrix": R.tolist(),
                "K1_mean_R2": mean_r2,
                "K2_spearman_R2_vs_correct": float(rho),
                "K2_p_value": float(pv),
                "K3_complementarity": 1 - mean_r2,
                "n": int(Xl.shape[0]),
            }
        results[cond] = per_task

    # K4: LMAS vs TextMAS redundancy
    k4 = {}
    if "latent_mas" in results and "text_mas" in results:
        for task in common.TASKS:
            l = results["latent_mas"].get(task, {}).get("K1_mean_R2")
            t = results["text_mas"].get(task, {}).get("K1_mean_R2")
            if l is not None and t is not None:
                k4[task] = {"latent_mean_R2": l, "text_mean_R2": t,
                            "diff": l - t}
    results["K4_compare"] = k4

    (out / "exp_k.json").write_text(json.dumps(results, indent=2))
    print(f"[exp_k] wrote {out/'exp_k.json'}")


if __name__ == "__main__":
    main()
