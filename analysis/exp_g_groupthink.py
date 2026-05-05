"""Exp G — groupthink vs diversity.

G1. Flocking coefficient F(t) = mean pairwise cos sim across agents at round t
G2. F(round=1) as predictor of final incorrectness — AUC
G3. Minority-report fraction (converged-wrong with recoverable dissent)
G4. Centroid AUC vs single-agent AUC (wisdom of crowds)
G5. Monotonic convergence rate (cleaner attractors on correct examples?)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from analysis import common
from analysis.stats import (
    auc_with_ci, bootstrap_ci, delong_paired_test, mannwhitney_u,
)


def _flocking(X: np.ndarray, t: int) -> np.ndarray:
    """X: [N, A, M, D]. Returns F(t) per example as mean pairwise cos."""
    Xn = X[:, :, t, :] / (np.linalg.norm(X[:, :, t, :], axis=-1, keepdims=True) + 1e-12)
    N, A, _ = Xn.shape
    out = np.zeros(N)
    if A < 2:
        return out
    pairs = [(i, j) for i in range(A) for j in range(A) if i < j]
    for (i, j) in pairs:
        out += np.sum(Xn[:, i] * Xn[:, j], axis=1)
    out /= len(pairs)
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--activations_dir", required=True, type=Path)
    args = p.parse_args()
    common.setup_logging()
    out = common.results_dir(args.activations_dir, "exp_g")
    results = {}

    for cond in ("latent_mas", "text_mas"):
        per_task = {}
        for task in common.TASKS:
            X, ifo = common.stack_post_aligned(args.activations_dir, cond, task)
            if X.size == 0 or X.shape[1] < 2:
                continue
            y = np.array([1 if i["correct"] else 0 for i in ifo])
            N, A, M, D = X.shape
            F = np.stack([_flocking(X, t) for t in range(M)], axis=1)  # [N, M]

            # G1: per-round F mean ± CI
            g1 = []
            for t in range(M):
                m, lo, hi = bootstrap_ci(F[:, t].tolist())
                g1.append({"round": t, "mean_F": m, "ci_lo": lo, "ci_hi": hi})

            # G2: F(round=1) → predict incorrect
            if y.sum() > 10 and (~y.astype(bool)).sum() > 10 and M >= 1:
                t1 = min(1, M - 1)
                g2 = auc_with_ci(F[:, t1].tolist(), (1 - y).tolist())
            else:
                g2 = {}

            # G3: converged-wrong with recoverable dissent
            converged_wrong = (F[:, -1] > 0.9) & (y == 0)
            n_cw = int(converged_wrong.sum())
            recoverable = 0
            for n_idx in np.where(converged_wrong)[0]:
                # is there an agent whose final-round projection points more
                # toward "correct manifold" than the others?
                # use a simple heuristic: highest distance from mean of others
                Xn = X[n_idx, :, -1, :]
                Xn = Xn / (np.linalg.norm(Xn, axis=1, keepdims=True) + 1e-12)
                centroid = Xn.mean(0); centroid /= (np.linalg.norm(centroid) + 1e-12)
                dists = np.array([1 - np.dot(Xn[a], centroid) for a in range(A)])
                if dists.max() > 0.05:  # at least one agent diverges
                    recoverable += 1
            g3 = {"n_converged_wrong": n_cw,
                  "n_with_dissent": int(recoverable),
                  "fraction": float(recoverable / max(n_cw, 1))}

            # G4: centroid AUC vs single-agent AUC
            if y.sum() > 10 and (~y.astype(bool)).sum() > 10:
                centroid = X[:, :, -1, :].mean(axis=1)  # [N, D]
                single = X[:, 0, -1, :]
                skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
                s_c = np.zeros_like(y, dtype=float); s_s = np.zeros_like(y, dtype=float)
                for tr, te in skf.split(centroid, y):
                    s_c[te] = LogisticRegression(C=1.0, max_iter=1000).fit(
                        centroid[tr], y[tr]).predict_proba(centroid[te])[:, 1]
                    s_s[te] = LogisticRegression(C=1.0, max_iter=1000).fit(
                        single[tr], y[tr]).predict_proba(single[te])[:, 1]
                g4_c = auc_with_ci(s_c, y)
                g4_s = auc_with_ci(s_s, y)
                g4_d = delong_paired_test(s_c, s_s, y)
            else:
                g4_c = g4_s = g4_d = {}

            # G5: monotonic convergence
            mono = (np.diff(F, axis=1) <= 0).all(axis=1)  # F decreases monotonically
            g5 = {
                "frac_monotonic_correct": float(mono[y == 1].mean()) if y.sum() else 0,
                "frac_monotonic_incorrect": float(mono[y == 0].mean()) if (~y.astype(bool)).sum() else 0,
            }

            per_task[task] = {"G1": g1, "G2": g2, "G3": g3,
                              "G4_centroid": g4_c, "G4_single": g4_s, "G4_delong": g4_d,
                              "G5": g5}
        results[cond] = per_task

    (out / "exp_g.json").write_text(json.dumps(results, indent=2))
    print(f"[exp_g] wrote {out/'exp_g.json'}")


if __name__ == "__main__":
    main()
