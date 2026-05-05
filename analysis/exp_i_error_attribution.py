"""Exp I — error attribution.

I1. Error introduction score EI(i, m) = max(0, dist(h_im, correct) - dist(h_{i,m-1}, correct))
I2. Distribution of blame across (agent, round)
I3. Error contagion: how many subsequent agents diverge after a high-EI event
I4. Confirmation bias: do incorrect initial directions get reinforced?
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import chisquare

from analysis import common
from analysis.stats import bootstrap_ci, mannwhitney_u


def _correct_centroid(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Centroid of correct examples in mean-pooled hidden space."""
    H = X.mean(axis=(1, 2))
    if y.sum() < 5:
        return H.mean(axis=0)
    return H[y == 1].mean(axis=0)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--activations_dir", required=True, type=Path)
    args = p.parse_args()
    common.setup_logging()
    out = common.results_dir(args.activations_dir, "exp_i")
    results = {}

    for cond in ("latent_mas", "text_mas"):
        per_task = {}
        for task in common.TASKS:
            X, ifo = common.stack_post_aligned(args.activations_dir, cond, task)
            if X.size == 0 or X.shape[1] < 1 or X.shape[2] < 2:
                continue
            y = np.array([1 if i["correct"] else 0 for i in ifo])
            N, A, M, D = X.shape
            centroid = _correct_centroid(X, y)
            cnorm = np.linalg.norm(centroid) + 1e-12

            # I1: per-(i, m) EI on incorrect examples
            EI = np.zeros((N, A, M))
            for n in range(N):
                for a in range(A):
                    prev_dist = None
                    for m in range(M):
                        h = X[n, a, m]
                        d = 1 - np.dot(h, centroid) / (np.linalg.norm(h) + 1e-12) / cnorm
                        if prev_dist is not None:
                            EI[n, a, m] = max(0, d - prev_dist)
                        prev_dist = d

            # I2: blame distribution among incorrect examples
            failures = (y == 0)
            blame_counts = np.zeros((A, M), dtype=int)
            for n in np.where(failures)[0]:
                flat = EI[n].flatten()
                idx = np.argmax(flat)
                a, m = np.unravel_index(idx, EI[n].shape)
                blame_counts[a, m] += 1
            total = max(blame_counts.sum(), 1)
            blame_dist = (blame_counts / total).tolist()
            try:
                chi = chisquare(blame_counts.flatten())
                chi_p = float(chi.pvalue)
            except Exception:
                chi_p = float("nan")

            # I3: contagion — count subsequent rounds with EI > threshold
            thr = float(EI.mean() + 2 * EI.std())
            contagion = []
            for n in np.where(failures)[0]:
                flat = EI[n].flatten()
                idx = np.argmax(flat)
                a, m = np.unravel_index(idx, EI[n].shape)
                # count subsequent (a', m') > thr
                subs = 0
                for ap in range(A):
                    for mp in range(M):
                        if (ap > a) or (ap == a and mp > m):
                            if EI[n, ap, mp] > thr:
                                subs += 1
                contagion.append(subs)
            cont_m, cont_lo, cont_hi = bootstrap_ci(contagion) if contagion else (0, 0, 0)

            # I4: confirmation bias — direction round 1 vs round 2 cosine shift
            i4 = {}
            if M >= 2:
                cos_shift_correct = []
                cos_shift_incorrect = []
                for n in range(N):
                    for a in range(A):
                        h1 = X[n, a, 0]; h2 = X[n, a, 1]
                        nrm = (np.linalg.norm(h1) + 1e-12) * (np.linalg.norm(h2) + 1e-12)
                        s = float(np.dot(h1, h2) / nrm)
                        if y[n] == 1:
                            cos_shift_correct.append(s)
                        else:
                            cos_shift_incorrect.append(s)
                if cos_shift_correct and cos_shift_incorrect:
                    mw = mannwhitney_u(cos_shift_correct, cos_shift_incorrect)
                    i4 = {
                        "mean_cos_shift_correct": float(np.mean(cos_shift_correct)),
                        "mean_cos_shift_incorrect": float(np.mean(cos_shift_incorrect)),
                        "mannwhitney": mw,
                    }

            per_task[task] = {
                "I2_blame_distribution": blame_dist,
                "I2_chi2_p": chi_p,
                "I3_contagion_mean": cont_m,
                "I3_contagion_ci": [cont_lo, cont_hi],
                "I3_threshold": thr,
                "I4_confirmation": i4,
                "n_failures": int(failures.sum()),
            }
        results[cond] = per_task

    (out / "exp_i.json").write_text(json.dumps(results, indent=2))
    print(f"[exp_i] wrote {out/'exp_i.json'}")


if __name__ == "__main__":
    main()
