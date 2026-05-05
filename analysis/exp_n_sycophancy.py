"""Exp N — sycophancy and minority suppression.

N1. Dominance score per agent (mean R² of subsequent agents on this one)
N2. Sycophancy: when dominant agent points wrong, do later agents shift toward it?
N3. Minority suppression: among examples where one agent is closer to correct
    manifold but in minority, does final answer follow majority or minority?
N4. Sycophancy direction (contrastive): AUC for predicting incorrectness
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats as sp_stats
from sklearn.linear_model import Ridge

from analysis import common
from analysis.stats import auc_with_ci, bootstrap_ci


def _r2(y_true, y_pred):
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    return 1 - ss_res / max(ss_tot, 1e-12)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--activations_dir", required=True, type=Path)
    args = p.parse_args()
    common.setup_logging()
    out = common.results_dir(args.activations_dir, "exp_n")
    results = {}

    for cond in ("latent_mas", "text_mas"):
        per_task = {}
        for task in common.TASKS:
            X, ifo = common.stack_post_aligned(args.activations_dir, cond, task)
            if X.size == 0 or X.shape[1] < 2:
                continue
            y = np.array([1 if i["correct"] else 0 for i in ifo])
            N, A, M, D = X.shape
            Xl = X[:, :, -1, :]  # last round

            # N1 — dominance: for each i, mean R² of agents j>i predicting from i
            dominance = np.zeros(A)
            for i in range(A):
                r2s = []
                for j in range(A):
                    if j <= i:
                        continue
                    try:
                        clf = Ridge(alpha=1.0).fit(Xl[:, i], Xl[:, j])
                        r2s.append(_r2(Xl[:, j], clf.predict(Xl[:, i])))
                    except Exception:
                        pass
                dominance[i] = float(np.mean(r2s)) if r2s else 0.0
            dominant = int(np.argmax(dominance))

            # N2 — sycophancy: among incorrect examples, do later agents
            # have higher cos similarity with dominant agent than with the rest?
            shifts = []
            for n in np.where(y == 0)[0]:
                Xn = Xl[n]
                Xn = Xn / (np.linalg.norm(Xn, axis=1, keepdims=True) + 1e-12)
                cos_dom = []
                cos_else = []
                for j in range(A):
                    if j == dominant:
                        continue
                    cos_dom.append(float(np.dot(Xn[j], Xn[dominant])))
                    others = [k for k in range(A) if k != j and k != dominant]
                    if others:
                        cos_else.append(float(np.mean([np.dot(Xn[j], Xn[k]) for k in others])))
                if cos_dom and cos_else:
                    shifts.append(np.mean(cos_dom) - np.mean(cos_else))
            if shifts:
                m, lo, hi = bootstrap_ci(shifts)
                # binomial: how often is shift > 0 (toward dominant)
                n_pos = int(np.sum(np.array(shifts) > 0))
                p = float(sp_stats.binomtest(n_pos, len(shifts), p=0.5).pvalue)
            else:
                m = lo = hi = float("nan"); n_pos = 0; p = 1.0

            # N4 — sycophancy direction (contrastive: shift>0 vs shift<0 on
            # examples where minority is closer to correct manifold)
            # simplified: contrastive on (correct, incorrect) and project final-round centroid
            if y.sum() > 10 and (~y.astype(bool)).sum() > 10:
                Hc = Xl[y == 1].mean(axis=(0, 1))
                Hi_ = Xl[y == 0].mean(axis=(0, 1))
                d_syc = (Hi_ - Hc); d_syc /= (np.linalg.norm(d_syc) + 1e-12)
                proj = (Xl.mean(axis=1)) @ d_syc
                auc_n4 = auc_with_ci(proj.tolist(), (1 - y).tolist())
            else:
                auc_n4 = {}

            per_task[task] = {
                "N1_dominance_per_agent": dominance.tolist(),
                "N1_dominant_agent": dominant,
                "N2_n_with_shift": len(shifts),
                "N2_mean_shift_toward_dominant": m,
                "N2_ci": [lo, hi],
                "N2_fraction_positive": float(n_pos / max(len(shifts), 1)),
                "N2_binomial_p": p,
                "N4_sycophancy_direction_auc": auc_n4,
            }
        results[cond] = per_task

    (out / "exp_n.json").write_text(json.dumps(results, indent=2))
    print(f"[exp_n] wrote {out/'exp_n.json'}")


if __name__ == "__main__":
    main()
