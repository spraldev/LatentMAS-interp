"""Exp M — W_a ablation (post-hoc analysis of conditions already collected).

Reads metadata.json from:
  latent_mas (trained W_a, full system)
  exp_m_identity_wa (W_a = I)
  latent_mas_random_wa_orth (W_a = random orthogonal)
  latent_mas_zero_wa (W_a = 0)
  no_transfer (no cross-agent transfer at all)

Reports: per-task accuracy + Wilson CI + paired McNemar against trained.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from analysis import common
from analysis.stats import mcnemar, wilson_ci


CONDS = [
    "latent_mas",
    "exp_m_identity_wa",
    "latent_mas_random_wa_orth",
    "latent_mas_zero_wa",
    "no_transfer",
]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--activations_dir", required=True, type=Path)
    args = p.parse_args()
    common.setup_logging()
    out = common.results_dir(args.activations_dir, "exp_m")
    results = {}

    for task in common.TASKS:
        per_cond = {}
        metas_by_cond = {}
        for cond in CONDS:
            metas = {ex.idx: ex.meta for ex in common.iter_examples(
                args.activations_dir, cond, task)}
            if not metas:
                continue
            n = len(metas)
            n_correct = sum(1 for m in metas.values() if m.get("correct"))
            lo, hi = wilson_ci(n_correct, n)
            per_cond[cond] = {"n": n, "n_correct": n_correct,
                              "accuracy": n_correct / n,
                              "ci_lo": lo, "ci_hi": hi}
            metas_by_cond[cond] = metas

        # paired McNemar: every other vs latent_mas
        pairs = {}
        if "latent_mas" in metas_by_cond:
            ref = metas_by_cond["latent_mas"]
            for cond in CONDS:
                if cond == "latent_mas" or cond not in metas_by_cond:
                    continue
                other = metas_by_cond[cond]
                shared = sorted(set(ref) & set(other))
                if not shared:
                    pairs[cond] = {"n_paired": 0}
                    continue
                a = [bool(ref[i].get("correct")) for i in shared]
                b = [bool(other[i].get("correct")) for i in shared]
                # mcnemar(a,b): a=ref(latent_mas), b=cond → diff = cond - lmas
                pairs[cond] = mcnemar(a, b)

        # M4: convergence rate (flocking coefficient) per condition
        m4 = {}
        for cond in CONDS:
            from analysis.exp_g_groupthink import _flocking
            X, _ = common.stack_post_aligned(args.activations_dir, cond, task,
                                              limit=200)
            if X.size == 0 or X.shape[1] < 2:
                continue
            F = np.stack([_flocking(X, t) for t in range(X.shape[2])], axis=1)
            m4[cond] = {f"round_{t}": float(F[:, t].mean()) for t in range(F.shape[1])}

        # M5: no_transfer vs identity_wa decomposition
        decomp = {}
        if "no_transfer" in per_cond and "exp_m_identity_wa" in per_cond:
            decomp = {
                "no_transfer_acc": per_cond["no_transfer"]["accuracy"],
                "identity_wa_acc": per_cond["exp_m_identity_wa"]["accuracy"],
                "trained_wa_acc": per_cond.get("latent_mas", {}).get("accuracy"),
                "interpretation": (
                    "If no_transfer ≈ identity_wa: KV channel carries no info, only W_a matters. "
                    "If no_transfer << identity_wa: KV content is load-bearing even with trivial W_a."
                ),
            }

        results[task] = {"accuracy": per_cond,
                         "paired_vs_latent_mas": pairs,
                         "M4_flocking": m4,
                         "M5_decomposition": decomp}

    (out / "exp_m.json").write_text(json.dumps(results, indent=2))
    print(f"[exp_m] wrote {out/'exp_m.json'}")


if __name__ == "__main__":
    main()
