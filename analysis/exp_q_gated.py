"""Exp Q — gated communication (post-hoc).

Conditions involved (all run by final_run.py):
  vanilla    : latent_mas
  topk_gated : Q1 (top-k subspace projection on W_a)
  random_gated : Q control (random fallback at same rate)
  confidence_gated : Q2 (probe-gated; needs Exp P probe artifacts)
  cot_matched, single_agent_latent_sampled : confidence-gate baseline

For each gating variant we report:
  - paired McNemar vs latent_mas overall + per bucket
  - paired McNemar vs random gate (significance per ROADMAP Q controls)
  - difficulty-tier breakdown (Q3): easy/medium/hard by gold-solution length

ROADMAP success criterion: probe gate (Q2) must beat random gate AND
confidence gate by McNemar p<0.05 on paired examples.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from analysis import common
from analysis.stats import mcnemar, wilson_ci


def _difficulty_tier(meta: dict, task: str) -> str:
    if task == "gsm8k":
        sol = meta.get("question", "")
        n_eq = sol.count("=")
        if n_eq <= 2:
            return "easy"
        if n_eq <= 4:
            return "medium"
        return "hard"
    elif task == "mbppplus":
        # use prompt length as proxy
        ql = len(meta.get("question", "").split())
        if ql < 50:
            return "easy"
        if ql < 100:
            return "medium"
        return "hard"
    else:  # arc_challenge
        return "medium"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--activations_dir", required=True, type=Path)
    args = p.parse_args()
    common.setup_logging()
    out = common.results_dir(args.activations_dir, "exp_q")
    results = {}

    GATES = ["topk_gated", "random_gated", "confidence_gated"]
    for task in common.TASKS:
        per_task = {}
        ref = {ex.idx: ex.meta for ex in common.iter_examples(
            args.activations_dir, "latent_mas", task)}
        b = common.load_buckets(str(args.activations_dir), task)

        for gate in GATES:
            metas = {ex.idx: ex.meta for ex in common.iter_examples(
                args.activations_dir, gate, task)}
            if not metas:
                continue
            shared = sorted(set(ref) & set(metas))
            if not shared:
                continue
            a = [bool(ref[i].get("correct")) for i in shared]
            g = [bool(metas[i].get("correct")) for i in shared]
            mc = mcnemar(a, g)

            # per-bucket breakdown
            per_bucket = {}
            for bk in (1, 2, 3, 4):
                idxs = [i for i in shared if b.get(i) == bk]
                if not idxs:
                    continue
                aa = [bool(ref[i].get("correct")) for i in idxs]
                gg = [bool(metas[i].get("correct")) for i in idxs]
                per_bucket[f"B{bk}"] = mcnemar(aa, gg)

            # difficulty breakdown
            per_diff = defaultdict(lambda: {"a": [], "g": []})
            for i in shared:
                tier = _difficulty_tier(ref[i], task)
                per_diff[tier]["a"].append(bool(ref[i].get("correct")))
                per_diff[tier]["g"].append(bool(metas[i].get("correct")))
            per_diff_out = {tier: mcnemar(d["a"], d["g"]) for tier, d in per_diff.items()
                            if len(d["a"]) >= 5}

            per_task[gate] = {
                "n_paired": len(shared),
                "vs_lmas_overall": mc,
                "vs_lmas_per_bucket": per_bucket,
                "vs_lmas_per_difficulty": per_diff_out,
            }

        # confidence_gated vs random_gated and vs cot_matched (gating controls)
        if "confidence_gated" in per_task:
            cg = {ex.idx: ex.meta for ex in common.iter_examples(
                args.activations_dir, "confidence_gated", task)}
            for ctrl in ("random_gated", "cot_matched", "single_agent_latent_sampled"):
                ctrl_metas = {ex.idx: ex.meta for ex in common.iter_examples(
                    args.activations_dir, ctrl, task)}
                shared = sorted(set(cg) & set(ctrl_metas))
                if not shared:
                    continue
                a = [bool(ctrl_metas[i].get("correct")) for i in shared]
                g = [bool(cg[i].get("correct")) for i in shared]
                # mcnemar(a,b): a=ctrl, b=cg → diff_pp = cg - ctrl
                per_task.setdefault("confidence_gated_controls", {})[ctrl] = {
                    "n_paired": len(shared),
                    **mcnemar(a, g),
                }

        results[task] = per_task

    (out / "exp_q.json").write_text(json.dumps(results, indent=2))
    print(f"[exp_q] wrote {out/'exp_q.json'}")


if __name__ == "__main__":
    main()
