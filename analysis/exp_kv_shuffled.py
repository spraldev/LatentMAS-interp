"""Exp KV-Shuffled — content vs structural confound control.

kv_shuffled injects KV caches from a *different random example*, breaking
semantic content while preserving format. Comparing vs latent_mas tells us
whether the content of the latent channel matters (not just its presence).

Comparisons:
  kv_shuffled vs latent_mas   — does content matter?
  kv_shuffled vs kv_blocked   — does having *any* KV help vs none?
  kv_shuffled vs no_transfer  — same as above from a different angle

Output: results/exp_kv_shuffled/exp_kv_shuffled.json
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from analysis import common
from analysis.stats import mcnemar, wilson_ci


def _difficulty_tier(meta: dict, task: str) -> str:
    if task == "gsm8k":
        n_eq = meta.get("question", "").count("=")
        if n_eq <= 2:
            return "easy"
        if n_eq <= 4:
            return "medium"
        return "hard"
    elif task == "mbppplus":
        ql = len(meta.get("question", "").split())
        if ql < 50:
            return "easy"
        if ql < 100:
            return "medium"
        return "hard"
    else:
        return "medium"


def _paired(ref_metas: dict, other_metas: dict, buckets: dict, task: str) -> dict:
    shared = sorted(set(ref_metas) & set(other_metas))
    if not shared:
        return {"n_paired": 0, "skipped": True}
    a = [bool(ref_metas[i].get("correct")) for i in shared]
    b = [bool(other_metas[i].get("correct")) for i in shared]
    overall = mcnemar(a, b)

    per_bucket = {}
    for bk in (1, 2, 3, 4):
        idxs = [i for i in shared if buckets.get(i) == bk]
        if len(idxs) < 5:
            continue
        aa = [bool(ref_metas[i].get("correct")) for i in idxs]
        bb = [bool(other_metas[i].get("correct")) for i in idxs]
        per_bucket[f"B{bk}"] = mcnemar(aa, bb)

    per_diff = defaultdict(lambda: {"a": [], "b": []})
    for i in shared:
        tier = _difficulty_tier(ref_metas[i], task)
        per_diff[tier]["a"].append(bool(ref_metas[i].get("correct")))
        per_diff[tier]["b"].append(bool(other_metas[i].get("correct")))
    per_diff_out = {tier: mcnemar(d["a"], d["b"])
                    for tier, d in per_diff.items() if len(d["a"]) >= 5}

    return {
        "n_paired": len(shared),
        "vs_overall": overall,
        "vs_per_bucket": per_bucket,
        "vs_per_difficulty": per_diff_out,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--activations_dir", required=True, type=Path)
    args = p.parse_args()
    common.setup_logging()
    out = common.results_dir(args.activations_dir, "exp_kv_shuffled")

    results = {}
    for task in common.TASKS:
        buckets = common.load_buckets(str(args.activations_dir), task)

        kv_shuf = {ex.idx: ex.meta for ex in common.iter_examples(
            args.activations_dir, "kv_shuffled", task)}
        if not kv_shuf:
            print(f"  [skip] kv_shuffled/{task} not found")
            results[task] = {"skipped": True}
            continue

        lmas = {ex.idx: ex.meta for ex in common.iter_examples(
            args.activations_dir, "latent_mas", task)}
        kv_blocked = {ex.idx: ex.meta for ex in common.iter_examples(
            args.activations_dir, "kv_blocked", task)}
        no_transfer = {ex.idx: ex.meta for ex in common.iter_examples(
            args.activations_dir, "no_transfer", task)}

        # accuracy of kv_shuffled itself
        n = len(kv_shuf)
        n_correct = sum(1 for m in kv_shuf.values() if m.get("correct"))
        lo, hi = wilson_ci(n_correct, n)

        results[task] = {
            "kv_shuffled_accuracy": n_correct / n if n else 0,
            "kv_shuffled_ci": [lo, hi],
            "kv_shuffled_n": n,
            "vs_latent_mas": _paired(lmas, kv_shuf, buckets, task),
            "vs_kv_blocked": _paired(kv_blocked, kv_shuf, buckets, task),
            "vs_no_transfer": _paired(no_transfer, kv_shuf, buckets, task),
        }

        print(f"  [{task}] kv_shuffled acc={n_correct/n*100:.1f}%  n={n}")
        ov = results[task]["vs_latent_mas"].get("vs_overall", {})
        print(f"    vs latent_mas: diff={ov.get('diff_pp',0):+.1f}pp  p={ov.get('p_value',1):.4f}")

    out_path = out / "exp_kv_shuffled.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"[exp_kv_shuffled] wrote {out_path}")


if __name__ == "__main__":
    main()
