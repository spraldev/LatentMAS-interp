"""Headline tables: per-condition accuracy, compute, bucket distribution.

Reads metadata.json from every condition × task and produces:
  results/report/accuracy.csv          — Table 1 candidate
  results/report/compute.csv           — forward passes / wall-clock / tokens
  results/report/buckets.csv           — bucket B1/B2/B3/B4 per task
  results/report/headline_check.json   — FATAL 3 thresholds
  results/report/summary.json          — combined dump

Headline thresholds (locked in ROADMAP FATAL 3):
  LatentMAS vs Best-of-N: ≥3pp on ≥2/3 tasks (McNemar p<0.01)
  LatentMAS vs CoT       : ≥2pp on ≥2/3 tasks (McNemar p<0.05)
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np

from analysis import common
from analysis.stats import mcnemar, wilson_ci


def collect_per_example(root: Path, condition: str, task: str) -> Dict[int, Dict]:
    out: Dict[int, Dict] = {}
    for ex in common.iter_examples(root, condition, task):
        out[ex.idx] = ex.meta
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--activations_dir", required=True, type=Path)
    args = p.parse_args()

    common.setup_logging()
    out = common.results_dir(args.activations_dir, "report")

    rows_acc, rows_compute = [], []
    bucket_rows = []

    for task in common.TASKS:
        for cond in common.CORE_CONDITIONS:
            metas = collect_per_example(args.activations_dir, cond, task)
            if not metas:
                continue
            n = len(metas)
            n_correct = sum(1 for m in metas.values() if m.get("correct"))
            lo, hi = wilson_ci(n_correct, n)
            rows_acc.append({
                "condition": cond, "task": task, "n": n,
                "n_correct": n_correct,
                "accuracy": n_correct / n,
                "ci_lo": lo, "ci_hi": hi,
            })

            # compute
            fp = [m.get("compute", {}).get("forward_passes", 0) for m in metas.values()]
            wc = [m.get("compute", {}).get("wall_clock_ms", 0) for m in metas.values()]
            tk = [m.get("compute", {}).get("generated_tokens", 0) for m in metas.values()]
            mem = [m.get("compute", {}).get("gpu_mem_peak_mb", 0) for m in metas.values()]
            rows_compute.append({
                "condition": cond, "task": task, "n": n,
                "forward_passes_mean": float(np.mean(fp)) if fp else 0,
                "wall_clock_ms_mean": float(np.mean(wc)) if wc else 0,
                "wall_clock_ms_std": float(np.std(wc)) if wc else 0,
                "generated_tokens_mean": float(np.mean(tk)) if tk else 0,
                "gpu_mem_peak_mb_mean": float(np.mean(mem)) if mem else 0,
            })

        # buckets
        b = common.load_buckets(str(args.activations_dir), task)
        if b:
            c = Counter(b.values())
            n = sum(c.values())
            bucket_rows.append({
                "task": task, "n": n,
                "B1": c[1], "B2": c[2], "B3": c[3], "B4": c[4],
                "B1_pct": 100.0 * c[1] / n if n else 0,
                "B2_pct": 100.0 * c[2] / n if n else 0,
            })

    _write_csv(out / "accuracy.csv", rows_acc)
    _write_csv(out / "compute.csv", rows_compute)
    _write_csv(out / "buckets.csv", bucket_rows)

    # headline check (FATAL 3)
    headline = _check_headline(args.activations_dir)
    (out / "headline_check.json").write_text(json.dumps(headline, indent=2))

    # combined summary
    summary = {"accuracy": rows_acc, "compute": rows_compute,
               "buckets": bucket_rows, "headline": headline}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))

    # print headline check at the bottom
    print(f"\n=== HEADLINE CHECK (FATAL 3) ===")
    for k, v in headline.items():
        print(f"  {k}: {v}")
    print(f"\nResults written to {out}")


def _check_headline(root: Path) -> Dict:
    """LatentMAS vs Best-of-N (≥3pp on ≥2/3 tasks, McNemar p<0.01) and
    LatentMAS vs CoT (≥2pp on ≥2/3 tasks, McNemar p<0.05)."""
    out = {"vs_best_of_n": {}, "vs_cot": {}}
    for task in common.TASKS:
        lmas = collect_per_example(root, "latent_mas", task)
        bon = collect_per_example(root, "best_of_n", task)
        cot = collect_per_example(root, "cot_matched", task)
        # paired: only examples present in BOTH conditions
        for label, other in (("vs_best_of_n", bon), ("vs_cot", cot)):
            shared = sorted(set(lmas) & set(other))
            if not shared:
                out[label][task] = {"n_paired": 0, "skipped": True}
                continue
            a = [bool(lmas[i].get("correct")) for i in shared]
            b = [bool(other[i].get("correct")) for i in shared]
            # we report (other vs lmas) so diff_pp is "lmas - other"
            # mcnemar(a, b) gives diff = b - a; flip sign:
            mc = mcnemar(b, a)  # a=other, b=lmas → diff_pp = lmas - other
            out[label][task] = {
                "n_paired": len(shared),
                "lmas_acc": mc["b_accuracy"],
                "other_acc": mc["a_accuracy"],
                "diff_pp": mc["diff_pp"],
                "p_value": mc["p_value"],
            }
    # decision
    bon_pass = sum(1 for v in out["vs_best_of_n"].values()
                   if isinstance(v, dict) and v.get("diff_pp", 0) >= 3
                   and v.get("p_value", 1) < 0.01)
    cot_pass = sum(1 for v in out["vs_cot"].values()
                   if isinstance(v, dict) and v.get("diff_pp", 0) >= 2
                   and v.get("p_value", 1) < 0.05)
    out["vs_best_of_n_tasks_pass"] = bon_pass
    out["vs_cot_tasks_pass"] = cot_pass
    out["headline_passed"] = (bon_pass >= 2) and (cot_pass >= 2)
    return out


def _write_csv(path: Path, rows: List[Dict]) -> None:
    if not rows:
        path.write_text("")
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


if __name__ == "__main__":
    main()
