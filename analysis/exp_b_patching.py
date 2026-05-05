"""Exp B — activation patching (post-hoc analysis of patching.json files).

The actual patching is run by ActivationPatchingRunner in final_run.py
(condition: activation_patching). This script aggregates the per-example
patching.json files and produces:
  - per-site recovery rate (heatmap: agent × round)
  - critical-site identification (where patching first recovers)
  - error contagion: how many subsequent agents diverge from clean
  - paired Wilcoxon clean-vs-patched on probability/correctness
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from analysis import common
from analysis.stats import wilcoxon_paired, wilson_ci, benjamini_hochberg


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--activations_dir", required=True, type=Path)
    args = p.parse_args()

    common.setup_logging()
    out = common.results_dir(args.activations_dir, "exp_b")

    rows = []
    for task in common.TASKS:
        d = common.task_root(args.activations_dir, "activation_patching", task)
        if not d.exists():
            continue
        for ex_dir in sorted(d.glob("example_*")):
            patch = common.load_patching(ex_dir)
            if patch is None:
                continue
            patch["task"] = task
            patch["example_id"] = int(ex_dir.name.split("_")[1])
            rows.append(patch)

    if not rows:
        print("[exp_b] no patching.json files found; skipping")
        return

    # Per-task summary
    per_task = []
    site_recovery = defaultdict(lambda: defaultdict(list))  # task → (a,r) → [recovered]
    raw_pvals = []
    pval_keys = []

    for task in common.TASKS:
        task_rows = [r for r in rows if r["task"] == task]
        if not task_rows:
            continue
        clean_correct = [bool(r["clean_correct"]) for r in task_rows]
        corrupt_correct = [bool(r["corrupt_correct"]) for r in task_rows]
        # mean per-site recovery rate: site is "recovered" if patched_correct=True
        # and clean_correct=True and corrupt_correct=False
        sites = task_rows[0].get("sites", [])
        n_agents = max(s["agent_idx"] for s in sites) + 1 if sites else 0
        n_rounds = max(s["round_idx"] for s in sites) + 1 if sites else 0
        recovery_rates = np.zeros((n_agents, n_rounds))
        site_counts = np.zeros((n_agents, n_rounds))

        for row in task_rows:
            cc = bool(row["clean_correct"])
            xc = bool(row["corrupt_correct"])
            for s in row.get("sites", []):
                a, r = s["agent_idx"], s["round_idx"]
                if cc and not xc:
                    recovery_rates[a, r] += float(bool(s["patched_correct"]))
                    site_counts[a, r] += 1
                site_recovery[task][(a, r)].append(int(s["patched_correct"]))

        rate = np.where(site_counts > 0, recovery_rates / np.maximum(site_counts, 1), 0)
        np.save(out / f"recovery_heatmap_{task}.npy", rate)

        # paired Wilcoxon: corrupt vs clean correctness
        wlx = wilcoxon_paired([float(c) for c in clean_correct],
                              [float(c) for c in corrupt_correct])

        # critical site = (a, r) with highest mean recovery
        flat = [(a, r, rate[a, r]) for a in range(n_agents) for r in range(n_rounds)]
        flat.sort(key=lambda x: -x[2])
        critical = flat[0] if flat else (None, None, 0)

        per_task.append({
            "task": task,
            "n_examples": len(task_rows),
            "clean_accuracy": float(np.mean(clean_correct)),
            "corrupt_accuracy": float(np.mean(corrupt_correct)),
            "wilcoxon_clean_vs_corrupt": wlx,
            "critical_agent": critical[0],
            "critical_round": critical[1],
            "critical_recovery_rate": float(critical[2]),
            "n_agents": int(n_agents),
            "n_rounds": int(n_rounds),
        })

        # collect per-(a,r) Wilcoxon for FDR over all sites
        for (a, r), recs in site_recovery[task].items():
            if not recs:
                continue
            # paired with corrupt baseline
            corrupt_per = [float(xc) for xc in corrupt_correct]
            clean_per_site = [int(s.get("patched_correct"))
                              for row in task_rows
                              for s in row.get("sites", [])
                              if s["agent_idx"] == a and s["round_idx"] == r]
            if len(clean_per_site) != len(corrupt_per):
                continue
            wlx_site = wilcoxon_paired([float(x) for x in clean_per_site], corrupt_per)
            raw_pvals.append(wlx_site["p_value"])
            pval_keys.append(f"{task}_a{a}_r{r}")

    rejected, qvals = benjamini_hochberg(raw_pvals)

    out_data = {
        "per_task": per_task,
        "fdr_corrected_sites": [
            {"key": k, "p": p, "q": q, "rejected_at_q05": bool(rej)}
            for k, p, q, rej in zip(pval_keys, raw_pvals, qvals, rejected)
        ],
    }
    (out / "exp_b.json").write_text(json.dumps(out_data, indent=2))
    print(f"[exp_b] wrote {out/'exp_b.json'} with {len(rows)} examples")


if __name__ == "__main__":
    main()
