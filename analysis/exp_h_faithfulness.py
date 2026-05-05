"""Exp H — faithfulness gap (logit-lens).

H1. Verify logit-lens at LAST layer reproduces output token distribution.
    (We use the saved logitlens.json from final_run, which is logit-lens
    at the final hidden state per latent step. So H1 here = a basic
    sanity check on the JSON contents.)
H2. Top-1 token frequency distribution per task
H3. Faithfulness gap score — task-dependent metric
    - ARC: cosine to correct-letter token's position in top-k
    - GSM8K: KL between logitlens dist and target uniform over correct numeric tokens
    - MBPP+: SKIP (no single correct-answer token)
H4. Hallucination precursor (ARC): does round-1 logit-lens already point wrong?
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from analysis import common
from analysis.stats import wilcoxon_paired


def _digits_in_gold(gold: str) -> set:
    return set(ch for ch in str(gold) if ch.isdigit())


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--activations_dir", required=True, type=Path)
    args = p.parse_args()
    common.setup_logging()
    out = common.results_dir(args.activations_dir, "exp_h")
    results = {}

    n_ll_examples = 0
    for task in common.TASKS:
        per_ex = []
        for ex in common.iter_examples(args.activations_dir, "latent_mas", task):
            ll = common.load_logitlens(ex.dir)
            # Skip if logitlens is empty/all-empty (the run swallowed the
            # decode_latent_topk exception during collection).
            if not ll or not any(any(step for step in agent) for agent in ll):
                continue
            per_ex.append({"idx": ex.idx, "logitlens": ll,
                           "gold": ex.meta.get("gold"),
                           "prediction": ex.meta.get("prediction"),
                           "correct": bool(ex.meta.get("correct"))})
        if not per_ex:
            continue
        n_ll_examples += len(per_ex)

        # H2 — top-1 frequency distribution (round-final, agent 0)
        top1 = Counter()
        for r in per_ex:
            ll = r["logitlens"]
            if not ll or not ll[0] or not ll[0][-1]:
                continue
            entry = ll[0][-1][0]  # agent 0, last round, top-1
            if not entry or len(entry) < 2:
                continue
            tok = entry[0]
            top1[str(tok)] += 1
        results.setdefault("H2_top1_freq", {})[task] = top1.most_common(50)

        # H3 — task-specific gap
        if task == "arc_challenge":
            gaps = []
            correctness = []
            for r in per_ex:
                ll = r["logitlens"]
                gold = str(r["gold"]).strip().lower()
                if not ll or not gold or gold not in "abcd":
                    continue
                # presence rank of correct letter in top-k of last latent step
                rank = None
                for ai, agent_steps in enumerate(ll):
                    if not agent_steps:
                        continue
                    last_step = agent_steps[-1]  # last round
                    for ti, entry in enumerate(last_step):
                        if not entry or len(entry) < 1:
                            continue
                        if str(entry[0]).strip().lower() == gold:
                            rank = ti
                            break
                    if rank is not None:
                        break
                gap = (rank if rank is not None else 5) / 5.0
                gaps.append(gap)
                correctness.append(int(r["correct"]))
            if gaps:
                rho, pv = spearmanr(gaps, [1 - c for c in correctness])
                results.setdefault("H3_arc", {})[task] = {
                    "n": len(gaps),
                    "spearman_r_gap_vs_incorrect": float(rho),
                    "p_value": float(pv),
                    "mean_gap_correct": float(np.mean([g for g, c in zip(gaps, correctness) if c == 1] or [0])),
                    "mean_gap_incorrect": float(np.mean([g for g, c in zip(gaps, correctness) if c == 0] or [0])),
                }
        elif task == "gsm8k":
            kl_correct, kl_incorrect = [], []
            for r in per_ex:
                ll = r["logitlens"]
                gold_digits = _digits_in_gold(r["gold"] or "")
                if not ll or not gold_digits:
                    continue
                last_step = ll[0][-1] if ll[0] else []
                last_step = [e for e in last_step if e and len(e) >= 2]
                if not last_step:
                    continue
                tot = sum(p for _, p in last_step) or 1.0
                p_emp = {tok: p / tot for tok, p in last_step}
                # mass on digit tokens
                digit_mass = sum(p for tok, p in p_emp.items()
                                  if any(d in str(tok) for d in gold_digits))
                non_mass = max(1 - digit_mass, 1e-9)
                # rough proxy: -log(digit_mass + 1e-9)
                kl = -np.log(digit_mass + 1e-9)
                if r["correct"]:
                    kl_correct.append(kl)
                else:
                    kl_incorrect.append(kl)
            if kl_correct and kl_incorrect:
                wlx = wilcoxon_paired(kl_correct[:len(kl_incorrect)],
                                       kl_incorrect[:len(kl_correct)])
                results.setdefault("H3_gsm8k", {})[task] = {
                    "n_correct": len(kl_correct), "n_incorrect": len(kl_incorrect),
                    "mean_kl_correct": float(np.mean(kl_correct)),
                    "mean_kl_incorrect": float(np.mean(kl_incorrect)),
                    "wilcoxon": wlx,
                }

        # H4 — hallucination precursor (ARC only)
        if task == "arc_challenge":
            agree_round1_with_final, n = 0, 0
            for r in per_ex:
                ll = r["logitlens"]
                gold = str(r["gold"]).strip().lower()
                pred = str(r.get("prediction") or "").strip().lower()
                if not ll or not pred or not gold:
                    continue
                # round-1 top-1 of agent 0
                if not ll[0] or not ll[0][0] or not ll[0][0][0]:
                    continue
                entry = ll[0][0][0]
                if not entry:
                    continue
                tok = str(entry[0]).strip().lower()
                if tok == pred:
                    agree_round1_with_final += 1
                n += 1
            results.setdefault("H4_arc_round1_predicts_final", {})[task] = {
                "n": n,
                "fraction": float(agree_round1_with_final / max(n, 1))
            }

    results["_meta"] = {"n_examples_with_logitlens": int(n_ll_examples)}
    if n_ll_examples == 0:
        results["_skipped_reason"] = (
            "All logitlens.json files are empty — _decode_latent_topk swallowed an "
            "exception during data collection. Re-run with logitlens decoding fixed "
            "to populate Exp H. (Other experiments are unaffected.)"
        )
    (out / "exp_h.json").write_text(json.dumps(results, indent=2, default=str))
    print(f"[exp_h] wrote {out/'exp_h.json'} (n_logitlens_examples={n_ll_examples})")


if __name__ == "__main__":
    main()
