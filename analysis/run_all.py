"""Orchestrator: runs every analysis script in dependency order.

Order matters because:
  - exp_l writes basis.pt that exp_p reads
  - exp_p writes probe.joblib that final_run.py confidence_gated reads
    (re-run final_run.py --conditions confidence_gated AFTER exp_p)
  - report.py is the headline check; run it first to flag FATAL 3 early

Usage:
  python -m analysis.run_all --activations_dir /workspace/activations
  python -m analysis.run_all --activations_dir /workspace/activations --only L,P,Q
"""
from __future__ import annotations

import argparse
import importlib
import logging
import sys
import time
from pathlib import Path

from analysis import common


# (key, module-name, label)
PIPELINE = [
    ("report", "analysis.report", "Headline accuracy + bucket distribution"),
    ("M", "analysis.exp_m_wa_ablation", "Exp M — W_a ablation"),
    ("L", "analysis.exp_l_subspace", "Exp L — communication subspace"),
    ("P", "analysis.exp_p_probe", "Exp P — predictive probe"),
    ("Q", "analysis.exp_q_gated", "Exp Q — gated communication"),
    ("D", "analysis.exp_d_trajectory", "Exp D — trajectory geometry"),
    ("C", "analysis.exp_c_task_geometry", "Exp C — task-domain geometry"),
    ("H", "analysis.exp_h_faithfulness", "Exp H — faithfulness gap"),
    ("B", "analysis.exp_b_patching", "Exp B — activation patching post-hoc"),
    ("A", "analysis.exp_a_wa_mechanism", "Exp A — W_a mechanism"),
    ("E", "analysis.exp_e_role", "Exp E — role specialization"),
    ("F", "analysis.exp_f_information", "Exp F — information theory"),
    ("G", "analysis.exp_g_groupthink", "Exp G — groupthink"),
    ("I", "analysis.exp_i_error_attribution", "Exp I — error attribution"),
    ("J", "analysis.exp_j_uncertainty", "Exp J — uncertainty"),
    ("K", "analysis.exp_k_redundancy", "Exp K — redundancy"),
    ("N", "analysis.exp_n_sycophancy", "Exp N — sycophancy"),
    ("O", "analysis.exp_o_layer_routing", "Exp O — layer routing"),
]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--activations_dir", required=True, type=Path)
    p.add_argument("--only", type=str, default=None,
                   help="Comma-separated subset of keys (e.g. L,P,Q,report)")
    p.add_argument("--skip", type=str, default=None,
                   help="Comma-separated keys to skip")
    args = p.parse_args()
    common.setup_logging()
    log = logging.getLogger("run_all")

    only_set = set(args.only.split(",")) if args.only else None
    skip_set = set(args.skip.split(",")) if args.skip else set()

    summary = []
    t_total = time.time()
    for key, module, label in PIPELINE:
        if only_set and key not in only_set:
            continue
        if key in skip_set:
            continue
        log.info("=" * 70)
        log.info("[%s] %s", key, label)
        log.info("=" * 70)
        t0 = time.time()
        try:
            mod = importlib.import_module(module)
            sys.argv = [module, "--activations_dir", str(args.activations_dir)]
            mod.main()
            ok = True
            err = None
        except Exception as e:
            log.exception("[%s] failed: %s", key, e)
            ok = False
            err = str(e)
        wall = time.time() - t0
        summary.append({"key": key, "module": module, "ok": ok,
                        "wall_seconds": wall, "error": err})
        log.info("[%s] %s in %.1fs", key, "OK" if ok else "FAILED", wall)

    log.info("=" * 70)
    log.info("ALL DONE in %.1fs", time.time() - t_total)
    for s in summary:
        log.info("  %-7s %s  (%.1fs)%s",
                 s["key"], "OK" if s["ok"] else "FAIL", s["wall_seconds"],
                 f" — {s['error']}" if s["error"] else "")


if __name__ == "__main__":
    main()
