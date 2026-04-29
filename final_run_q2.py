"""
final_run_q2.py — Exp Q2: confidence-gated LatentMAS.

Run AFTER final_run.py AND after training the Exp P probe offline.

The probe predicts per-example whether cross-agent communication will help
(i.e. predicts Bucket 1 membership from round-1 post-W_a latent_thoughts).
If the probe predicts the example will NOT benefit (low Bucket-1 probability),
fall back to single-agent CoT for that example.

Usage:
  # minimal — probe auto-discovered at <output_dir>/exp_p_probe.pkl
  python final_run_q2.py --output_dir /kaggle/working/activations

  # explicit probe path
  python final_run_q2.py \\
      --output_dir /kaggle/working/activations \\
      --probe_path /kaggle/working/exp_p_probe.pkl \\
      --probe_threshold 0.4

Output: <output_dir>/confidence_gated/<task>/example_XXXX/metadata.json
        Same format as final_run.py conditions — drop-in for McNemar analysis.

Probe file format (exp_p_probe.pkl):
  A sklearn-compatible classifier with .predict_proba(X) where X is shape
  [N, D] of post-W_a round-1 hidden states (float32). Produced by the
  offline Exp P analysis script.
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

# reuse everything from final_run
from final_run import (
    CaptureConfig,
    ComputeAccount,
    InstrumentedLatentMAS,
    KVInterventionPolicy,
    LatentMASCondition,
    SingleAgentTextRunner,
    assign_buckets,
    dir_size_bytes,
    fmt_bytes,
    gpu_mem_peak_mb,
    approx_flops,
    load_task,
    lock_splits,
    make_wa_override,
    reset_gpu_mem_peak,
    score_with_safe_exec,
    setup_logging,
)
from models import ModelWrapper, _past_length
from utils import auto_device, set_seed

log = logging.getLogger("final_run_q2")


def load_probe(probe_path: str):
    with open(probe_path, "rb") as f:
        return pickle.load(f)


def get_round1_hidden(latent_thoughts_path: Path) -> Optional[np.ndarray]:
    """Load round-1 post-W_a hidden from a saved latent_thoughts.pt.
    Returns shape [D] as float32 numpy, or None if file missing."""
    if not latent_thoughts_path.exists():
        return None
    try:
        d = torch.load(latent_thoughts_path, map_location="cpu", weights_only=False)
        # post_aligned shape: [n_agents, latent_steps, D]
        post = d.get("post_aligned")
        if post is None:
            return None
        # use agent 0, step 0 (round 1 of planner)
        return post[0, 0, :].float().numpy()
    except Exception:
        return None


class ConfidenceGatedRunner:
    """Per-example: run probe on saved round-1 hidden → if prob < threshold,
    fall back to single-agent; otherwise run full LatentMAS."""

    def __init__(
        self,
        mw: ModelWrapper,
        args: argparse.Namespace,
        cfg: CaptureConfig,
        probe,
        threshold: float,
        lmas_out_dir: Path,
    ):
        self.mw = mw
        self.args = args
        self.cfg = cfg
        self.probe = probe
        self.threshold = threshold
        self.lmas_out_dir = lmas_out_dir  # path to latent_mas/<task>/ from main run
        self.lmas_cond = LatentMASCondition(mw, args, cfg, wa_mode="trained", kv_mode="normal")
        self.sa = SingleAgentTextRunner(mw, args)

    def run_one(self, item: Dict, idx: int, ex_dir: Path) -> Dict:
        ex_dir.mkdir(parents=True, exist_ok=True)
        task = self.args.task_current

        # try to load saved round-1 hidden from prior latent_mas run
        prior_pt = self.lmas_out_dir / f"example_{idx:04d}" / "latent_thoughts.pt"
        h = get_round1_hidden(prior_pt)

        gate = "lmas"  # default: run full LatentMAS
        probe_prob = None
        if h is not None:
            try:
                prob = float(self.probe.predict_proba(h.reshape(1, -1))[0, 1])
                probe_prob = prob
                if prob < self.threshold:
                    gate = "single_agent_fallback"
            except Exception as e:
                log.warning("[q2] probe failed on idx=%d: %s", idx, e)

        reset_gpu_mem_peak()
        t0 = time.time()

        if gate == "single_agent_fallback":
            text, accs = self.sa.generate_one(
                item["question"], max_new_tokens=self.args.max_new_tokens,
                temperature=self.args.temperature, top_p=self.args.top_p,
                do_sample=True,
            )
            pred, gold, ok = score_with_safe_exec(task, item, text)
            meta = {
                "question": item["question"], "gold": gold,
                "prediction": pred, "raw_prediction": text,
                "correct": bool(ok), "task": task,
                "model_name": self.args.model_name,
                "gate": gate, "probe_prob": probe_prob,
                "threshold": self.threshold,
                "compute": accs.asdict(),
            }
            (ex_dir / "metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
        else:
            meta = self.lmas_cond.run_and_capture(item, ex_dir)
            meta["gate"] = gate
            meta["probe_prob"] = probe_prob
            meta["threshold"] = self.threshold
            # rewrite metadata with gate fields
            (ex_dir / "metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))

        return meta


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", required=True)
    p.add_argument("--probe_path", default=None,
                   help="Path to pickled sklearn probe from Exp P. "
                        "Defaults to <output_dir>/exp_p_probe.pkl.")
    p.add_argument("--probe_threshold", type=float, default=0.4,
                   help="Fall back to single-agent if Bucket-1 prob < threshold.")
    p.add_argument("--model_name", default="Qwen/Qwen3-4B")
    p.add_argument("--tasks", nargs="+", default=["gsm8k", "arc_challenge", "mbppplus"])
    p.add_argument("--max_samples", type=int, default=200)
    p.add_argument("--latent_steps", type=int, default=4)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--split", type=str, default="test")
    p.add_argument("--max_new_tokens", type=int, default=2048)
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--latent_space_realign", action="store_true")
    p.add_argument("--no_layer_hidden", action="store_true")
    p.add_argument("--no_kv_latent", action="store_true")
    p.add_argument("--save_kv_full", action="store_true")
    p.add_argument("--prompt_hidden_last", type=int, default=64)
    p.add_argument("--decode_latent_topk", type=int, default=5)
    p.add_argument("--storage_warn_gb", type=float, default=18.0)
    p.add_argument("--log_level", default="INFO")
    p.add_argument("--log_file", default=None)
    # ModelWrapper compat
    p.add_argument("--use_vllm", action="store_true")
    p.add_argument("--use_second_HF_model", action="store_true")
    p.add_argument("--enable_prefix_caching", action="store_true")
    p.add_argument("--device2", type=str, default="cuda:1")
    p.add_argument("--tensor_parallel_size", type=int, default=1)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    p.add_argument("--method", type=str, default="latent_mas")
    p.add_argument("--generate_bs", type=int, default=1)
    p.add_argument("--latent_only", action="store_true")
    p.add_argument("--sequential_info_only", action="store_true")
    p.add_argument("--text_mas_context_length", type=int, default=-1)
    p.add_argument("--think", action="store_true")
    p.add_argument("--prompt", choices=["sequential", "hierarchical"], default="sequential")

    args = p.parse_args()
    args.use_vllm = False

    set_seed(args.seed)
    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    log_file = Path(args.log_file) if args.log_file else (out_root / "run_q2.log")
    setup_logging(log_file, args.log_level)

    probe_path = Path(args.probe_path) if args.probe_path else (out_root / "exp_p_probe.pkl")
    if not probe_path.exists():
        log.error(
            "Probe file not found: %s\n"
            "  Run final_run.py first to collect latent_mas activations, then train the\n"
            "  Exp P probe offline (sklearn classifier on post-W_a round-1 hiddens) and\n"
            "  save it to %s (or pass --probe_path <path>).",
            probe_path, probe_path,
        )
        raise SystemExit(1)

    log.info("loading probe from %s", probe_path)
    probe = load_probe(str(probe_path))
    log.info("probe loaded: %s  threshold=%.2f", type(probe).__name__, args.probe_threshold)

    device = auto_device(args.device)
    log.info("loading model %s on %s", args.model_name, device)
    mw = ModelWrapper(args.model_name, device, use_vllm=False, args=args)

    cfg = CaptureConfig(
        save_attention=True,
        save_all_layer_hidden=not args.no_layer_hidden,
        save_kv_latent_only=not args.no_kv_latent,
        save_kv_full=args.save_kv_full,
        save_prompt_hidden_last=args.prompt_hidden_last,
        decode_latent_topk=args.decode_latent_topk,
        storage_warn_gb=args.storage_warn_gb,
    )

    cond_root = out_root / "confidence_gated"
    cond_root.mkdir(parents=True, exist_ok=True)

    for task in args.tasks:
        args.task_current = task
        args.task = task
        task_dir = cond_root / task
        task_dir.mkdir(parents=True, exist_ok=True)
        lmas_task_dir = out_root / "latent_mas" / task

        items = load_task(task, args.split, args.max_samples)
        log.info("[confidence_gated/%s] %d examples  lmas_dir=%s", task, len(items), lmas_task_dir)

        try:
            lock_splits(out_root, task, len(items), seed=args.seed)
        except Exception as e:
            log.warning("split lock failed: %s", e)

        runner = ConfidenceGatedRunner(
            mw, args, cfg, probe,
            threshold=args.probe_threshold,
            lmas_out_dir=lmas_task_dir,
        )

        meta_rows = []
        fallback_count = 0

        for idx, item in enumerate(tqdm(items, desc=f"confidence_gated/{task}")):
            ex_dir = task_dir / f"example_{idx:04d}"
            if (ex_dir / "metadata.json").exists():
                try:
                    meta_rows.append({"example_id": idx,
                                      **json.loads((ex_dir / "metadata.json").read_text())})
                except Exception:
                    pass
                continue
            try:
                meta = runner.run_one(item, idx, ex_dir)
                if meta.get("gate") == "single_agent_fallback":
                    fallback_count += 1
                meta_rows.append({"example_id": idx, **meta})
            except Exception as e:
                log.exception("[error] task=%s idx=%d: %s", task, idx, e)

        fallback_rate = fallback_count / max(len(items), 1)
        log.info("[confidence_gated/%s] fallback_rate=%.1f%%  disk=%s",
                 task, fallback_rate * 100, fmt_bytes(dir_size_bytes(cond_root)))

        if meta_rows:
            import csv
            keys = sorted({k for row in meta_rows for k in row.keys()
                           if not isinstance(row.get(k), (dict, list))})
            with (task_dir / "metadata.csv").open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=keys)
                w.writeheader()
                for row in meta_rows:
                    w.writerow({k: row.get(k, "") for k in keys})

    log.info("Q2 COMPLETE  out=%s", out_root)


if __name__ == "__main__":
    main()
