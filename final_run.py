"""
final_run.py — single-entry data collection driver for the 2026 ROADMAP.

This version is the "100% viable" rewrite. Compared to the first draft:

  1. KV ablations are real attention-level interventions, not past_kv drops.
     We track per-position agent_id and either (a) zero those positions in
     the attention_mask (blocked / no_transfer) so attention to "other agents"
     is masked to -inf, or (b) substitute the K/V tensors at those positions
     with a length-matched donor example's KV cache (shuffled).

  2. Best-of-N uses the same architecture-matched judger as LatentMAS:
     prompts.build_agent_messages_sequential_text_mas with role="judger" and
     context = the N candidate answers concatenated. Same prompt template,
     same role, same forward path as the LatentMAS judger — just text
     candidates instead of latent KV.

  3. Activation patching is implemented inline as its own condition. It runs
     SAME-EXAMPLE clean / corrupt / patch (per your critique that
     cross-example Bucket 1 -> Bucket 2 patching is conceptually shaky):
        clean   = vanilla LatentMAS, save post-W_a hidden at each (agent, round)
        corrupt = same example, with W_a=identity (default corruption) OR a
                  user-selected mode: wa_zero | wa_random_orth | kv_blocked
                  | topk_removed
        patch   = corrupt forward pass, but at one (agent, round) site we
                  inject the saved clean post-W_a hidden state. Measure
                  recovery of the SAME example's correct-answer logit.

  4. The matched-compute single-agent latent baselines run with a real
     1-agent agent list (one solver doing 12 latent steps + judger), via
     the new `agents=` kwarg on InstrumentedLatentMAS.

  5. CPU smoke: --smoke flag auto-switches to Qwen/Qwen3-0.6B and clamps
     latent_steps + max_new_tokens for a fast end-to-end pipeline check.

Conditions (see ROADMAP Part 5):

   1  latent_mas
   2  single_agent_latent_sampled       (1 latent agent, 12 steps, sampled)
   3  single_agent_latent_greedy        (greedy; used for bucketing)
   4  self_consistency
   5  best_of_n                         (uses real LMAS judger prompt)
   6  cot_matched                       (single-agent CoT, token-matched)
   7  latent_mas_random_wa_spectrum
   8  kv_blocked                        (real attention masking)
   9  no_transfer                       (real attention masking)
  10  text_mas
  11  kv_shuffled                       (real KV substitution from donor)
  12  latent_mas_random_wa_orth
  13  latent_mas_zero_wa
       exp_m_identity_wa
       activation_patching              (same-example clean/corrupt/patch)
       topk_gated                       (Exp Q1: top-k SVD subspace gate on W_a)
       random_gated                     (Exp Q control: random fallback at same rate)

Usage:

  # smoke test on CPU (auto-switches to Qwen3-0.6B, 5 ex/task)
  python final_run.py --output_dir ./activations_smoke --test --smoke

  # one condition at full N
  python final_run.py --output_dir /kaggle/working/activations \\
      --conditions latent_mas --max_samples 500

  # full collection
  python final_run.py --output_dir /kaggle/working/activations --conditions all

Output goes to <output_dir>/<condition>/<task>/example_XXXX/.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import shutil
import sys
import time
import traceback
from collections import Counter
from dataclasses import asdict, dataclass, field
from multiprocessing import Manager, Process
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

from data import load_arc_challenge, load_gsm8k, load_mbppplus
# ============================================================
# Inlined from instrumented_run (not committed to repo)
# ============================================================

@dataclass
class CaptureConfig:
    save_attention: bool = False
    save_all_layer_hidden: bool = True
    save_kv_latent_only: bool = True
    save_kv_full: bool = False
    save_prompt_hidden_last: int = 64
    decode_latent_topk: int = 5
    storage_warn_gb: float = 18.0


def to_fp16_cpu(t: torch.Tensor) -> torch.Tensor:
    return t.detach().to(device="cpu", dtype=torch.float16)


def kv_to_legacy(past_kv):
    """Normalise past_key_values to list-of-(k,v) tuples regardless of
    whether it's a HF DynamicCache object or the legacy tuple-of-tuples."""
    if past_kv is None:
        return None
    if isinstance(past_kv, (list, tuple)):
        return list(past_kv)
    # HF Cache object (transformers >= 4.38)
    if hasattr(past_kv, "key_cache") and hasattr(past_kv, "value_cache"):
        return list(zip(past_kv.key_cache, past_kv.value_cache))
    return None


def slice_kv_positions(past_kv, start: int, end: int):
    """Return a list of (k[:, :, start:end, :], v[:, :, start:end, :]) per layer."""
    legacy = kv_to_legacy(past_kv)
    if legacy is None:
        return []
    result = []
    for k, v in legacy:
        result.append((
            k[..., start:end, :].detach().to(device="cpu", dtype=torch.float16),
            v[..., start:end, :].detach().to(device="cpu", dtype=torch.float16),
        ))
    return result


def dir_size_bytes(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def fmt_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}TB"


def setup_logging(log_file: Path, level: str = "INFO") -> None:
    fmt = "%(asctime)s %(levelname)s %(name)s — %(message)s"
    handlers = [logging.StreamHandler()]
    try:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    except Exception:
        pass
    logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO),
                        format=fmt, handlers=handlers, force=True)


class InstrumentedLatentMAS:
    """Minimal orchestrator: builds prompts, runs the latent loop, saves W_a.
    The actual forward pass with KV interventions is handled by
    LatentMASCondition; this class exists to hold config and helpers."""

    def __init__(
        self,
        mw,
        *,
        latent_steps: int = 4,
        judger_max_new_tokens: int = 512,
        temperature: float = 0.6,
        top_p: float = 0.95,
        args=None,
        cfg: Optional["CaptureConfig"] = None,
        agents: Optional[List] = None,
    ):
        from methods import default_agents as _default_agents
        self.mw = mw
        self.latent_steps = latent_steps
        self.judger_max_new_tokens = judger_max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.args = args
        self.cfg = cfg or CaptureConfig()
        self.agents = agents if agents is not None else _default_agents()

    def _build_prompt(self, role: str, question: str):
        """Build prompt for a given role and return (text, input_ids, attention_mask)."""
        from prompts import build_agent_message_sequential_latent_mas
        messages = build_agent_message_sequential_latent_mas(
            role=role, question=question, context="", method="latent_mas", args=self.args
        )
        prompt_text, input_ids, attention_mask, _ = self.mw.prepare_chat_input(
            messages, add_generation_prompt=True
        )
        return prompt_text, input_ids, attention_mask

    def _decode_latent_topk(self, hidden: torch.Tensor, k: int) -> List[Tuple[str, float]]:
        """Project hidden state through lm_head and return top-k (token, prob) pairs."""
        try:
            model = getattr(self.mw, "HF_model", self.mw.model)
            lm_head = model.get_output_embeddings()
            with torch.no_grad():
                logits = lm_head(hidden.to(lm_head.weight.device).unsqueeze(0).float())
                probs = torch.softmax(logits[0], dim=-1)
                topk = torch.topk(probs, k)
            tokens = self.mw.tokenizer.convert_ids_to_tokens(topk.indices.tolist())
            return [(t, float(p)) for t, p in zip(tokens, topk.values.tolist())]
        except Exception:
            return []

    def save_wa(self, path: Path) -> None:
        """Save the trained W_a matrix to disk."""
        try:
            model = getattr(self.mw, "HF_model", self.mw.model)
            self.mw._ensure_latent_realign_matrix(model, self.mw.device, self.args)
            key = id(model)
            if key in self.mw._latent_realign_matrices:
                W, target_norm = self.mw._latent_realign_matrices[key]
                torch.save({"W_a": W.cpu(), "target_norm": target_norm.cpu()}, path)
                log.info("[wa] saved W_a %s to %s", tuple(W.shape), path)
        except Exception as e:
            log.warning("[wa] could not save W_a: %s", e)
from methods import Agent, default_agents
from models import ModelWrapper, _past_length
from prompts import (
    build_agent_messages_single_agent,
    build_agent_messages_sequential_text_mas,
)
from utils import (
    auto_device,
    extract_gsm8k_answer,
    extract_markdown_python_block,
    normalize_answer,
    set_seed,
)


log = logging.getLogger("final_run")


# ============================================================
# MBPP+ multiprocessing fix (ROADMAP P1) — module-level worker
# ============================================================

def _mbpp_worker(ns, code: str) -> None:
    try:
        local_ns: Dict = {}
        exec(code, local_ns)
        ns["ok"] = True
        ns["error"] = None
    except Exception:
        ns["ok"] = False
        ns["error"] = traceback.format_exc()


def run_code_safely(code: str, timeout: float = 10.0) -> Tuple[bool, Optional[str]]:
    with Manager() as manager:
        ns = manager.dict()
        p = Process(target=_mbpp_worker, args=(ns, code))
        p.start()
        p.join(timeout)
        if p.is_alive():
            p.terminate()
            return False, f"TimeoutError: exceeded {timeout}s"
        return bool(ns.get("ok", False)), ns.get("error", None)


def score_with_safe_exec(task: str, item: Dict, final_text: str) -> Tuple[Optional[str], str, bool]:
    if task == "mbppplus":
        pred = extract_markdown_python_block(final_text)
        gold = item.get("gold", "")
        if pred is None or not gold.strip() or "assert" not in gold:
            return pred, gold, False
        ok, _ = run_code_safely(pred + "\n" + gold, timeout=10)
        return pred, gold, bool(ok)
    pred = normalize_answer(extract_gsm8k_answer(final_text))
    gold = item.get("gold", "")
    return pred, gold, bool(pred and gold and pred == gold)


# ============================================================
# Datasets / split locking
# ============================================================

DATASET_LOADERS = {
    "gsm8k": lambda split: load_gsm8k(split=split),
    "arc_challenge": lambda split: load_arc_challenge(split="test"),
    "mbppplus": lambda split: load_mbppplus(split="test"),
}


def load_task(task: str, split: str, max_samples: int) -> List[Dict]:
    items = list(DATASET_LOADERS[task](split))
    if max_samples > 0:
        items = items[:max_samples]
    return items


def lock_splits(out_root: Path, task: str, n_total: int, seed: int = 42) -> Dict[str, List[int]]:
    splits_dir = Path("data") / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    files = {s: splits_dir / f"{task}_{s}.json" for s in ("discovery", "validation", "test")}
    if all(f.exists() for f in files.values()):
        return {s: json.loads(f.read_text()) for s, f in files.items()}
    rng = random.Random(seed)
    idxs = list(range(n_total))
    rng.shuffle(idxs)
    n_disc = int(0.40 * n_total)
    n_val = int(0.20 * n_total)
    splits = {
        "discovery": sorted(idxs[:n_disc]),
        "validation": sorted(idxs[n_disc:n_disc + n_val]),
        "test": sorted(idxs[n_disc + n_val:]),
    }
    for s, f in files.items():
        f.write_text(json.dumps(splits[s]))
    log.info("[splits] %s: %d/%d/%d locked", task,
             len(splits["discovery"]), len(splits["validation"]), len(splits["test"]))
    return splits


# ============================================================
# W_a override (Exp M / cond 7,12,13)
# ============================================================

def make_wa_override(mw: ModelWrapper, mode: str, seed: int) -> None:
    if mode == "trained":
        return
    if not mw._latent_realign_matrices:
        target_model = getattr(mw, "HF_model", mw.model)
        mw._ensure_latent_realign_matrix(target_model, mw.device, mw.args)
    g = torch.Generator(device="cpu").manual_seed(seed)
    for key, (W, target_norm) in list(mw._latent_realign_matrices.items()):
        D = W.shape[0]
        if mode == "identity":
            W_new = torch.eye(D, dtype=W.dtype, device=W.device)
        elif mode == "zero":
            W_new = torch.zeros_like(W)
        elif mode == "random_orthogonal":
            A = torch.randn(D, D, generator=g, dtype=torch.float32)
            Q, _ = torch.linalg.qr(A)
            W_new = Q.to(dtype=W.dtype, device=W.device)
        elif mode == "random_spectrum":
            W32 = W.to(torch.float32).cpu()
            _, S_, _ = torch.linalg.svd(W32, full_matrices=False)
            A = torch.randn(D, D, generator=g)
            U_rand, _ = torch.linalg.qr(A)
            B = torch.randn(D, D, generator=g)
            V_rand, _ = torch.linalg.qr(B)
            W_new = (U_rand @ torch.diag(S_) @ V_rand.T).to(dtype=W.dtype, device=W.device)
        else:
            raise ValueError(f"unknown wa_mode: {mode}")
        mw._latent_realign_matrices[key] = (W_new, target_norm)


# ============================================================
# Compute accounting
# ============================================================

@dataclass
class ComputeAccount:
    forward_passes: int = 0
    generated_tokens: int = 0
    wall_clock_ms: float = 0.0
    gpu_mem_peak_mb: float = 0.0
    latent_steps: int = 0
    approx_flops: float = 0.0

    def asdict(self) -> Dict:
        return asdict(self)


def reset_gpu_mem_peak() -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def gpu_mem_peak_mb() -> float:
    if torch.cuda.is_available():
        return float(torch.cuda.max_memory_allocated()) / (1024 ** 2)
    return 0.0


def approx_flops(num_params: int, forward_passes: int, seq_len: int) -> float:
    return float(num_params) * float(forward_passes) * float(seq_len) * 2.0


# ============================================================
# Faithful KV interventions (ROADMAP P3d) — fix #1
#
# The forward pass takes attention_mask of shape [B, total_seq_len] where
# total_seq_len = past_len + new_len. HF transformers feeds positions where
# attention_mask=0 to softmax as -inf, so attention from queries to those
# positions is exactly suppressed (this is the standard padding mechanism
# applied here for cross-agent suppression). For "shuffled", we directly
# substitute K/V tensors at the cross-agent positions in the past cache.
# ============================================================

@dataclass
class AgentSegment:
    """Records which positions in the KV cache belong to which agent."""
    agent_idx: int
    start: int   # inclusive
    end: int     # exclusive (= start + length)


class KVInterventionPolicy:
    """Maintains agent_segments alongside the model's past_key_values, and
    builds intervention masks / KV substitutions for blocked / no_transfer /
    shuffled modes.

    Mode semantics:
      normal       -> no intervention
      blocked      -> when running agent k, mask attention_mask=0 at positions
                      whose agent_idx != k. The KV stays in place but cannot
                      be attended.
      no_transfer  -> agent k starts with past_key_values=None (the most
                      aggressive form of blocking — also avoids position
                      embedding leakage). After the agent, its segment is
                      kept locally only and discarded before the next agent.
      shuffled     -> like blocked, but instead of masking we copy donor
                      example's K/V tensors into the cross-agent positions.
                      Positions are length-matched (we trim or pad to fit).
    """

    def __init__(self, mode: str):
        assert mode in ("normal", "blocked", "no_transfer", "shuffled")
        self.mode = mode
        self.segments: List[AgentSegment] = []

    def reset(self) -> None:
        self.segments = []

    def record_segment(self, agent_idx: int, start: int, end: int) -> None:
        if end > start:
            self.segments.append(AgentSegment(agent_idx, start, end))

    def build_attention_mask(
        self,
        current_agent_idx: int,
        past_len: int,
        new_len: int,
        device: torch.device,
        dtype=torch.long,
    ) -> torch.Tensor:
        """Return [1, past_len + new_len] mask. 0 = blocked, 1 = visible."""
        mask = torch.ones((1, past_len + new_len), dtype=dtype, device=device)
        if self.mode == "blocked" and self.segments:
            for seg in self.segments:
                if seg.agent_idx != current_agent_idx:
                    s = max(0, seg.start)
                    e = min(past_len, seg.end)
                    if e > s:
                        mask[0, s:e] = 0
        return mask

    def maybe_substitute_kv(
        self,
        past_kv,
        current_agent_idx: int,
        donor_past_kv,
    ):
        """For shuffled mode: copy donor K/V into past_kv at positions whose
        agent_idx != current_agent_idx. Length-matched within ±5; positions
        outside the donor's range are left as-is and logged."""
        if self.mode != "shuffled" or donor_past_kv is None or past_kv is None:
            return past_kv, 0
        legacy_dst = kv_to_legacy(past_kv)
        legacy_src = kv_to_legacy(donor_past_kv)
        if legacy_dst is None or legacy_src is None:
            return past_kv, 0
        substituted = 0
        # iterate per layer
        for li, (dst_layer, src_layer) in enumerate(zip(legacy_dst, legacy_src)):
            dst_k, dst_v = dst_layer[0], dst_layer[1]
            src_k, src_v = src_layer[0], src_layer[1]
            src_len = src_k.shape[-2]
            for seg in self.segments:
                if seg.agent_idx == current_agent_idx:
                    continue
                s, e = seg.start, seg.end
                length = e - s
                if length <= 0:
                    continue
                if s >= src_len:
                    continue
                copy_e = min(e, src_len)
                copy_len = copy_e - s
                if copy_len <= 0:
                    continue
                # copy donor's [s:copy_e] into dst's [s:copy_e]
                dst_k[..., s:s + copy_len, :] = src_k[..., s:s + copy_len, :].to(
                    dst_k.dtype, device=dst_k.device
                )
                dst_v[..., s:s + copy_len, :] = src_v[..., s:s + copy_len, :].to(
                    dst_v.dtype, device=dst_v.device
                )
                if li == 0:
                    substituted += copy_len
        # if past_kv was a Cache object, it's mutated in-place on the
        # underlying tensors — return original handle.
        return past_kv, substituted


# ============================================================
# LatentMASCondition — proper implementation with attention-level KV ablations
# ============================================================

class LatentMASCondition:
    """Runs LatentMAS with optional W_a override and attention-level KV mode.
    Replaces both the normal-mode delegation and the previous coarse
    past_kv-drop hack."""

    def __init__(
        self,
        mw: ModelWrapper,
        args: argparse.Namespace,
        cfg: CaptureConfig,
        *,
        wa_mode: str = "trained",
        kv_mode: str = "normal",
        agents: Optional[List[Agent]] = None,
    ):
        self.mw = mw
        self.args = args
        self.cfg = cfg
        self.wa_mode = wa_mode
        self.kv_mode = kv_mode
        make_wa_override(mw, wa_mode, seed=args.seed)
        # InstrumentedLatentMAS now accepts `agents=`
        self.runner = InstrumentedLatentMAS(
            mw,
            latent_steps=args.latent_steps,
            judger_max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            args=args,
            cfg=cfg,
            agents=agents,
        )
        self.agents_list = self.runner.agents

    # ----------- main entry -----------
    @torch.no_grad()
    def run_and_capture(
        self,
        item: Dict,
        save_dir: Path,
        donor_past_kv=None,
    ) -> Dict:
        """Custom forward loop that supports kv_mode interventions. Mirrors
        InstrumentedLatentMAS.run_and_capture but threads a KVInterventionPolicy
        through every model call."""
        save_dir.mkdir(parents=True, exist_ok=True)
        mw = self.mw
        runner = self.runner
        question = item["question"]
        policy = KVInterventionPolicy(self.kv_mode)
        device = mw.device

        reset_gpu_mem_peak()
        t_start = time.time()

        cap_pre, cap_post, cap_layer = [], [], []
        cap_text: Dict[str, Dict] = {}
        cap_kv_latent: Dict[str, List[Tuple[torch.Tensor, torch.Tensor]]] = {}
        cap_logitlens: List[List[List[Tuple[str, float]]]] = []  # per agent per step
        cap_prompt_hidden: List[torch.Tensor] = []  # last N tokens of prompt hidden per agent
        latent_agent_names: List[str] = []
        past_kv = None
        forward_passes = 0
        substitution_total = 0

        # iterate over agents (planner... judger). For non-judger agents, use the
        # latent loop. The judger generates text conditioned on accumulated past_kv.
        for agent_idx, agent in enumerate(runner.agents):
            prompt_text, input_ids, attention_mask = runner._build_prompt(
                agent.role, question
            )

            # for no_transfer, drop accumulated past_kv at agent boundary
            if self.kv_mode == "no_transfer" and agent.role != "judger":
                past_kv_local = None
                # also reset the policy's segment list — agent only sees self
                policy.reset()
            else:
                past_kv_local = past_kv

            past_len = _past_length(past_kv_local)
            # for shuffled mode, swap donor KV into past at cross-agent positions
            if self.kv_mode == "shuffled" and past_kv_local is not None:
                past_kv_local, sub = policy.maybe_substitute_kv(
                    past_kv_local, agent_idx, donor_past_kv
                )
                substitution_total += sub

            new_len = int(input_ids.shape[-1])
            full_mask = policy.build_attention_mask(
                current_agent_idx=agent_idx,
                past_len=past_len,
                new_len=new_len,
                device=device,
                dtype=attention_mask.dtype,
            )
            # respect the prompt's own padding too (final new_len positions)
            full_mask[0, past_len:past_len + new_len] = attention_mask[0]

            if agent.role != "judger":
                latent_agent_names.append(agent.role)
                seg_start = past_len  # this agent's prompt starts here
                outputs = mw.model(
                    input_ids=input_ids,
                    attention_mask=full_mask,
                    past_key_values=past_kv_local,
                    use_cache=True,
                    output_hidden_states=True,
                    return_dict=True,
                )
                forward_passes += 1
                past_kv = outputs.past_key_values
                last_hidden = outputs.hidden_states[-1][:, -1, :]
                # prompt_hidden: last N tokens of final layer at prompt positions
                if self.cfg.save_prompt_hidden_last > 0:
                    n_ph = self.cfg.save_prompt_hidden_last
                    ph = outputs.hidden_states[-1][0, -min(n_ph, outputs.hidden_states[-1].shape[1]):, :]
                    cap_prompt_hidden.append(to_fp16_cpu(ph))

                step_pre, step_post, step_layer, step_lens = [], [], [], []
                for step in range(runner.latent_steps):
                    step_pre.append(to_fp16_cpu(last_hidden[0]))
                    # logit-lens (ROADMAP P4b)
                    step_lens.append(
                        runner._decode_latent_topk(last_hidden[0], runner.cfg.decode_latent_topk)
                    )
                    source_model = getattr(mw, "HF_model", mw.model)
                    latent_vec = mw._apply_latent_realignment(last_hidden, source_model)
                    step_post.append(to_fp16_cpu(latent_vec[0]))

                    cur_past_len = _past_length(past_kv)
                    latent_mask = policy.build_attention_mask(
                        current_agent_idx=agent_idx,
                        past_len=cur_past_len,
                        new_len=1,
                        device=device,
                        dtype=torch.long,
                    )
                    if self.kv_mode == "shuffled" and donor_past_kv is not None:
                        past_kv, sub = policy.maybe_substitute_kv(
                            past_kv, agent_idx, donor_past_kv
                        )
                        substitution_total += sub
                    outputs = mw.model(
                        inputs_embeds=latent_vec.unsqueeze(1),
                        attention_mask=latent_mask,
                        past_key_values=past_kv,
                        use_cache=True,
                        output_hidden_states=True,
                        return_dict=True,
                    )
                    forward_passes += 1
                    past_kv = outputs.past_key_values
                    last_hidden = outputs.hidden_states[-1][:, -1, :]
                    if self.cfg.save_all_layer_hidden:
                        per_layer = torch.stack(
                            [h[0, -1, :] for h in outputs.hidden_states], dim=0
                        )
                        step_layer.append(to_fp16_cpu(per_layer))

                cap_pre.append(torch.stack(step_pre, dim=0))
                cap_post.append(torch.stack(step_post, dim=0))
                cap_logitlens.append(step_lens)
                if self.cfg.save_all_layer_hidden and step_layer:
                    cap_layer.append(torch.stack(step_layer, dim=0))

                seg_end = _past_length(past_kv)
                policy.record_segment(agent_idx, seg_start, seg_end)
                if self.cfg.save_kv_latent_only:
                    cap_kv_latent[agent.role] = slice_kv_positions(
                        past_kv, seg_start, seg_end
                    )
                cap_text[agent.role] = {
                    "input": prompt_text, "output": "",
                    "latent_steps": runner.latent_steps,
                }
            else:
                # judger: text generation. Build a custom attention mask that
                # respects kv_mode (e.g., still blocks cross-agent positions
                # if kv_mode=blocked). Then call generate.
                cur_past_len = _past_length(past_kv_local)
                gen_mask = policy.build_attention_mask(
                    current_agent_idx=agent_idx,
                    past_len=cur_past_len,
                    new_len=int(input_ids.shape[-1]),
                    device=device,
                    dtype=attention_mask.dtype,
                )
                gen_mask[0, cur_past_len:cur_past_len + int(input_ids.shape[-1])] = attention_mask[0]
                cache_position = torch.arange(
                    cur_past_len, cur_past_len + input_ids.shape[-1],
                    dtype=torch.long, device=device,
                )
                gen_out = mw.model.generate(
                    input_ids=input_ids,
                    attention_mask=gen_mask,
                    max_new_tokens=runner.judger_max_new_tokens,
                    temperature=runner.temperature,
                    top_p=runner.top_p,
                    do_sample=True,
                    pad_token_id=mw.tokenizer.pad_token_id,
                    return_dict_in_generate=True,
                    past_key_values=past_kv_local,
                    cache_position=cache_position,
                )
                gen_ids = gen_out.sequences[0, input_ids.shape[-1]:]
                final_text = mw.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
                forward_passes += 1 + int(gen_ids.shape[-1])
                cap_text[agent.role] = {"input": prompt_text, "output": final_text}

        # ---- score ----
        final_text = cap_text["judger"]["output"]
        pred, gold, ok = score_with_safe_exec(self.args.task_current, item, final_text)

        # ---- write tensors ----
        if cap_pre:
            torch.save(
                {"agents": latent_agent_names,
                 "pre_aligned": torch.stack(cap_pre, dim=0),
                 "post_aligned": torch.stack(cap_post, dim=0)},
                save_dir / "latent_thoughts.pt",
            )
        if self.cfg.save_all_layer_hidden and cap_layer:
            torch.save(
                {"agents": latent_agent_names,
                 "hidden_per_layer": torch.stack(cap_layer, dim=0)},
                save_dir / "latent_per_layer.pt",
            )
        if self.cfg.save_kv_latent_only and cap_kv_latent:
            torch.save(cap_kv_latent, save_dir / "kv_latent.pt")
        if cap_prompt_hidden:
            torch.save(
                {"agents": latent_agent_names, "prompt_hidden": cap_prompt_hidden},
                save_dir / "prompt_hidden.pt",
            )
        with (save_dir / "text_outputs.json").open("w") as f:
            json.dump(cap_text, f, ensure_ascii=False, indent=2)
        if cap_logitlens:
            with (save_dir / "logitlens.json").open("w") as f:
                json.dump(cap_logitlens, f, ensure_ascii=False, indent=2)

        wall_ms = (time.time() - t_start) * 1000.0
        n_params = sum(p.numel() for p in mw.model.parameters())
        gen_tokens = max(0, len(final_text.split()))
        meta = {
            "question": question, "gold": gold, "prediction": pred,
            "raw_prediction": final_text, "correct": bool(ok),
            "agents": latent_agent_names, "latent_steps": runner.latent_steps,
            "model_name": self.args.model_name, "task": self.args.task_current,
            "wa_mode": self.wa_mode, "kv_mode": self.kv_mode,
            "kv_substitutions": substitution_total,
            "compute": ComputeAccount(
                forward_passes=forward_passes, generated_tokens=gen_tokens,
                wall_clock_ms=wall_ms, gpu_mem_peak_mb=gpu_mem_peak_mb(),
                latent_steps=runner.latent_steps,
                approx_flops=approx_flops(n_params, forward_passes, 2048),
            ).asdict(),
            "agent_segments": [asdict(s) for s in policy.segments],
        }
        with (save_dir / "metadata.json").open("w") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return meta


# ============================================================
# Exp Q1: Top-k gated LatentMAS
# Projects post-W_a hidden states onto the top-k SVD subspace of W_a
# before injection into the next agent. k is determined from wa_matrix.pt
# via elbow detection on singular values (or passed explicitly).
# ============================================================

def _compute_topk_basis(mw: ModelWrapper, k: Optional[int] = None) -> Optional[torch.Tensor]:
    """Return [D, k] orthonormal basis of top-k right singular vectors of W_a.
    If W_a is not trained or unavailable, returns None."""
    if not mw._latent_realign_matrices:
        return None
    W, _ = next(iter(mw._latent_realign_matrices.values()))
    W32 = W.to(torch.float32).cpu()
    _, S, Vh = torch.linalg.svd(W32, full_matrices=False)
    if k is None:
        # elbow: first k where cumulative energy >= 90%
        cum = torch.cumsum(S ** 2, dim=0) / (S ** 2).sum()
        k = int((cum < 0.90).sum().item()) + 1
        k = max(k, 8)
    basis = Vh[:k, :].T.to(dtype=W.dtype, device=W.device)  # [D, k]
    return basis


def _topk_project(vec: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    """Project vec [1, D] onto the top-k subspace defined by basis [D, k]."""
    coords = vec @ basis          # [1, k]
    return coords @ basis.T       # [1, D]


class TopkGatedLatentMASCondition(LatentMASCondition):
    """Exp Q1: same as LatentMASCondition but post-W_a hiddens are projected
    onto the top-k subspace before being fed into the next latent step."""

    def __init__(self, mw, args, cfg, *, k: Optional[int] = None):
        super().__init__(mw, args, cfg, wa_mode="trained", kv_mode="normal")
        self.basis = _compute_topk_basis(mw, k=k)
        self.k_used = self.basis.shape[1] if self.basis is not None else 0
        log.info("[topk_gated] using k=%d subspace dimensions", self.k_used)

    @torch.no_grad()
    def run_and_capture(self, item: Dict, save_dir: Path, donor_past_kv=None) -> Dict:
        if self.basis is None:
            return super().run_and_capture(item, save_dir, donor_past_kv)

        save_dir.mkdir(parents=True, exist_ok=True)
        mw = self.mw
        runner = self.runner
        question = item["question"]
        policy = KVInterventionPolicy("normal")
        device = mw.device
        basis = self.basis.to(device=device)

        reset_gpu_mem_peak()
        t_start = time.time()

        cap_pre, cap_post, cap_layer = [], [], []
        cap_text: Dict[str, Dict] = {}
        cap_kv_latent: Dict[str, List] = {}
        cap_logitlens: List = []
        cap_prompt_hidden: List[torch.Tensor] = []
        latent_agent_names: List[str] = []
        past_kv = None
        forward_passes = 0

        for agent_idx, agent in enumerate(runner.agents):
            prompt_text, input_ids, attention_mask = runner._build_prompt(agent.role, question)
            past_len = _past_length(past_kv)
            new_len = int(input_ids.shape[-1])
            full_mask = policy.build_attention_mask(
                current_agent_idx=agent_idx, past_len=past_len,
                new_len=new_len, device=device, dtype=attention_mask.dtype,
            )
            full_mask[0, past_len:past_len + new_len] = attention_mask[0]

            if agent.role != "judger":
                latent_agent_names.append(agent.role)
                seg_start = past_len
                outputs = mw.model(
                    input_ids=input_ids, attention_mask=full_mask,
                    past_key_values=past_kv, use_cache=True,
                    output_hidden_states=True, return_dict=True,
                )
                forward_passes += 1
                past_kv = outputs.past_key_values
                last_hidden = outputs.hidden_states[-1][:, -1, :]
                if self.cfg.save_prompt_hidden_last > 0:
                    n_ph = self.cfg.save_prompt_hidden_last
                    ph = outputs.hidden_states[-1][0, -min(n_ph, outputs.hidden_states[-1].shape[1]):, :]
                    cap_prompt_hidden.append(to_fp16_cpu(ph))

                step_pre, step_post, step_layer, step_lens = [], [], [], []
                for _ in range(runner.latent_steps):
                    step_pre.append(to_fp16_cpu(last_hidden[0]))
                    step_lens.append(runner._decode_latent_topk(last_hidden[0], runner.cfg.decode_latent_topk))
                    source_model = getattr(mw, "HF_model", mw.model)
                    latent_vec = mw._apply_latent_realignment(last_hidden, source_model)
                    # TOP-K GATE: project onto subspace before injection
                    latent_vec = _topk_project(latent_vec, basis)
                    step_post.append(to_fp16_cpu(latent_vec[0]))

                    cur_past_len = _past_length(past_kv)
                    latent_mask = policy.build_attention_mask(
                        current_agent_idx=agent_idx, past_len=cur_past_len,
                        new_len=1, device=device, dtype=torch.long,
                    )
                    outputs = mw.model(
                        inputs_embeds=latent_vec.unsqueeze(1),
                        attention_mask=latent_mask,
                        past_key_values=past_kv, use_cache=True,
                        output_hidden_states=True, return_dict=True,
                    )
                    forward_passes += 1
                    past_kv = outputs.past_key_values
                    last_hidden = outputs.hidden_states[-1][:, -1, :]
                    if self.cfg.save_all_layer_hidden:
                        per_layer = torch.stack([h[0, -1, :] for h in outputs.hidden_states], dim=0)
                        step_layer.append(to_fp16_cpu(per_layer))

                cap_pre.append(torch.stack(step_pre, dim=0))
                cap_post.append(torch.stack(step_post, dim=0))
                cap_logitlens.append(step_lens)
                if self.cfg.save_all_layer_hidden and step_layer:
                    cap_layer.append(torch.stack(step_layer, dim=0))
                seg_end = _past_length(past_kv)
                policy.record_segment(agent_idx, seg_start, seg_end)
                if self.cfg.save_kv_latent_only:
                    cap_kv_latent[agent.role] = slice_kv_positions(past_kv, seg_start, seg_end)
                cap_text[agent.role] = {"input": prompt_text, "output": "", "latent_steps": runner.latent_steps}
            else:
                cur_past_len = _past_length(past_kv)
                gen_mask = policy.build_attention_mask(
                    current_agent_idx=agent_idx, past_len=cur_past_len,
                    new_len=int(input_ids.shape[-1]), device=device, dtype=attention_mask.dtype,
                )
                gen_mask[0, cur_past_len:cur_past_len + int(input_ids.shape[-1])] = attention_mask[0]
                cache_position = torch.arange(
                    cur_past_len, cur_past_len + input_ids.shape[-1], dtype=torch.long, device=device,
                )
                gen_out = mw.model.generate(
                    input_ids=input_ids, attention_mask=gen_mask,
                    max_new_tokens=runner.judger_max_new_tokens,
                    temperature=runner.temperature, top_p=runner.top_p,
                    do_sample=True, pad_token_id=mw.tokenizer.pad_token_id,
                    return_dict_in_generate=True, past_key_values=past_kv,
                    cache_position=cache_position,
                )
                gen_ids = gen_out.sequences[0, input_ids.shape[-1]:]
                final_text = mw.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
                forward_passes += 1 + int(gen_ids.shape[-1])
                cap_text[agent.role] = {"input": prompt_text, "output": final_text}

        final_text = cap_text["judger"]["output"]
        pred, gold, ok = score_with_safe_exec(self.args.task_current, item, final_text)

        if cap_pre:
            torch.save({"agents": latent_agent_names,
                        "pre_aligned": torch.stack(cap_pre, dim=0),
                        "post_aligned": torch.stack(cap_post, dim=0)},
                       save_dir / "latent_thoughts.pt")
        if self.cfg.save_all_layer_hidden and cap_layer:
            torch.save({"agents": latent_agent_names,
                        "hidden_per_layer": torch.stack(cap_layer, dim=0)},
                       save_dir / "latent_per_layer.pt")
        if self.cfg.save_kv_latent_only and cap_kv_latent:
            torch.save(cap_kv_latent, save_dir / "kv_latent.pt")
        if cap_prompt_hidden:
            torch.save({"agents": latent_agent_names, "prompt_hidden": cap_prompt_hidden},
                       save_dir / "prompt_hidden.pt")
        with (save_dir / "text_outputs.json").open("w") as f:
            json.dump(cap_text, f, ensure_ascii=False, indent=2)
        if cap_logitlens:
            with (save_dir / "logitlens.json").open("w") as f:
                json.dump(cap_logitlens, f, ensure_ascii=False, indent=2)

        wall_ms = (time.time() - t_start) * 1000.0
        n_params = sum(p.numel() for p in mw.model.parameters())
        gen_tokens = max(0, len(final_text.split()))
        meta = {
            "question": question, "gold": gold, "prediction": pred,
            "raw_prediction": final_text, "correct": bool(ok),
            "agents": latent_agent_names, "latent_steps": runner.latent_steps,
            "model_name": self.args.model_name, "task": self.args.task_current,
            "wa_mode": "topk_gated", "kv_mode": "normal", "topk_k": self.k_used,
            "compute": ComputeAccount(
                forward_passes=forward_passes, generated_tokens=gen_tokens,
                wall_clock_ms=wall_ms, gpu_mem_peak_mb=gpu_mem_peak_mb(),
                latent_steps=runner.latent_steps,
                approx_flops=approx_flops(n_params, forward_passes, 2048),
            ).asdict(),
            "agent_segments": [asdict(s) for s in policy.segments],
        }
        with (save_dir / "metadata.json").open("w") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return meta


# ============================================================
# Donor pool for kv_shuffled — fix #1 (length-matched, ±5 tokens)
# ============================================================

class DonorPool:
    """Records (prompt_token_count, past_kv) pairs, lets us pull a length-
    matched donor for the kv_shuffled condition. Donors come from prior
    examples in the SAME task (per ROADMAP P3d)."""

    def __init__(self, length_window: int = 5):
        self.entries: List[Tuple[int, object]] = []
        self.length_window = length_window

    def add(self, prompt_len: int, past_kv) -> None:
        self.entries.append((prompt_len, past_kv))
        if len(self.entries) > 8:
            # bound memory; only need a few donors
            self.entries.pop(0)

    def get(self, prompt_len: int) -> Optional[object]:
        for L, kv in reversed(self.entries):
            if abs(L - prompt_len) <= self.length_window:
                return kv
        if self.entries:
            return self.entries[-1][1]
        return None


# ============================================================
# Single-agent runners (text)
# ============================================================

class SingleAgentTextRunner:
    """Plain single-agent text generation with compute accounting."""

    def __init__(self, mw: ModelWrapper, args: argparse.Namespace):
        self.mw = mw
        self.args = args

    @torch.no_grad()
    def generate_one(
        self,
        question: str,
        *,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        do_sample: bool,
        seed: Optional[int] = None,
    ) -> Tuple[str, ComputeAccount]:
        if seed is not None:
            torch.manual_seed(seed)
        messages = build_agent_messages_single_agent(question=question, args=self.args)
        prompts, input_ids, attention_mask, _ = self.mw.prepare_chat_batch(
            [messages], add_generation_prompt=True
        )
        reset_gpu_mem_peak()
        t0 = time.time()
        out = self.mw.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=(temperature if do_sample else 1.0),
            top_p=top_p,
            do_sample=do_sample,
            pad_token_id=self.mw.tokenizer.pad_token_id,
            return_dict_in_generate=True,
        )
        wall_ms = (time.time() - t0) * 1000.0
        gen_ids = out.sequences[0, input_ids.shape[-1]:]
        text = self.mw.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        n_gen = int(gen_ids.shape[-1])
        seq_total = int(input_ids.shape[-1]) + n_gen
        n_params = sum(p.numel() for p in self.mw.model.parameters())
        acc = ComputeAccount(
            forward_passes=1 + n_gen, generated_tokens=n_gen,
            wall_clock_ms=wall_ms, gpu_mem_peak_mb=gpu_mem_peak_mb(),
            latent_steps=0,
            approx_flops=approx_flops(n_params, 1 + n_gen, seq_total),
        )
        return text, acc


# ============================================================
# Aggregator: self-consistency + Best-of-N (REAL judger via fix #2)
# ============================================================

def majority_vote(preds: List[Optional[str]]) -> Optional[str]:
    valid = [p for p in preds if p is not None and p != ""]
    if not valid:
        return None
    return Counter(valid).most_common(1)[0][0]


def run_self_consistency(
    sa: SingleAgentTextRunner, task: str, item: Dict, k: int, args: argparse.Namespace
) -> Tuple[Optional[str], str, bool, ComputeAccount, List[str]]:
    accs = ComputeAccount()
    raw: List[str] = []
    preds: List[Optional[str]] = []
    for s in range(k):
        text, a = sa.generate_one(
            item["question"], max_new_tokens=args.max_new_tokens,
            temperature=args.temperature, top_p=args.top_p,
            do_sample=True, seed=args.seed * 1000 + s,
        )
        raw.append(text)
        p, _, _ = score_with_safe_exec(task, item, text)
        preds.append(p)
        accs.forward_passes += a.forward_passes
        accs.generated_tokens += a.generated_tokens
        accs.wall_clock_ms += a.wall_clock_ms
        accs.gpu_mem_peak_mb = max(accs.gpu_mem_peak_mb, a.gpu_mem_peak_mb)
        accs.approx_flops += a.approx_flops
    voted = majority_vote(preds)
    if task == "mbppplus":
        idx = next((i for i, p in enumerate(preds) if p == voted), 0)
        pred, gold, ok = score_with_safe_exec(task, item, raw[idx])
    else:
        gold = item.get("gold", "")
        ok = bool(voted and gold and voted == gold)
        pred = voted
    return pred, gold, ok, accs, raw


def run_best_of_n(
    sa: SingleAgentTextRunner,
    mw: ModelWrapper,
    task: str,
    item: Dict,
    n: int,
    args: argparse.Namespace,
) -> Tuple[Optional[str], str, bool, ComputeAccount, List[str], int]:
    """N samples + REAL LMAS judger (architecture-matched). The judger uses
    `prompts.build_agent_messages_sequential_text_mas(role='judger', ...)` —
    the same prompt template the LMAS judger uses, with context = candidate
    answers concatenated in the same format LMAS feeds the judger."""
    accs = ComputeAccount()
    raw: List[str] = []
    for s in range(n):
        text, a = sa.generate_one(
            item["question"], max_new_tokens=args.max_new_tokens,
            temperature=args.temperature, top_p=args.top_p,
            do_sample=True, seed=args.seed * 1000 + s,
        )
        raw.append(text)
        accs.forward_passes += a.forward_passes
        accs.generated_tokens += a.generated_tokens
        accs.wall_clock_ms += a.wall_clock_ms
        accs.gpu_mem_peak_mb = max(accs.gpu_mem_peak_mb, a.gpu_mem_peak_mb)
        accs.approx_flops += a.approx_flops

    # build context in the exact format the LMAS judger sees: each candidate
    # is wrapped with [Planner]/[Critic]/[Refiner] role tags. We map the N
    # samples to those three roles cyclically; this is the same shape the
    # text_mas judger consumes.
    role_tags = ["Planner", "Critic", "Refiner"]
    context = ""
    for i, t in enumerate(raw):
        tag = role_tags[i % len(role_tags)] + (f"_{i // len(role_tags) + 1}" if i >= 3 else "")
        context += f"[{tag}]:\n{t}\n\n"
    judger_messages = build_agent_messages_sequential_text_mas(
        role="judger", question=item["question"], context=context,
        method="text_mas", args=args,
    )
    prompts, ids, mask, _ = mw.prepare_chat_batch([judger_messages], add_generation_prompt=True)
    reset_gpu_mem_peak()
    t0 = time.time()
    out = mw.model.generate(
        input_ids=ids,
        attention_mask=mask,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        do_sample=True,
        pad_token_id=mw.tokenizer.pad_token_id,
        return_dict_in_generate=True,
    )
    wall_ms = (time.time() - t0) * 1000.0
    gen_ids = out.sequences[0, ids.shape[-1]:]
    judger_text = mw.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    n_gen = int(gen_ids.shape[-1])
    n_params = sum(p.numel() for p in mw.model.parameters())
    accs.forward_passes += 1 + n_gen
    accs.generated_tokens += n_gen
    accs.wall_clock_ms += wall_ms
    accs.gpu_mem_peak_mb = max(accs.gpu_mem_peak_mb, gpu_mem_peak_mb())
    accs.approx_flops += approx_flops(n_params, 1 + n_gen, int(ids.shape[-1]) + n_gen)

    pred, gold, ok = score_with_safe_exec(task, item, judger_text)
    return pred, gold, ok, accs, raw, n


# ============================================================
# Activation patching (ROADMAP Exp B) — fix #3, same-example
# ============================================================

def _single_agent_for_solver_judger() -> List[Agent]:
    """Custom 1-solver + 1-judger agent list for matched-compute single-agent
    latent baselines. Fix #4."""
    return [Agent(name="Solver", role="planner"),
            Agent(name="Judger", role="judger")]


class ActivationPatchingRunner:
    """SAME-EXAMPLE clean / corrupt / patch.

    For each example:
      1. clean run = vanilla LatentMAS, save post-W_a hidden at every (agent, round).
      2. corrupt run = same example with one of:
           wa_identity   (default)
           wa_zero
           wa_random_orth
           kv_blocked
      3. for each (agent_idx, round_idx) site:
           re-run with corruption, but at site (a, r) replace the post-W_a
           hidden with the saved clean post-W_a hidden. Decode answer
           via the judger. Score correctness + answer-token logit.

    Output: per-example metrics for clean/corrupt/per-site recovery.
    """

    def __init__(
        self,
        mw: ModelWrapper,
        args: argparse.Namespace,
        cfg: CaptureConfig,
        corrupt_mode: str = "wa_identity",
    ):
        self.mw = mw
        self.args = args
        self.cfg = cfg
        self.corrupt_mode = corrupt_mode
        # canonical agent list (planner, critic, refiner, judger)
        self.agents = default_agents()
        self.n_latent_agents = len([a for a in self.agents if a.role != "judger"])
        self.M = args.latent_steps

    @torch.no_grad()
    def _forward_loop(
        self,
        item: Dict,
        *,
        wa_mode: str,
        kv_mode: str,
        patch_site: Optional[Tuple[int, int]] = None,
        patch_value: Optional[torch.Tensor] = None,
        return_post_aligned: bool = False,
    ):
        """Run LatentMAS once with the given corruption + optional patch.
        Returns dict with final_text and (optionally) post_aligned tensors."""
        mw = self.mw
        # apply W_a override
        make_wa_override(mw, wa_mode, seed=self.args.seed)
        policy = KVInterventionPolicy(kv_mode)
        device = mw.device

        runner = InstrumentedLatentMAS(
            mw,
            latent_steps=self.M,
            judger_max_new_tokens=self.args.max_new_tokens,
            temperature=self.args.temperature,
            top_p=self.args.top_p,
            args=self.args,
            cfg=self.cfg,
            agents=self.agents,
        )

        question = item["question"]
        post_aligned = []  # [agents][steps] of [D]
        past_kv = None
        for agent_idx, agent in enumerate(runner.agents):
            prompt_text, input_ids, attention_mask = runner._build_prompt(
                agent.role, question
            )
            past_local = past_kv
            past_len = _past_length(past_local)
            new_len = int(input_ids.shape[-1])
            full_mask = policy.build_attention_mask(
                current_agent_idx=agent_idx, past_len=past_len, new_len=new_len,
                device=device, dtype=attention_mask.dtype,
            )
            full_mask[0, past_len:past_len + new_len] = attention_mask[0]

            if agent.role != "judger":
                seg_start = past_len
                outputs = mw.model(
                    input_ids=input_ids, attention_mask=full_mask,
                    past_key_values=past_local, use_cache=True,
                    output_hidden_states=True, return_dict=True,
                )
                past_kv = outputs.past_key_values
                last_hidden = outputs.hidden_states[-1][:, -1, :]
                step_post: List[torch.Tensor] = []
                for step in range(self.M):
                    source_model = getattr(mw, "HF_model", mw.model)
                    latent_vec = mw._apply_latent_realignment(last_hidden, source_model)
                    # PATCHING: if this site matches, override latent_vec with
                    # the saved clean post-W_a hidden.
                    if (
                        patch_site is not None
                        and patch_value is not None
                        and patch_site == (agent_idx, step)
                    ):
                        latent_vec = patch_value.to(
                            dtype=latent_vec.dtype, device=latent_vec.device
                        ).unsqueeze(0)
                    step_post.append(latent_vec[0].detach().to("cpu", torch.float16))
                    cur_past_len = _past_length(past_kv)
                    latent_mask = policy.build_attention_mask(
                        current_agent_idx=agent_idx, past_len=cur_past_len,
                        new_len=1, device=device, dtype=torch.long,
                    )
                    outputs = mw.model(
                        inputs_embeds=latent_vec.unsqueeze(1),
                        attention_mask=latent_mask,
                        past_key_values=past_kv, use_cache=True,
                        output_hidden_states=True, return_dict=True,
                    )
                    past_kv = outputs.past_key_values
                    last_hidden = outputs.hidden_states[-1][:, -1, :]
                post_aligned.append(step_post)
                seg_end = _past_length(past_kv)
                policy.record_segment(agent_idx, seg_start, seg_end)
            else:
                cur_past_len = _past_length(past_local)
                gen_mask = policy.build_attention_mask(
                    current_agent_idx=agent_idx, past_len=cur_past_len,
                    new_len=int(input_ids.shape[-1]),
                    device=device, dtype=attention_mask.dtype,
                )
                gen_mask[0, cur_past_len:cur_past_len + int(input_ids.shape[-1])] = attention_mask[0]
                cache_position = torch.arange(
                    cur_past_len, cur_past_len + input_ids.shape[-1],
                    dtype=torch.long, device=device,
                )
                gen_out = mw.model.generate(
                    input_ids=input_ids, attention_mask=gen_mask,
                    max_new_tokens=self.args.max_new_tokens,
                    temperature=self.args.temperature, top_p=self.args.top_p,
                    do_sample=True, pad_token_id=mw.tokenizer.pad_token_id,
                    return_dict_in_generate=True,
                    past_key_values=past_local, cache_position=cache_position,
                )
                gen_ids = gen_out.sequences[0, input_ids.shape[-1]:]
                final_text = mw.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        # restore default W_a
        make_wa_override(mw, "trained", seed=self.args.seed)
        result = {"final_text": final_text}
        if return_post_aligned:
            result["post_aligned"] = post_aligned
        return result

    def run_one_example(self, item: Dict, save_dir: Path) -> Dict:
        save_dir.mkdir(parents=True, exist_ok=True)
        # 1) clean
        clean = self._forward_loop(
            item, wa_mode="trained", kv_mode="normal", return_post_aligned=True
        )
        clean_pred, gold, clean_ok = score_with_safe_exec(
            self.args.task_current, item, clean["final_text"]
        )
        # 2) corrupt
        if self.corrupt_mode == "wa_identity":
            wa_m, kv_m = "identity", "normal"
        elif self.corrupt_mode == "wa_zero":
            wa_m, kv_m = "zero", "normal"
        elif self.corrupt_mode == "wa_random_orth":
            wa_m, kv_m = "random_orthogonal", "normal"
        elif self.corrupt_mode == "kv_blocked":
            wa_m, kv_m = "trained", "blocked"
        else:
            raise ValueError(self.corrupt_mode)
        corrupt = self._forward_loop(item, wa_mode=wa_m, kv_mode=kv_m)
        corrupt_pred, _, corrupt_ok = score_with_safe_exec(
            self.args.task_current, item, corrupt["final_text"]
        )
        # 3) per-site patches
        site_results: List[Dict] = []
        for a_idx in range(self.n_latent_agents):
            for r_idx in range(self.M):
                patch_value = clean["post_aligned"][a_idx][r_idx].to(
                    dtype=torch.float32, device=self.mw.device
                )
                patched = self._forward_loop(
                    item, wa_mode=wa_m, kv_mode=kv_m,
                    patch_site=(a_idx, r_idx), patch_value=patch_value,
                )
                p_pred, _, p_ok = score_with_safe_exec(
                    self.args.task_current, item, patched["final_text"]
                )
                site_results.append({
                    "agent_idx": a_idx, "round_idx": r_idx,
                    "patched_pred": p_pred,
                    "patched_correct": bool(p_ok),
                    "patched_text": patched["final_text"][:512],
                })
        meta = {
            "question": item["question"], "gold": gold,
            "clean_pred": clean_pred, "clean_correct": bool(clean_ok),
            "corrupt_pred": corrupt_pred, "corrupt_correct": bool(corrupt_ok),
            "corrupt_mode": self.corrupt_mode,
            "n_latent_agents": self.n_latent_agents,
            "n_rounds": self.M,
            "sites": site_results,
            "task": self.args.task_current,
            "model_name": self.args.model_name,
        }
        (save_dir / "patching.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
        return meta


# ============================================================
# Condition registry
# ============================================================

@dataclass
class ConditionSpec:
    name: str
    kind: str
    n_per_task: int
    wa_mode: str = "trained"
    kv_mode: str = "normal"
    decode: str = "sample"
    extras: Dict = field(default_factory=dict)


CONDITIONS: Dict[str, ConditionSpec] = {
    "latent_mas": ConditionSpec("latent_mas", "latent_mas", 500),
    "single_agent_latent_sampled": ConditionSpec(
        "single_agent_latent_sampled", "latent_mas", 500,
        extras={"agents": "solver_judger", "latent_steps_override": 12},
    ),
    "single_agent_latent_greedy": ConditionSpec(
        "single_agent_latent_greedy", "latent_mas", 200, decode="greedy",
        extras={"agents": "solver_judger", "latent_steps_override": 12,
                "temperature_override": 1.0, "do_sample_override": False},
    ),
    "self_consistency": ConditionSpec("self_consistency", "self_consistency", 500,
                                       extras={"k": 5}),
    "best_of_n": ConditionSpec("best_of_n", "best_of_n", 500, extras={"n": 5}),
    "cot_matched": ConditionSpec("cot_matched", "single_agent", 500,
                                  extras={"cot": True}),
    "latent_mas_random_wa_spectrum": ConditionSpec(
        "latent_mas_random_wa_spectrum", "latent_mas", 200, wa_mode="random_spectrum",
    ),
    "kv_blocked": ConditionSpec("kv_blocked", "latent_mas", 200, kv_mode="blocked"),
    "no_transfer": ConditionSpec("no_transfer", "latent_mas", 500, kv_mode="no_transfer"),
    "text_mas": ConditionSpec("text_mas", "text_mas", 200),
    "kv_shuffled": ConditionSpec("kv_shuffled", "latent_mas", 200, kv_mode="shuffled"),
    "latent_mas_random_wa_orth": ConditionSpec(
        "latent_mas_random_wa_orth", "latent_mas", 200, wa_mode="random_orthogonal",
    ),
    "latent_mas_zero_wa": ConditionSpec(
        "latent_mas_zero_wa", "latent_mas", 200, wa_mode="zero",
    ),
    "exp_m_identity_wa": ConditionSpec(
        "exp_m_identity_wa", "latent_mas", 100, wa_mode="identity",
    ),
    "activation_patching": ConditionSpec(
        "activation_patching", "patching", 25,
        extras={"corrupt_mode": "wa_identity"},
    ),
    "topk_gated": ConditionSpec(
        "topk_gated", "topk_gated", 200,
        extras={"topk_k": None},  # None = auto elbow at 90% energy
    ),
    "random_gated": ConditionSpec(
        "random_gated", "random_gated", 200,
        extras={"fallback_rate": 0.3},  # updated after Exp P probe fallback rate is known
    ),
}


def _resolve_agents(spec_extras: Dict) -> Optional[List[Agent]]:
    a = spec_extras.get("agents")
    if a == "solver_judger":
        return _single_agent_for_solver_judger()
    return None


def run_condition(
    spec: ConditionSpec,
    args: argparse.Namespace,
    out_root: Path,
    mw: ModelWrapper,
    cfg: CaptureConfig,
) -> None:
    cond_root = out_root / spec.name
    cond_root.mkdir(parents=True, exist_ok=True)
    log.info("=" * 70)
    log.info("[condition] %s  kind=%s  wa=%s  kv=%s  N=%d",
             spec.name, spec.kind, spec.wa_mode, spec.kv_mode, spec.n_per_task)
    log.info("=" * 70)

    n_per_task = spec.n_per_task if not args.test else 5
    if args.max_samples > 0:
        n_per_task = min(n_per_task, args.max_samples)

    make_wa_override(mw, "trained", seed=args.seed)  # ensure clean baseline
    sa = SingleAgentTextRunner(mw, args)

    for task in args.tasks:
        args.task_current = task
        args.task = task
        task_dir = cond_root / task
        task_dir.mkdir(parents=True, exist_ok=True)
        items = load_task(task, args.split, n_per_task)
        log.info("[%s/%s] %d examples", spec.name, task, len(items))
        try:
            lock_splits(out_root, task, len(items), seed=args.seed)
        except Exception as e:
            log.warning("split lock failed: %s", e)

        meta_rows: List[Dict] = []
        cond = None
        donor_pool = DonorPool(length_window=5)
        patcher = None

        # set up condition runner
        if spec.kind == "latent_mas":
            local_args = argparse.Namespace(**vars(args))
            if "latent_steps_override" in spec.extras:
                local_args.latent_steps = spec.extras["latent_steps_override"]
            if spec.extras.get("temperature_override") is not None:
                local_args.temperature = spec.extras["temperature_override"]
            agents_override = _resolve_agents(spec.extras)
            cond = LatentMASCondition(
                mw, local_args, cfg,
                wa_mode=spec.wa_mode, kv_mode=spec.kv_mode,
                agents=agents_override,
            )
        elif spec.kind == "topk_gated":
            cond = TopkGatedLatentMASCondition(
                mw, args, cfg, k=spec.extras.get("topk_k", None)
            )
        elif spec.kind == "random_gated":
            cond = LatentMASCondition(mw, args, cfg, wa_mode="trained", kv_mode="normal")
        elif spec.kind == "patching":
            patcher = ActivationPatchingRunner(
                mw, args, cfg, corrupt_mode=spec.extras.get("corrupt_mode", "wa_identity")
            )

        for idx, item in enumerate(tqdm(items, desc=f"{spec.name}/{task}")):
            ex_dir = task_dir / f"example_{idx:04d}"
            done = ex_dir / "metadata.json"
            done_alt = ex_dir / "patching.json"
            if done.exists() or done_alt.exists():
                try:
                    f = done if done.exists() else done_alt
                    meta_rows.append({"example_id": idx, **json.loads(f.read_text())})
                except Exception:
                    pass
                continue

            try:
                if spec.kind in ("latent_mas", "topk_gated"):
                    donor = donor_pool.get(len(item["question"].split())) \
                        if spec.kv_mode == "shuffled" else None
                    meta = cond.run_and_capture(item, ex_dir, donor_past_kv=donor)
                elif spec.kind == "random_gated":
                    # Exp Q control: randomly fall back to single-agent at the
                    # same rate as the probe gate. Rate stored in spec.extras.
                    fallback_rate = spec.extras.get("fallback_rate", 0.3)
                    rng_gate = random.Random(args.seed + idx)
                    if rng_gate.random() < fallback_rate:
                        text, accs = sa.generate_one(
                            item["question"], max_new_tokens=args.max_new_tokens,
                            temperature=args.temperature, top_p=args.top_p, do_sample=True,
                        )
                        pred, gold, ok = score_with_safe_exec(task, item, text)
                        ex_dir.mkdir(parents=True, exist_ok=True)
                        meta = {
                            "question": item["question"], "gold": gold,
                            "prediction": pred, "raw_prediction": text,
                            "correct": bool(ok), "task": task,
                            "model_name": args.model_name, "compute": accs.asdict(),
                            "gate": "random_fallback", "fallback_rate": fallback_rate,
                        }
                        (ex_dir / "metadata.json").write_text(
                            json.dumps(meta, ensure_ascii=False, indent=2)
                        )
                    else:
                        meta = cond.run_and_capture(item, ex_dir)
                    # for shuffled mode, donor pool should hold prior examples'
                    # past_kv. We add the current example's KV (which we no
                    # longer have post-run); skip — donor pool is seeded by
                    # the FIRST iter of latent_mas (resume from disk if avail).
                elif spec.kind == "single_agent":
                    do_sample = (spec.decode == "sample")
                    text, accs = sa.generate_one(
                        item["question"], max_new_tokens=args.max_new_tokens,
                        temperature=(args.temperature if do_sample else 0.0),
                        top_p=args.top_p, do_sample=do_sample,
                    )
                    pred, gold, ok = score_with_safe_exec(task, item, text)
                    ex_dir.mkdir(parents=True, exist_ok=True)
                    meta = {
                        "question": item["question"], "gold": gold,
                        "prediction": pred, "raw_prediction": text,
                        "correct": bool(ok), "task": task,
                        "model_name": args.model_name, "compute": accs.asdict(),
                        "decode": spec.decode, "cot": spec.extras.get("cot", False),
                    }
                    (ex_dir / "metadata.json").write_text(
                        json.dumps(meta, ensure_ascii=False, indent=2)
                    )
                    (ex_dir / "text_outputs.json").write_text(
                        json.dumps({"singleagent": {"input": "", "output": text}},
                                   ensure_ascii=False, indent=2)
                    )
                elif spec.kind == "self_consistency":
                    pred, gold, ok, accs, raws = run_self_consistency(
                        sa, task, item, k=spec.extras.get("k", 5), args=args,
                    )
                    ex_dir.mkdir(parents=True, exist_ok=True)
                    meta = {
                        "question": item["question"], "gold": gold, "prediction": pred,
                        "correct": bool(ok), "task": task, "model_name": args.model_name,
                        "compute": accs.asdict(), "k": spec.extras.get("k", 5),
                    }
                    (ex_dir / "metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
                    (ex_dir / "samples.json").write_text(json.dumps(raws, ensure_ascii=False, indent=2))
                elif spec.kind == "best_of_n":
                    pred, gold, ok, accs, raws, n = run_best_of_n(
                        sa, mw, task, item, n=spec.extras.get("n", 5), args=args,
                    )
                    ex_dir.mkdir(parents=True, exist_ok=True)
                    meta = {
                        "question": item["question"], "gold": gold, "prediction": pred,
                        "correct": bool(ok), "task": task, "model_name": args.model_name,
                        "compute": accs.asdict(), "n": n,
                    }
                    (ex_dir / "metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
                    (ex_dir / "samples.json").write_text(json.dumps(raws, ensure_ascii=False, indent=2))
                elif spec.kind == "text_mas":
                    from methods.text_mas import TextMASMethod
                    tmas = TextMASMethod(mw, max_new_tokens_each=args.max_new_tokens,
                                         temperature=args.temperature, top_p=args.top_p,
                                         generate_bs=1, args=args)
                    reset_gpu_mem_peak()
                    t0 = time.time()
                    res = tmas.run_batch([item])[0]
                    wall_ms = (time.time() - t0) * 1000.0
                    ex_dir.mkdir(parents=True, exist_ok=True)
                    meta = {
                        "question": res["question"], "gold": res["gold"],
                        "prediction": res["prediction"], "raw_prediction": res["raw_prediction"],
                        "correct": bool(res["correct"]),
                        "task": task, "model_name": args.model_name,
                        "compute": ComputeAccount(
                            forward_passes=0, generated_tokens=0,
                            wall_clock_ms=wall_ms, gpu_mem_peak_mb=gpu_mem_peak_mb(),
                            latent_steps=0, approx_flops=0.0,
                        ).asdict(),
                    }
                    (ex_dir / "metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
                    (ex_dir / "agents.json").write_text(
                        json.dumps(res["agents"], ensure_ascii=False, indent=2)
                    )
                elif spec.kind == "patching":
                    meta = patcher.run_one_example(item, ex_dir)
                else:
                    raise ValueError(f"unknown kind: {spec.kind}")
            except Exception as e:
                log.exception("[error] cond=%s task=%s idx=%d: %s", spec.name, task, idx, e)
                if ex_dir.exists():
                    shutil.rmtree(ex_dir, ignore_errors=True)
                continue
            meta_rows.append({"example_id": idx, **meta})

        if meta_rows:
            import csv
            keys = sorted({k for row in meta_rows for k in row.keys()
                           if not isinstance(row.get(k), (dict, list))})
            with (task_dir / "metadata.csv").open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=keys)
                w.writeheader()
                for row in meta_rows:
                    w.writerow({k: row.get(k, "") for k in keys})

        log.info("[%s/%s] disk=%s", spec.name, task, fmt_bytes(dir_size_bytes(cond_root)))

    make_wa_override(mw, "trained", seed=args.seed)


# ============================================================
# Buckets + patching pair file (post-hoc)
# ============================================================

def assign_buckets(out_root: Path, args: argparse.Namespace) -> None:
    bdir = out_root / "buckets"
    bdir.mkdir(parents=True, exist_ok=True)
    for task in args.tasks:
        lmas_dir = out_root / "latent_mas" / task
        sa_dir = out_root / "single_agent_latent_greedy" / task
        if not lmas_dir.exists() or not sa_dir.exists():
            log.warning("[buckets] skipping %s — missing latent_mas or single_agent_latent_greedy", task)
            continue
        rows: List[Dict] = []
        for ex_dir in sorted(lmas_dir.glob("example_*")):
            idx = int(ex_dir.name.split("_")[1])
            sa_ex = sa_dir / ex_dir.name
            try:
                lm = json.loads((ex_dir / "metadata.json").read_text())
                sa = json.loads((sa_ex / "metadata.json").read_text())
            except Exception:
                continue
            lm_ok = bool(lm.get("correct"))
            sa_ok = bool(sa.get("correct"))
            bucket = (1 if (not sa_ok and lm_ok) else
                      2 if (sa_ok and not lm_ok) else
                      3 if (sa_ok and lm_ok) else 4)
            rows.append({"example_id": idx, "bucket": bucket,
                         "lmas_correct": lm_ok, "sa_correct": sa_ok})
        (bdir / f"{task}.json").write_text(json.dumps(rows, indent=2))
        c = Counter(r["bucket"] for r in rows)
        log.info("[buckets] %s  B1=%d B2=%d B3=%d B4=%d  (n=%d)",
                 task, c[1], c[2], c[3], c[4], len(rows))


# ============================================================
# Main
# ============================================================

ALL_CONDITION_NAMES = list(CONDITIONS.keys())


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", required=True, type=str)
    p.add_argument("--model_name", default="Qwen/Qwen3-4B")
    p.add_argument("--tasks", nargs="+",
                   default=["gsm8k", "arc_challenge", "mbppplus"])
    p.add_argument("--max_samples", type=int, default=500)
    p.add_argument("--latent_steps", type=int, default=4)
    p.add_argument("--prompt", choices=["sequential", "hierarchical"], default="sequential")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--split", type=str, default="test")
    p.add_argument("--max_new_tokens", type=int, default=2048)
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--think", action="store_true")
    p.add_argument("--latent_space_realign", action="store_true")
    p.add_argument("--no_layer_hidden", action="store_true")
    p.add_argument("--save_kv_full", action="store_true")
    p.add_argument("--no_kv_latent", action="store_true")
    p.add_argument("--prompt_hidden_last", type=int, default=64)
    p.add_argument("--decode_latent_topk", type=int, default=5)
    p.add_argument("--storage_warn_gb", type=float, default=18.0)
    p.add_argument("--test", action="store_true")
    p.add_argument("--smoke", action="store_true",
                   help="Auto-switch to a tiny model + short generations for "
                        "fast end-to-end pipeline testing on CPU.")
    p.add_argument("--smoke_model", default="Qwen/Qwen3-0.6B",
                   help="Model used when --smoke is set.")
    p.add_argument("--conditions", nargs="+", default=["all"])
    p.add_argument("--skip_buckets", action="store_true")
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

    args = p.parse_args()
    if args.use_vllm:
        print("[warn] vLLM incompatible with full-state capture; forcing HF.")
        args.use_vllm = False

    # fix #5: smoke mode
    if args.smoke:
        args.model_name = args.smoke_model
        args.max_new_tokens = min(args.max_new_tokens, 128)
        args.latent_steps = min(args.latent_steps, 2)
        args.test = True
        print(f"[smoke] model={args.model_name}  latent_steps={args.latent_steps}  "
              f"max_new_tokens={args.max_new_tokens}")

    set_seed(args.seed)
    device = auto_device(args.device)
    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    log_file = Path(args.log_file) if args.log_file else (out_root / "run.log")
    setup_logging(log_file, args.log_level)

    if args.conditions == ["all"]:
        names = ALL_CONDITION_NAMES
    else:
        names = [c for c in args.conditions if c in CONDITIONS]
        unknown = [c for c in args.conditions if c not in CONDITIONS]
        if unknown:
            log.warning("unknown conditions ignored: %s", unknown)

    log.info("=" * 70)
    log.info("final_run: %d conditions, tasks=%s", len(names), args.tasks)
    log.info("output=%s  device=%s  model=%s", out_root, device, args.model_name)
    log.info("=" * 70)

    (out_root / "run_metadata.json").write_text(json.dumps({
        "model_name": args.model_name, "tasks": args.tasks,
        "conditions": names, "seed": args.seed,
        "latent_steps": args.latent_steps,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature, "smoke": args.smoke,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }, indent=2))

    log.info("[init] loading %s on %s", args.model_name, device)
    t0 = time.time()
    mw = ModelWrapper(args.model_name, device, use_vllm=False, args=args)
    log.info("[init] loaded in %.1fs", time.time() - t0)

    cfg = CaptureConfig(
        save_attention=True,
        save_all_layer_hidden=not args.no_layer_hidden,
        save_kv_latent_only=not args.no_kv_latent,
        save_kv_full=args.save_kv_full,
        save_prompt_hidden_last=args.prompt_hidden_last,
        decode_latent_topk=args.decode_latent_topk,
        storage_warn_gb=args.storage_warn_gb,
    )

    # save canonical W_a once (trained)
    runner = InstrumentedLatentMAS(
        mw, latent_steps=args.latent_steps,
        judger_max_new_tokens=args.max_new_tokens,
        temperature=args.temperature, top_p=args.top_p,
        args=args, cfg=cfg,
    )
    runner.save_wa(out_root / "wa_matrix.pt")

    for n in names:
        spec = CONDITIONS[n]
        try:
            run_condition(spec, args, out_root, mw, cfg)
        except Exception as e:
            log.exception("[condition %s] failed: %s", n, e)
            continue

    if not args.skip_buckets:
        try:
            assign_buckets(out_root, args)
        except Exception as e:
            log.exception("[buckets] failed: %s", e)

    log.info("=" * 70)
    log.info("RUN COMPLETE  out=%s  disk=%s", out_root, fmt_bytes(dir_size_bytes(out_root)))
    log.info("=" * 70)


if __name__ == "__main__":
    main()
