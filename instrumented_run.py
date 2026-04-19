"""
Instrumented run file for LatentMAS interpretability experiments.

Runs LatentMAS once on a benchmark and saves all internal state needed for the
29 offline experiments described in experimental-design.txt:

For each example, per agent, per latent step, captures:
  - W_a alignment matrix (saved once at dataset root)
  - Pre-W_a hidden states h_t (last layer)              [Exp 1, 8, 26]
  - Post-W_a aligned latent thoughts e_{t+1} (injected) [Exp 1, 2, 3, 6, 8, 9, 10, 15-26]
  - All-layer hidden states at latent positions          [Exp 5, 7, 11, 19, 20]
  - Last-layer hidden states at prompt positions         [Exp 3, 4, 9, 24]
  - Compact KV cache (latent positions only) per agent  [Exp 3, 6, 11, 15, 24, 25]
  - Cross-agent attention patterns (optional, large)     [Exp 11]
  - Decoded text per agent + latent-decoded top-k tokens [Exp 4, 8]
  - Per-example metadata (correctness, gold, pred)       [all]

Designed for Kaggle T4 (16GB VRAM, 20GB dataset limit).

Usage:
  # test mode: 5 examples per benchmark
  python instrumented_run.py --output_dir ./activations --test

  # full primary run
  python instrumented_run.py --output_dir /kaggle/working/activations \
      --model_name Qwen/Qwen3-4B --latent_steps 4 --max_samples 500 \
      --tasks gsm8k arc_challenge mbppplus

  # vary-m sweep for Exp 29
  for m in 1 4 16 64; do
    python instrumented_run.py --output_dir ./activations_m$m --latent_steps $m ...
  done

The run is resumable: existing example_XXXX/ folders are skipped.
"""

import argparse
import json
import logging
import os
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from tqdm import tqdm

# --- project imports ---
from data import (
    load_arc_challenge,
    load_gsm8k,
    load_mbppplus,
)
from methods import default_agents
from models import ModelWrapper, _past_length
from prompts import (
    build_agent_message_hierarchical_latent_mas,
    build_agent_message_sequential_latent_mas,
)
from utils import (
    auto_device,
    extract_gsm8k_answer,
    extract_markdown_python_block,
    normalize_answer,
    run_with_timeout,
    set_seed,
)

try:
    from transformers.cache_utils import Cache
except ImportError:
    Cache = None


log = logging.getLogger("instrumented_run")


def setup_logging(log_file: Optional["Path"] = None, level: str = "INFO") -> None:
    fmt = "%(asctime)s [%(levelname)s] %(message)s"
    datefmt = "%H:%M:%S"
    handlers: List[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file is not None:
        handlers.append(logging.FileHandler(log_file, mode="a"))
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=fmt,
        datefmt=datefmt,
        handlers=handlers,
        force=True,
    )


# ============================================================
# Storage helpers
# ============================================================

def dir_size_bytes(path: Path) -> int:
    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            try:
                total += p.stat().st_size
            except OSError:
                pass
    return total


def fmt_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.2f}{unit}"
        n /= 1024
    return f"{n:.2f}PB"


def to_fp16_cpu(t: torch.Tensor) -> torch.Tensor:
    return t.detach().to(dtype=torch.float16, device="cpu").contiguous()


def kv_to_legacy(past):
    """Convert HF Cache object (or legacy) to a list of (k, v) per layer."""
    if past is None:
        return None
    # transformers >= 4.40: DynamicCache exposes key_cache/value_cache directly
    if hasattr(past, "key_cache") and hasattr(past, "value_cache"):
        return list(zip(past.key_cache, past.value_cache))
    # older Cache API
    if Cache is not None and isinstance(past, Cache) and hasattr(past, "to_legacy_cache"):
        return past.to_legacy_cache()
    return past


def slice_kv_positions(past, start: int, end: int) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """Return per-layer (k, v) sliced over the sequence dim [start:end]."""
    legacy = kv_to_legacy(past)
    out = []
    for layer in legacy:
        k, v = layer[0], layer[1]
        # k, v: [B, num_kv_heads, seq, head_dim]
        out.append((to_fp16_cpu(k[..., start:end, :]), to_fp16_cpu(v[..., start:end, :])))
    return out


# ============================================================
# Dataset loading
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


# ============================================================
# Answer scoring
# ============================================================

def score_prediction(task: str, item: Dict, final_text: str) -> Tuple[Optional[str], str, bool]:
    if task == "mbppplus":
        pred = extract_markdown_python_block(final_text)
        gold = item.get("gold", "")
        if pred is None:
            return None, gold, False
        ok, _ = run_with_timeout(pred + "\n" + gold, timeout=10)
        return pred, gold, bool(ok)
    pred = normalize_answer(extract_gsm8k_answer(final_text))
    gold = item.get("gold", "")
    return pred, gold, bool(pred and gold and pred == gold)


# ============================================================
# Instrumented runner
# ============================================================

@dataclass
class CaptureConfig:
    save_attention: bool = False           # last-layer attention from latent queries to all keys
    save_all_layer_hidden: bool = True     # all-layer hidden at latent positions
    save_kv_latent_only: bool = True       # per-layer KV at latent positions only (not full prompt KV)
    save_kv_full: bool = False             # per-layer KV across whole context (very large)
    save_prompt_hidden_last: int = 64      # last N positions of prompt (last layer); 0 to disable
    decode_latent_topk: int = 5            # top-k tokens decoded from latent via lm_head
    storage_warn_gb: float = 18.0          # stop run if dataset dir exceeds this


class InstrumentedLatentMAS:
    """Manual re-implementation of the LatentMAS forward loop with full activation capture.

    Reuses ModelWrapper for the model + W_a alignment, but bypasses
    methods.latent_mas.LatentMASMethod so we can hook every intermediate state.
    """

    def __init__(
        self,
        model_wrapper: ModelWrapper,
        *,
        latent_steps: int,
        judger_max_new_tokens: int,
        temperature: float,
        top_p: float,
        args: argparse.Namespace,
        cfg: CaptureConfig,
    ):
        self.mw = model_wrapper
        self.model = model_wrapper.model        # HF transformers model
        self.tokenizer = model_wrapper.tokenizer
        self.device = model_wrapper.device
        self.latent_steps = latent_steps
        self.judger_max_new_tokens = judger_max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.args = args
        self.cfg = cfg
        self.agents = default_agents()        # planner, critic, refiner, judger
        self.lm_head = self.model.get_output_embeddings()

    # --------------------------------------------------------
    # W_a save (once)
    # --------------------------------------------------------
    def save_wa(self, out_path: Path) -> None:
        if out_path.exists():
            return
        wa, target_norm = self.mw._ensure_latent_realign_matrix(self.model, self.device, self.args)
        torch.save(
            {
                "W_a": wa.detach().cpu(),
                "target_norm": target_norm.detach().cpu(),
                "d_h": int(wa.shape[0]),
                "model_name": self.args.model_name,
                "note": (
                    "W_a is the latent realignment matrix from ModelWrapper. "
                    "If --latent_space_realign was OFF during the run, W_a is the identity "
                    "and only target_norm scaling is applied; in that case the 'effective' "
                    "transform is hidden -> hidden * (target_norm / ||hidden||)."
                ),
            },
            out_path,
        )

    # --------------------------------------------------------
    # Build prompt for one agent
    # --------------------------------------------------------
    def _build_prompt(self, agent_role: str, question: str) -> Tuple[str, torch.Tensor, torch.Tensor]:
        if self.args.prompt == "sequential":
            messages = build_agent_message_sequential_latent_mas(
                role=agent_role, question=question, context="", method="latent_mas", args=self.args
            )
        else:
            messages = build_agent_message_hierarchical_latent_mas(
                role=agent_role, question=question, context="", method="latent_mas", args=self.args
            )
        prompts, input_ids, attention_mask, _ = self.mw.prepare_chat_batch(
            [messages], add_generation_prompt=True
        )
        prompt_text = prompts[0]
        if self.args.think:
            prompt_text = prompt_text + "<think>"
            enc = self.tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)
            input_ids = enc["input_ids"].to(self.device)
            attention_mask = enc["attention_mask"].to(self.device)
        return prompt_text, input_ids, attention_mask

    # --------------------------------------------------------
    # Decode latent vector to top-k vocab tokens via lm_head
    # --------------------------------------------------------
    @torch.no_grad()
    def _decode_latent_topk(self, hidden_last: torch.Tensor, k: int) -> List[Tuple[str, float]]:
        # hidden_last: [D]
        if self.lm_head is None or k <= 0:
            return []
        logits = self.lm_head(hidden_last.to(self.lm_head.weight.dtype))
        probs = torch.softmax(logits.float(), dim=-1)
        topp, topi = probs.topk(k)
        return [
            (self.tokenizer.decode([int(i)], skip_special_tokens=False), float(p))
            for i, p in zip(topi.tolist(), topp.tolist())
        ]

    # --------------------------------------------------------
    # Run one example end-to-end with capture
    # --------------------------------------------------------
    @torch.no_grad()
    def run_and_capture(self, item: Dict, save_dir: Path) -> Dict:
        save_dir.mkdir(parents=True, exist_ok=True)
        question = item["question"]
        log.info("[example] dir=%s  q_preview=%r", save_dir.name, question[:80])
        t_start = time.time()

        # captured tensors per latent agent
        cap_pre_aligned: List[torch.Tensor] = []      # [m, D] per agent (h_t before W_a)
        cap_post_aligned: List[torch.Tensor] = []     # [m, D] per agent (e_{t+1} injected)
        cap_layer_hidden: List[torch.Tensor] = []     # [m, num_layers+1, D] per agent
        cap_prompt_hidden: Dict[str, torch.Tensor] = {}  # last N tokens of prompt, last-layer
        cap_attn: Dict[str, torch.Tensor] = {}        # [m, num_layers, num_heads, kv_len_at_step]
        cap_kv_latent: Dict[str, List[Tuple[torch.Tensor, torch.Tensor]]] = {}
        cap_text: Dict[str, Dict] = {}

        latent_agent_names: List[str] = []
        past_kv = None

        for agent in self.agents:
            t_agent = time.time()
            prompt_text, input_ids, attention_mask = self._build_prompt(agent.role, question)
            log.info("  [agent=%s] prompt_tokens=%d past_kv=%d",
                     agent.role, int(input_ids.shape[-1]), _past_length(past_kv))

            if agent.role != "judger":
                latent_agent_names.append(agent.role)

                # mark cache position before this agent's prompt is consumed
                pre_prompt_len = _past_length(past_kv)

                # build attention mask spanning past + new
                full_mask = attention_mask
                if past_kv is not None and pre_prompt_len > 0:
                    past_mask = torch.ones(
                        (attention_mask.shape[0], pre_prompt_len),
                        dtype=attention_mask.dtype,
                        device=attention_mask.device,
                    )
                    full_mask = torch.cat([past_mask, attention_mask], dim=-1)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=full_mask,
                    past_key_values=past_kv,
                    use_cache=True,
                    output_hidden_states=True,
                    output_attentions=self.cfg.save_attention,
                    return_dict=True,
                )
                past_kv = outputs.past_key_values
                last_hidden = outputs.hidden_states[-1][:, -1, :]  # [1, D]

                # save last-layer prompt hidden (for probing/cross-agent mind reading)
                if self.cfg.save_prompt_hidden_last > 0:
                    keep = min(self.cfg.save_prompt_hidden_last, outputs.hidden_states[-1].shape[1])
                    cap_prompt_hidden[agent.role] = to_fp16_cpu(
                        outputs.hidden_states[-1][0, -keep:, :]
                    )

                # mark where this agent's latent thoughts will start in the cache
                latent_start_in_cache = _past_length(past_kv)

                step_pre: List[torch.Tensor] = []
                step_post: List[torch.Tensor] = []
                step_layer: List[torch.Tensor] = []
                step_attn: List[torch.Tensor] = []
                step_decode: List[List[Tuple[str, float]]] = []

                for step in range(self.latent_steps):
                    # h_t before W_a
                    step_pre.append(to_fp16_cpu(last_hidden[0]))
                    step_decode.append(self._decode_latent_topk(last_hidden[0], self.cfg.decode_latent_topk))

                    # apply W_a (uses HF_model if present, else self.model)
                    source_model = getattr(self.mw, "HF_model", self.model)
                    latent_vec = self.mw._apply_latent_realignment(last_hidden, source_model)  # [1, D]
                    step_post.append(to_fp16_cpu(latent_vec[0]))

                    latent_embed = latent_vec.unsqueeze(1)  # [1, 1, D]
                    past_len = _past_length(past_kv)
                    latent_mask = torch.ones(
                        (latent_embed.shape[0], past_len + 1),
                        dtype=torch.long,
                        device=self.device,
                    )
                    outputs = self.model(
                        inputs_embeds=latent_embed,
                        attention_mask=latent_mask,
                        past_key_values=past_kv,
                        use_cache=True,
                        output_hidden_states=True,
                        output_attentions=self.cfg.save_attention,
                        return_dict=True,
                    )
                    past_kv = outputs.past_key_values
                    last_hidden = outputs.hidden_states[-1][:, -1, :]

                    # all-layer hidden at this latent step: [num_layers+1, D]
                    if self.cfg.save_all_layer_hidden:
                        per_layer = torch.stack(
                            [h[0, -1, :] for h in outputs.hidden_states], dim=0
                        )
                        step_layer.append(to_fp16_cpu(per_layer))

                    if self.cfg.save_attention:
                        # [num_layers, num_heads, kv_len_now] -- queries=this latent token
                        attn_stack = torch.stack(
                            [a[0, :, -1, :] for a in outputs.attentions], dim=0
                        )
                        step_attn.append(to_fp16_cpu(attn_stack))

                # stash per-agent
                if step_pre:
                    cap_pre_aligned.append(torch.stack(step_pre, dim=0))      # [m, D]
                    cap_post_aligned.append(torch.stack(step_post, dim=0))    # [m, D]
                if self.cfg.save_all_layer_hidden and step_layer:
                    cap_layer_hidden.append(torch.stack(step_layer, dim=0))   # [m, L+1, D]
                if self.cfg.save_attention and step_attn:
                    # ragged across steps (kv_len grows by 1 each step) — store as list dict
                    cap_attn[agent.role] = [t for t in step_attn]
                if self.cfg.save_kv_latent_only:
                    latent_end = _past_length(past_kv)
                    cap_kv_latent[agent.role] = slice_kv_positions(
                        past_kv, latent_start_in_cache, latent_end
                    )

                log.info("    completed %d latent steps in %.2fs (last_hidden_norm=%.3f)",
                         self.latent_steps, time.time() - t_agent,
                         float(last_hidden.float().norm()))

                cap_text[agent.role] = {
                    "input": prompt_text,
                    "output": "",
                    "latent_steps": self.latent_steps,
                    "latent_decoded_topk": step_decode,
                }

            else:
                # judger: text generation conditioned on accumulated latent KV cache
                pre_prompt_len = _past_length(past_kv)
                full_mask = attention_mask
                if past_kv is not None and pre_prompt_len > 0:
                    past_mask = torch.ones(
                        (attention_mask.shape[0], pre_prompt_len),
                        dtype=attention_mask.dtype,
                        device=attention_mask.device,
                    )
                    full_mask = torch.cat([past_mask, attention_mask], dim=-1)

                cache_position = torch.arange(
                    pre_prompt_len,
                    pre_prompt_len + input_ids.shape[-1],
                    dtype=torch.long,
                    device=self.device,
                )
                gen_out = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=full_mask,
                    max_new_tokens=self.judger_max_new_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    return_dict_in_generate=True,
                    past_key_values=past_kv,
                    cache_position=cache_position,
                )
                gen_ids = gen_out.sequences[0, input_ids.shape[-1]:]
                final_text = self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
                log.info("    judger generated %d tokens in %.2fs",
                         int(gen_ids.shape[-1]), time.time() - t_agent)
                log.info("    judger output preview: %r", final_text[:160])
                cap_text[agent.role] = {
                    "input": prompt_text,
                    "output": final_text,
                }

        # ---- score ----
        final_text = cap_text["judger"]["output"]
        pred, gold, ok = score_prediction(self.args.task_current, item, final_text)
        log.info("  [score] pred=%r gold=%r correct=%s", str(pred)[:40], str(gold)[:40], ok)

        # ---- write to disk ----
        if cap_pre_aligned:
            torch.save(
                {
                    "agents": latent_agent_names,
                    "pre_aligned": torch.stack(cap_pre_aligned, dim=0),   # [A, m, D] h_t pre-W_a
                    "post_aligned": torch.stack(cap_post_aligned, dim=0), # [A, m, D] e_{t+1} injected
                },
                save_dir / "latent_thoughts.pt",
            )
        if self.cfg.save_all_layer_hidden and cap_layer_hidden:
            torch.save(
                {
                    "agents": latent_agent_names,
                    "hidden_per_layer": torch.stack(cap_layer_hidden, dim=0),  # [A, m, L+1, D]
                },
                save_dir / "latent_per_layer.pt",
            )
        if cap_prompt_hidden:
            torch.save(cap_prompt_hidden, save_dir / "prompt_hidden.pt")
        if self.cfg.save_kv_latent_only and cap_kv_latent:
            torch.save(cap_kv_latent, save_dir / "kv_latent.pt")
        if self.cfg.save_attention and cap_attn:
            torch.save(cap_attn, save_dir / "attention_latent.pt")
        with (save_dir / "text_outputs.json").open("w") as f:
            json.dump(cap_text, f, ensure_ascii=False, indent=2)

        meta = {
            "question": question,
            "gold": gold,
            "prediction": pred,
            "raw_prediction": final_text,
            "correct": bool(ok),
            "num_latent_agents": len(latent_agent_names),
            "agents": latent_agent_names,
            "latent_steps": self.latent_steps,
            "model_name": self.args.model_name,
            "task": self.args.task_current,
        }
        with (save_dir / "metadata.json").open("w") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        # free
        del past_kv, cap_pre_aligned, cap_post_aligned, cap_layer_hidden
        del cap_prompt_hidden, cap_attn, cap_kv_latent
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        ex_size = dir_size_bytes(save_dir)
        log.info("  [done] %s in %.2fs  size=%s", save_dir.name,
                 time.time() - t_start, fmt_bytes(ex_size))
        return meta


# ============================================================
# Verification of saved tensors
# ============================================================

def verify_example(save_dir: Path) -> List[str]:
    issues: List[str] = []
    needed = ["latent_thoughts.pt", "metadata.json", "text_outputs.json"]
    for n in needed:
        if not (save_dir / n).exists():
            issues.append(f"missing {n}")
    try:
        lt = torch.load(save_dir / "latent_thoughts.pt", map_location="cpu", weights_only=False)
        for k in ("pre_aligned", "post_aligned"):
            t = lt[k]
            if torch.isnan(t).any():
                issues.append(f"NaN in {k}")
            if (t == 0).all():
                issues.append(f"all zeros in {k}")
    except Exception as e:
        issues.append(f"load latent_thoughts failed: {e}")
    return issues


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Root dir for saved activations (e.g. /kaggle/working/activations)")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-4B")
    parser.add_argument("--tasks", nargs="+",
                        default=["gsm8k", "arc_challenge", "mbppplus"],
                        choices=list(DATASET_LOADERS.keys()))
    parser.add_argument("--max_samples", type=int, default=500,
                        help="Per-task example count")
    parser.add_argument("--latent_steps", type=int, default=4)
    parser.add_argument("--prompt", choices=["sequential", "hierarchical"], default="sequential")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--max_new_tokens", type=int, default=2048,
                        help="Max new tokens for the judger agent")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--think", action="store_true")
    parser.add_argument("--latent_space_realign", action="store_true",
                        help="Use the ridge-regression W_a (recommended for interp). "
                             "If off, W_a degenerates to identity + norm scaling.")

    # capture flags
    parser.add_argument("--no_layer_hidden", action="store_true",
                        help="Skip saving all-layer hidden at latent positions")
    parser.add_argument("--save_attention", action="store_true",
                        help="Save attention from latent queries to all keys (large)")
    parser.add_argument("--save_kv_full", action="store_true",
                        help="Save full per-layer KV cache (very large)")
    parser.add_argument("--no_kv_latent", action="store_true",
                        help="Skip saving per-layer KV at latent positions")
    parser.add_argument("--prompt_hidden_last", type=int, default=64,
                        help="Save last-layer hidden for last N prompt tokens; 0 to disable")
    parser.add_argument("--decode_latent_topk", type=int, default=5)

    # run control
    parser.add_argument("--test", action="store_true",
                        help="Run only 5 examples per task to verify pipeline")
    parser.add_argument("--storage_warn_gb", type=float, default=18.0)
    parser.add_argument("--check_storage_every", type=int, default=50)
    parser.add_argument("--verify_every", type=int, default=10)

    # vLLM args (kept for ModelWrapper compatibility; instrumentation uses HF path only)
    parser.add_argument("--use_vllm", action="store_true")
    parser.add_argument("--use_second_HF_model", action="store_true")
    parser.add_argument("--enable_prefix_caching", action="store_true")
    parser.add_argument("--device2", type=str, default="cuda:1")
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--method", type=str, default="latent_mas")  # for ModelWrapper switch
    parser.add_argument("--generate_bs", type=int, default=1)
    parser.add_argument("--latent_only", action="store_true")
    parser.add_argument("--sequential_info_only", action="store_true")
    parser.add_argument("--text_mas_context_length", type=int, default=-1)
    parser.add_argument("--log_level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--log_file", type=str, default=None,
                        help="Path to log file (default: <output_dir>/run.log)")

    args = parser.parse_args()

    if args.use_vllm:
        print("[warn] --use_vllm is incompatible with full-state capture; forcing HF path.")
        args.use_vllm = False

    if args.test:
        args.max_samples = 5

    set_seed(args.seed)
    device = auto_device(args.device)
    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    log_file = Path(args.log_file) if args.log_file else (out_root / "run.log")
    setup_logging(log_file, args.log_level)
    log.info("=" * 70)
    log.info("instrumented_run starting")
    log.info("  output_dir=%s  log_file=%s", out_root, log_file)
    log.info("  model=%s  device=%s  tasks=%s", args.model_name, device, args.tasks)
    log.info("  latent_steps=%d  max_samples_per_task=%d  test=%s",
             args.latent_steps, args.max_samples, args.test)
    log.info("=" * 70)

    # global metadata
    run_meta = {
        "model_name": args.model_name,
        "tasks": args.tasks,
        "max_samples_per_task": args.max_samples,
        "latent_steps": args.latent_steps,
        "prompt": args.prompt,
        "seed": args.seed,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_new_tokens": args.max_new_tokens,
        "latent_space_realign": args.latent_space_realign,
        "think": args.think,
        "save_attention": args.save_attention,
        "save_all_layer_hidden": not args.no_layer_hidden,
        "save_kv_latent_only": not args.no_kv_latent,
        "save_kv_full": args.save_kv_full,
        "prompt_hidden_last": args.prompt_hidden_last,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with (out_root / "metadata.json").open("w") as f:
        json.dump(run_meta, f, ensure_ascii=False, indent=2)

    log.info("[init] loading model %s on %s ...", args.model_name, device)
    t0 = time.time()
    model_wrapper = ModelWrapper(args.model_name, device, use_vllm=False, args=args)
    log.info("[init] model loaded in %.1fs", time.time() - t0)

    cfg = CaptureConfig(
        save_attention=args.save_attention,
        save_all_layer_hidden=not args.no_layer_hidden,
        save_kv_latent_only=not args.no_kv_latent,
        save_kv_full=args.save_kv_full,
        save_prompt_hidden_last=args.prompt_hidden_last,
        decode_latent_topk=args.decode_latent_topk,
        storage_warn_gb=args.storage_warn_gb,
    )

    runner = InstrumentedLatentMAS(
        model_wrapper,
        latent_steps=args.latent_steps,
        judger_max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        args=args,
        cfg=cfg,
    )

    # save W_a once
    runner.save_wa(out_root / "wa_matrix.pt")
    log.info("[init] W_a saved to %s", out_root / "wa_matrix.pt")

    total_processed = 0
    overall_correct = 0

    for task in args.tasks:
        args.task_current = task
        task_dir = out_root / task
        task_dir.mkdir(parents=True, exist_ok=True)
        items = load_task(task, args.split, args.max_samples)
        log.info("[task=%s] %d examples", task, len(items))

        # per-task metadata.csv (rebuild from existing folders + new)
        meta_rows: List[Dict] = []
        meta_csv = task_dir / "metadata.csv"

        pbar = tqdm(items, desc=task)
        for idx, item in enumerate(pbar):
            ex_dir = task_dir / f"example_{idx:04d}"
            done_marker = ex_dir / "metadata.json"
            if done_marker.exists():
                # resume: load existing meta into csv buffer
                try:
                    with done_marker.open() as f:
                        meta_rows.append({"example_id": idx, **json.load(f)})
                except Exception:
                    pass
                continue

            try:
                meta = runner.run_and_capture(item, ex_dir)
            except Exception as e:
                log.exception("[error] example %d (%s) failed: %s", idx, task, e)
                # clean partial dir to allow retry
                if ex_dir.exists():
                    shutil.rmtree(ex_dir, ignore_errors=True)
                continue

            meta_rows.append({"example_id": idx, **meta})
            total_processed += 1
            overall_correct += int(meta["correct"])

            if (idx + 1) % args.verify_every == 0:
                issues = verify_example(ex_dir)
                if issues:
                    log.warning("[verify] example %d: %s", idx, issues)
                else:
                    log.info("[verify] example %d ok", idx)

            if (idx + 1) % args.check_storage_every == 0:
                used = dir_size_bytes(out_root)
                used_gb = used / (1024 ** 3)
                pbar.set_postfix(disk=fmt_bytes(used), acc=f"{overall_correct}/{total_processed}")
                log.info("[storage] %s used  running_acc=%d/%d",
                         fmt_bytes(used), overall_correct, total_processed)
                if used_gb > cfg.storage_warn_gb:
                    log.error("[stop] storage %.2fGB exceeds limit %.2fGB",
                              used_gb, cfg.storage_warn_gb)
                    sys.exit(2)

        # write per-task metadata.csv
        if meta_rows:
            import csv
            keys = sorted({k for row in meta_rows for k in row.keys()})
            with meta_csv.open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=keys)
                w.writeheader()
                for row in meta_rows:
                    w.writerow({k: row.get(k, "") for k in keys})

    final_used = dir_size_bytes(out_root)
    summary = {
        "processed_this_run": total_processed,
        "correct_this_run": overall_correct,
        "disk_usage": fmt_bytes(final_used),
        "output_dir": str(out_root),
    }
    log.info("=" * 70)
    log.info("RUN COMPLETE: %s", json.dumps(summary))
    log.info("=" * 70)


if __name__ == "__main__":
    main()
