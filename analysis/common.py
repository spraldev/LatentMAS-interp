"""Shared loaders, paths, bucket lookup, split lookup.

Data layout (matches final_run.py output):
  <root>/wa_matrix.pt
  <root>/run_metadata.json
  <root>/buckets/<task>.json
  <root>/<condition>/<task>/example_XXXX/{metadata.json, latent_thoughts.pt,
                                          latent_per_layer.pt, kv_latent.pt,
                                          prompt_hidden.pt, text_outputs.json,
                                          logitlens.json, patching.json}

A single "row" in our analysis is identified by (condition, task, example_id).
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch

log = logging.getLogger("analysis")

TASKS = ["gsm8k", "arc_challenge", "mbppplus"]

CORE_CONDITIONS = [
    "latent_mas",
    "single_agent_latent_sampled",
    "single_agent_latent_greedy",
    "self_consistency",
    "best_of_n",
    "cot_matched",
    "latent_mas_random_wa_spectrum",
    "kv_blocked",
    "no_transfer",
    "text_mas",
    "kv_shuffled",
    "latent_mas_random_wa_orth",
    "latent_mas_zero_wa",
    "exp_m_identity_wa",
    "topk_gated",
    "random_gated",
    "confidence_gated",
]


# ============================================================
# Path helpers
# ============================================================

def cond_root(root: Path, condition: str) -> Path:
    return root / condition


def task_root(root: Path, condition: str, task: str) -> Path:
    return root / condition / task


def example_dir(root: Path, condition: str, task: str, idx: int) -> Path:
    return root / condition / task / f"example_{idx:04d}"


def existing_example_ids(root: Path, condition: str, task: str) -> List[int]:
    d = task_root(root, condition, task)
    if not d.exists():
        return []
    out: List[int] = []
    for p in d.iterdir():
        if p.is_dir() and p.name.startswith("example_"):
            try:
                out.append(int(p.name.split("_")[1]))
            except Exception:
                continue
    return sorted(out)


# ============================================================
# Splits (locked by final_run.py into data/splits/)
# ============================================================

@lru_cache(maxsize=None)
def load_split(task: str, split: str) -> List[int]:
    """Load locked example-id split. Returns [] if file is missing."""
    p = Path("data/splits") / f"{task}_{split}.json"
    if not p.exists():
        log.warning("[splits] %s missing — returning empty list", p)
        return []
    return json.loads(p.read_text())


def split_filter(task: str, split: str) -> set:
    return set(load_split(task, split))


# ============================================================
# Buckets
# ============================================================

@lru_cache(maxsize=None)
def load_buckets(root_str: str, task: str) -> Dict[int, int]:
    """Return {example_id: bucket} for the given task. {} if missing."""
    p = Path(root_str) / "buckets" / f"{task}.json"
    if not p.exists():
        log.warning("[buckets] %s missing", p)
        return {}
    rows = json.loads(p.read_text())
    return {int(r["example_id"]): int(r["bucket"]) for r in rows}


# ============================================================
# Per-example loaders
# ============================================================

def load_metadata(ex_dir: Path) -> Optional[Dict]:
    p = ex_dir / "metadata.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def load_patching(ex_dir: Path) -> Optional[Dict]:
    p = ex_dir / "patching.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def load_latent_thoughts(ex_dir: Path) -> Optional[Dict]:
    """Returns {'agents': [...], 'pre_aligned': [A,M,D], 'post_aligned': [A,M,D]}."""
    p = ex_dir / "latent_thoughts.pt"
    if not p.exists():
        return None
    try:
        d = torch.load(p, map_location="cpu", weights_only=False)
        d["pre_aligned"] = d["pre_aligned"].to(torch.float32)
        d["post_aligned"] = d["post_aligned"].to(torch.float32)
        return d
    except Exception as e:
        log.warning("[load_latent_thoughts] %s: %s", p, e)
        return None


def load_latent_per_layer(ex_dir: Path) -> Optional[Dict]:
    """Returns {'agents': [...], 'hidden_per_layer': [A,M,L+1,D]}.
    May be missing if final_run was launched with --no_layer_hidden."""
    p = ex_dir / "latent_per_layer.pt"
    if not p.exists():
        return None
    try:
        d = torch.load(p, map_location="cpu", weights_only=False)
        d["hidden_per_layer"] = d["hidden_per_layer"].to(torch.float32)
        return d
    except Exception:
        return None


def load_logitlens(ex_dir: Path) -> Optional[List]:
    p = ex_dir / "logitlens.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def load_text_outputs(ex_dir: Path) -> Optional[Dict]:
    p = ex_dir / "text_outputs.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def load_kv_latent(ex_dir: Path) -> Optional[Dict]:
    p = ex_dir / "kv_latent.pt"
    if not p.exists():
        return None
    try:
        return torch.load(p, map_location="cpu", weights_only=False)
    except Exception:
        return None


def load_wa(root: Path) -> Optional[Dict]:
    p = root / "wa_matrix.pt"
    if not p.exists():
        return None
    try:
        return torch.load(p, map_location="cpu", weights_only=False)
    except Exception:
        return None


# ============================================================
# Iteration helpers
# ============================================================

@dataclass
class Example:
    condition: str
    task: str
    idx: int
    dir: Path
    meta: Dict


def iter_examples(
    root: Path,
    condition: str,
    task: str,
    *,
    split: Optional[str] = None,
    bucket: Optional[int] = None,
    require_correct: Optional[bool] = None,
    limit: Optional[int] = None,
) -> Iterator[Example]:
    split_ids = split_filter(task, split) if split else None
    buckets = load_buckets(str(root), task) if bucket is not None else {}
    n = 0
    for idx in existing_example_ids(root, condition, task):
        if split_ids is not None and idx not in split_ids:
            continue
        if bucket is not None and buckets.get(idx) != bucket:
            continue
        ex_dir = example_dir(root, condition, task, idx)
        meta = load_metadata(ex_dir)
        if meta is None:
            continue
        if require_correct is not None and bool(meta.get("correct")) != require_correct:
            continue
        yield Example(condition, task, idx, ex_dir, meta)
        n += 1
        if limit and n >= limit:
            return


# ============================================================
# Bulk activation collection
# ============================================================

def stack_post_aligned(
    root: Path, condition: str, task: str,
    *, split: Optional[str] = None, bucket: Optional[int] = None,
    limit: Optional[int] = None,
) -> Tuple[np.ndarray, List[Dict]]:
    """Return (X [N, A, M, D], info) where info[i] has example metadata."""
    Xs: List[np.ndarray] = []
    info: List[Dict] = []
    for ex in iter_examples(root, condition, task, split=split, bucket=bucket, limit=limit):
        lt = load_latent_thoughts(ex.dir)
        if lt is None:
            continue
        post = lt["post_aligned"].numpy()  # [A, M, D]
        Xs.append(post)
        info.append({
            "example_id": ex.idx,
            "task": task,
            "condition": condition,
            "correct": bool(ex.meta.get("correct")),
            "agents": lt.get("agents", []),
            "gold": ex.meta.get("gold"),
            "prediction": ex.meta.get("prediction"),
        })
    if not Xs:
        return np.zeros((0,)), []
    return np.stack(Xs, axis=0), info


def stack_pre_aligned(
    root: Path, condition: str, task: str,
    *, split: Optional[str] = None, bucket: Optional[int] = None,
    limit: Optional[int] = None,
) -> Tuple[np.ndarray, List[Dict]]:
    Xs: List[np.ndarray] = []
    info: List[Dict] = []
    for ex in iter_examples(root, condition, task, split=split, bucket=bucket, limit=limit):
        lt = load_latent_thoughts(ex.dir)
        if lt is None:
            continue
        Xs.append(lt["pre_aligned"].numpy())
        info.append({
            "example_id": ex.idx, "task": task, "condition": condition,
            "correct": bool(ex.meta.get("correct")),
            "agents": lt.get("agents", []),
        })
    if not Xs:
        return np.zeros((0,)), []
    return np.stack(Xs, axis=0), info


# ============================================================
# Output dirs
# ============================================================

def results_dir(root: Path, exp_name: str) -> Path:
    d = root / "results" / exp_name
    d.mkdir(parents=True, exist_ok=True)
    return d


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
        force=True,
    )
