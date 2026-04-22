"""
run_experiments.py

Runs the 91 offline LatentMAS interpretability experiments (IDs drawn from
all-experiments.txt) over the saved activations in ./activations/.

For each experiment we compute a real statistical / geometric metric using
only what is stored on disk. Each experiment emits results/exp_NNN.json with
the shape:

    {
      "id": 34,
      "name": "Cross-Task Latent Clustering",
      "question": "...",
      "metrics": { ... numeric results ... },
      "finding": "one-line interpretation",
      "score": 0.0-1.0,       # heuristic strength of the effect
      "status": "ok" | "skipped" | "error",
      "notes": "..."
    }

A final results/summary.json ranks all experiments by score so you can ask
"which experiments had good results?".

Usage:
    python run_experiments.py --activations ./activations --out results \
        --tasks gsm8k arc_challenge mbppplus --max-examples 200

By default we subsample examples per task (for speed); pass --max-examples 0
to use everything available.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# lazy torch import (so summary can be read without torch)
# ---------------------------------------------------------------------------
try:
    import torch
except Exception as e:  # pragma: no cover
    print("torch is required:", e, file=sys.stderr)
    raise


# ---------------------------------------------------------------------------
# small numeric helpers
# ---------------------------------------------------------------------------
def _np(x) -> np.ndarray:
    if torch.is_tensor(x):
        # float32 keeps size reasonable; most ops re-cast to float64 as needed
        return x.detach().float().cpu().numpy().astype(np.float32)
    return np.asarray(x)


def _f64(x) -> np.ndarray:
    return np.asarray(x, dtype=np.float64)


def cosine(a: np.ndarray, b: np.ndarray, eps: float = 1e-9) -> float:
    a = a.ravel(); b = b.ravel()
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    if na < eps or nb < eps:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def cosine_matrix(A: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    n = np.linalg.norm(A, axis=1, keepdims=True)
    n = np.maximum(n, eps)
    An = A / n
    return An @ An.T


def pca(X: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (components [k,D], explained_variance [k], mean [D])."""
    mu = X.mean(axis=0)
    Xc = X - mu
    # SVD on Xc
    # economy SVD
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    k = min(k, Vt.shape[0])
    ev = (S[:k] ** 2) / max(1, X.shape[0] - 1)
    return Vt[:k], ev, mu


def cka_linear(X: np.ndarray, Y: np.ndarray) -> float:
    """Linear centered-kernel-alignment between two [N,D] matrices."""
    X = X - X.mean(0); Y = Y - Y.mean(0)
    xy = np.linalg.norm(X.T @ Y, "fro") ** 2
    xx = np.linalg.norm(X.T @ X, "fro")
    yy = np.linalg.norm(Y.T @ Y, "fro")
    d = xx * yy
    if d < 1e-20:
        return 0.0
    return float(xy / d)


def participation_ratio(eigs: np.ndarray) -> float:
    s = eigs.sum()
    if s <= 0:
        return 0.0
    return float(s * s / (np.square(eigs).sum() + 1e-20))


def safe_entropy(p: np.ndarray, eps: float = 1e-12) -> float:
    p = p[p > eps]
    return float(-(p * np.log(p)).sum())


def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x); return 1 / (1 + z)
    z = math.exp(x); return z / (1 + z)


def auc_roc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Binary AUC via Mann-Whitney. labels in {0,1}."""
    pos = scores[labels == 1]; neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    # rank-based
    all_s = np.concatenate([pos, neg])
    order = np.argsort(all_s)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(all_s) + 1)
    r_pos = ranks[: len(pos)].sum()
    auc = (r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))
    return float(auc)


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, float); y = np.asarray(y, float)
    if x.std() < 1e-12 or y.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


# ---------------------------------------------------------------------------
# dataset loader
# ---------------------------------------------------------------------------
@dataclass
class ExampleRec:
    task: str
    idx: int
    path: Path
    meta: dict
    # filled on demand
    _pre: Optional[np.ndarray] = None   # [A, m, D]
    _post: Optional[np.ndarray] = None  # [A, m, D]
    _perlayer: Optional[np.ndarray] = None  # [A, m, L+1, D]
    _prompt: Optional[Dict[str, np.ndarray]] = None  # agent -> [64,D]
    _text: Optional[dict] = None

    def latents(self) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        if self._pre is None:
            d = torch.load(self.path / "latent_thoughts.pt",
                           map_location="cpu", weights_only=False)
            self._pre = _np(d["pre_aligned"])
            self._post = _np(d["post_aligned"])
            self._agents = list(d["agents"])
        return self._pre, self._post, self._agents

    def per_layer(self) -> np.ndarray:
        if self._perlayer is None:
            d = torch.load(self.path / "latent_per_layer.pt",
                           map_location="cpu", weights_only=False)
            self._perlayer = _np(d["hidden_per_layer"])
        return self._perlayer

    def prompt(self) -> Dict[str, np.ndarray]:
        if self._prompt is None:
            d = torch.load(self.path / "prompt_hidden.pt",
                           map_location="cpu", weights_only=False)
            self._prompt = {k: _np(v) for k, v in d.items()}
        return self._prompt

    def text(self) -> dict:
        if self._text is None:
            with open(self.path / "text_outputs.json") as f:
                self._text = json.load(f)
        return self._text

    def drop_heavy(self):
        self._perlayer = None
        self._prompt = None


@dataclass
class Dataset:
    root: Path
    tasks: List[str]
    wa: Dict[str, np.ndarray] = field(default_factory=dict)
    wa_meta: Dict[str, dict] = field(default_factory=dict)
    examples: Dict[str, List[ExampleRec]] = field(default_factory=dict)

    def load(self, max_examples: int = 0):
        # W_a (single, saved at root)
        wa_path = self.root / "wa_matrix.pt"
        if wa_path.exists():
            d = torch.load(wa_path, map_location="cpu", weights_only=False)
            W = _np(d["W_a"])
            meta = {k: (v if not torch.is_tensor(v) else _np(v))
                    for k, v in d.items() if k != "W_a"}
            for t in self.tasks:
                self.wa[t] = W
                self.wa_meta[t] = meta

        for t in self.tasks:
            tdir = self.root / t
            if not tdir.is_dir():
                continue
            examples = []
            sub = sorted([p for p in tdir.iterdir()
                          if p.is_dir() and p.name.startswith("example_")])
            if max_examples and max_examples > 0:
                sub = sub[:max_examples]
            for p in sub:
                mpath = p / "metadata.json"
                if not mpath.exists():
                    continue
                with open(mpath) as f:
                    meta = json.load(f)
                examples.append(ExampleRec(
                    task=t, idx=int(p.name.split("_")[1]),
                    path=p, meta=meta
                ))
            self.examples[t] = examples

    def all_examples(self) -> List[ExampleRec]:
        out = []
        for t in self.tasks:
            out.extend(self.examples.get(t, []))
        return out

    # ------------------------------------------------------------------
    # cached aggregates used by many experiments
    # ------------------------------------------------------------------
    _cache_light: Optional[dict] = None

    def light_aggregate(self) -> dict:
        """Load pre/post latent thoughts for *every* example, plus labels.

        Returns dict with:
          pre   : [N, A, m, D]
          post  : [N, A, m, D]
          tasks : [N] string
          correct : [N] bool
          example_ids : [N] "task/idx"
        """
        if self._cache_light is not None:
            return self._cache_light
        pre_list, post_list, tasks, correct, ids = [], [], [], [], []
        for t in self.tasks:
            for ex in self.examples.get(t, []):
                pre, post, agents = ex.latents()
                pre_list.append(pre); post_list.append(post)
                tasks.append(t)
                correct.append(bool(ex.meta.get("correct", False)))
                ids.append(f"{t}/{ex.idx}")
        # align shapes via stacking (assumes identical A, m, D across examples)
        pre = np.stack(pre_list) if pre_list else np.zeros((0, 0, 0, 0))
        post = np.stack(post_list) if post_list else np.zeros((0, 0, 0, 0))
        self._cache_light = dict(
            pre=pre, post=post,
            tasks=np.array(tasks),
            correct=np.array(correct),
            ids=np.array(ids),
        )
        return self._cache_light


# ---------------------------------------------------------------------------
# experiment registry
# ---------------------------------------------------------------------------
Registry: Dict[int, Tuple[str, str, Callable]] = {}


def experiment(_id: int, name: str, question: str):
    def deco(fn):
        Registry[_id] = (name, question, fn)
        return fn
    return deco


def finding(metrics: dict, score: float, msg: str, notes: str = "",
            status: str = "ok") -> dict:
    return dict(metrics=metrics, score=float(max(0.0, min(1.0, score))),
                finding=msg, notes=notes, status=status)


# ===========================================================================
# TRACK 1 — W_a / matrix analysis
# ===========================================================================
@experiment(1, "The Latent Editor (W_a SVD)",
            "Is W_a an editor: does it systematically amplify some directions "
            "and discard others?")
def exp_001(ds: Dataset) -> dict:
    W = ds.wa[ds.tasks[0]]
    # W maps z -> z' with z' = target_norm * normalize(W z). Analyse singular
    # values of W as a proxy for how strongly each direction is amplified /
    # suppressed.
    s = np.linalg.svd(W, compute_uv=False)
    s = s / (s.max() + 1e-20)
    # fraction of directions with s < 0.1 == strongly suppressed
    suppressed = float((s < 0.1).mean())
    amplified = float((s > 0.9).mean())
    # "editing" strength = std of normalised singular values
    edit = float(s.std())
    return finding(
        metrics=dict(n_singular=int(len(s)), suppressed_frac=suppressed,
                     amplified_frac=amplified, edit_strength=edit,
                     top1=float(s[0]), bottom1=float(s[-1]),
                     spread=float(s[0] / (s[-1] + 1e-12))),
        score=min(1.0, edit * 4),
        msg=f"W_a singular spectrum spread={s[0]/(s[-1]+1e-12):.1f}; "
            f"{suppressed*100:.1f}% of directions suppressed (<0.1 of max sv), "
            f"{amplified*100:.1f}% nearly preserved. Active 'editor'." if suppressed > 0.2
            else "W_a behaves mostly as a near-identity / passive translator."
    )


@experiment(30, "W_a Fixed-Point",
            "Does W_a have fixed-point directions v* with W_a v* ≈ v*?")
def exp_030(ds: Dataset) -> dict:
    W = ds.wa[ds.tasks[0]]
    eigvals, eigvecs = np.linalg.eig(W)
    real_eigs = np.real(eigvals)
    fp_frac = float(np.mean(np.abs(real_eigs - 1) < 0.05))
    near_one = float(np.mean(np.abs(np.abs(eigvals) - 1) < 0.05))
    return finding(
        metrics=dict(n_eigs=int(len(eigvals)),
                     frac_real_near_1=fp_frac,
                     frac_mag_near_1=near_one,
                     max_real=float(np.max(real_eigs)),
                     median_abs=float(np.median(np.abs(eigvals)))),
        score=min(1.0, near_one * 3),
        msg=f"{near_one*100:.2f}% of W_a eigenvalues have |λ|≈1 "
            f"(candidate fixed-point directions)."
    )


@experiment(31, "W_a Effective Rank / Bottleneck",
            "Effective rank of W_a — does low rank predict better task perf?")
def exp_031(ds: Dataset) -> dict:
    W = ds.wa[ds.tasks[0]]
    s = np.linalg.svd(W, compute_uv=False)
    p = s / s.sum()
    eff_rank_exp = float(np.exp(safe_entropy(p)))
    pr = participation_ratio(s ** 2)
    # fraction of variance captured in top 1% of dims
    k = max(1, len(s) // 100)
    var_top = float((s[:k] ** 2).sum() / (s ** 2).sum())
    return finding(
        metrics=dict(d=int(W.shape[0]), effective_rank_entropy=eff_rank_exp,
                     participation_ratio=pr, top_1pct_var=var_top),
        score=min(1.0, 1.0 - eff_rank_exp / W.shape[0]),
        msg=f"Effective rank (entropy) {eff_rank_exp:.1f} / {W.shape[0]} — "
            f"W_a is a {'tight' if eff_rank_exp < W.shape[0]*0.5 else 'wide'} bottleneck."
    )


@experiment(32, "Layer-wise W_a Equivalents",
            "Which transformer layer is functionally closest to W_a?")
def exp_032(ds: Dataset) -> dict:
    # use first ~40 examples' per-layer hiddens to estimate, per layer l,
    # how similar the map hidden_{agent_i, step_t, layer=L} -> post_aligned is
    # to the W_a map, via CKA between linear-projected hidden and post_aligned.
    W = ds.wa[ds.tasks[0]]
    sample = ds.all_examples()[:40]
    if not sample:
        return finding({}, 0, "no data", status="skipped")
    layer_scores = None
    for ex in sample:
        pl = ex.per_layer()   # [A, m, L+1, D]
        pre, post, _ = ex.latents()
        A, m, Lp1, D = pl.shape
        flat_post = post.reshape(-1, D)  # [A*m, D]
        if layer_scores is None:
            layer_scores = np.zeros(Lp1)
            counts = np.zeros(Lp1)
        for l in range(Lp1):
            h = pl[:, :, l, :].reshape(-1, D)
            # similarity of h to post via CKA
            layer_scores[l] += cka_linear(h, flat_post)
            counts[l] += 1
        ex.drop_heavy()
    layer_scores = layer_scores / np.maximum(counts, 1)
    best = int(np.argmax(layer_scores))
    return finding(
        metrics=dict(best_layer=best, best_cka=float(layer_scores[best]),
                     n_layers=int(len(layer_scores)),
                     cka_profile=[float(x) for x in layer_scores]),
        score=float(layer_scores[best]),
        msg=f"Layer {best}/{len(layer_scores)-1} is most aligned (CKA "
            f"{layer_scores[best]:.3f}) with W_a's output geometry."
    )


@experiment(33, "W_a Condition Number / Error amplification",
            "Does W_a's condition number predict error amplification?")
def exp_033(ds: Dataset) -> dict:
    W = ds.wa[ds.tasks[0]]
    s = np.linalg.svd(W, compute_uv=False)
    cond = float(s[0] / (s[-1] + 1e-20))
    # measure amplification empirically: ||post|| / ||pre||
    light = ds.light_aggregate()
    pre = light["pre"]; post = light["post"]
    if pre.size == 0:
        return finding(dict(cond=cond), 0, "no data", status="skipped")
    pre_n = np.linalg.norm(pre, axis=-1); post_n = np.linalg.norm(post, axis=-1)
    ratios = post_n / (pre_n + 1e-9)
    return finding(
        metrics=dict(condition_number=cond,
                     mean_post_over_pre=float(ratios.mean()),
                     std_ratio=float(ratios.std()),
                     max_ratio=float(ratios.max())),
        score=min(1.0, math.log10(cond + 1) / 6),
        msg=f"W_a condition number {cond:.2e}; empirical mean "
            f"||post||/||pre|| = {ratios.mean():.3f}."
    )


# ===========================================================================
# TRACK 2 — representation / probing
# ===========================================================================
@experiment(2, "Emergent Private Latent Vocabulary",
            "Do agents re-use a consistent set of latent activation patterns?")
def exp_002(ds: Dataset) -> dict:
    light = ds.light_aggregate()
    post = light["post"]
    if post.size == 0:
        return finding({}, 0, "no data", status="skipped")
    N, A, m, D = post.shape
    X = post.reshape(-1, D)
    # k-means via numpy (lloyd, small K)
    k = 64
    rng = np.random.default_rng(0)
    idx = rng.choice(len(X), size=k, replace=False)
    cent = X[idx].copy()
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    for _ in range(10):
        Cn = cent / (np.linalg.norm(cent, axis=1, keepdims=True) + 1e-9)
        sims = Xn @ Cn.T
        assign = sims.argmax(axis=1)
        for c in range(k):
            mask = assign == c
            if mask.any():
                cent[c] = X[mask].mean(axis=0)
    # cluster size distribution → reuse
    sizes = np.bincount(assign, minlength=k).astype(float)
    H = safe_entropy(sizes / sizes.sum())
    max_H = math.log(k)
    reuse = 1 - H / max_H  # 0 = all unique, 1 = all same
    # count of "common thoughts" (clusters with > 1% mass)
    common = int((sizes / sizes.sum() > 0.01).sum())
    return finding(
        metrics=dict(k=k, entropy=H, max_entropy=max_H,
                     reuse_ratio=float(reuse),
                     n_common_clusters=common,
                     largest_cluster_frac=float(sizes.max() / sizes.sum())),
        score=float(reuse),
        msg=f"Latent thoughts cluster into {common} common 'words' "
            f"(reuse ratio {reuse:.3f}). Evidence for private vocabulary."
    )


@experiment(3, "Cross-Agent Mind-Reading",
            "Can a probe on agent B's incoming memory recover agent A's state?")
def exp_003(ds: Dataset) -> dict:
    light = ds.light_aggregate()
    post = light["post"]  # [N, A, m, D]
    if post.shape[1] < 2:
        return finding({}, 0, "need ≥2 agents", status="skipped")
    # Linear probe: predict agent_0's post[:,0,0,:] from concat of
    # post[:,1,0,:] (critic first thought). Use split cross-val R^2.
    X = post[:, 1, 0, :]
    Y = post[:, 0, 0, :]
    N = X.shape[0]
    rng = np.random.default_rng(0)
    perm = rng.permutation(N)
    split = int(0.8 * N)
    tr, te = perm[:split], perm[split:]
    # ridge
    lam = 1e-2
    Xtr = X[tr]; Ytr = Y[tr]
    A = Xtr.T @ Xtr + lam * np.eye(X.shape[1])
    B = Xtr.T @ Ytr
    W = np.linalg.solve(A, B)
    pred = X[te] @ W
    ss_res = ((pred - Y[te]) ** 2).sum()
    ss_tot = ((Y[te] - Y[te].mean(axis=0)) ** 2).sum()
    r2 = float(1 - ss_res / (ss_tot + 1e-20))
    return finding(
        metrics=dict(r2_cross_agent=r2, n_test=int(len(te))),
        score=max(0.0, r2),
        msg=f"Linear probe B→A reconstructs with R²={r2:.3f}."
    )


@experiment(9, "Uncertainty Encoding",
            "Do latent-thought norms correlate with correctness better than text?")
def exp_009(ds: Dataset) -> dict:
    light = ds.light_aggregate()
    post = light["post"]; correct = light["correct"]
    if post.size == 0:
        return finding({}, 0, "no data", status="skipped")
    norm = np.linalg.norm(post, axis=-1).mean(axis=(1, 2))  # [N]
    var = post.var(axis=(1, 2)).mean(axis=-1)  # [N]
    auc_norm = auc_roc(norm, correct.astype(int))
    auc_var = auc_roc(var, correct.astype(int))
    eff = max(abs(auc_norm - 0.5), abs(auc_var - 0.5))
    return finding(
        metrics=dict(auc_norm_vs_correct=auc_norm,
                     auc_variance_vs_correct=auc_var,
                     pearson_norm_correct=pearson(norm, correct.astype(float))),
        score=min(1.0, eff * 2),
        msg=f"Latent norm AUC vs correctness={auc_norm:.3f}; "
            f"variance AUC={auc_var:.3f}."
    )


@experiment(10, "Emergent Role Specialization",
            "Do agents occupy stable, distinct regions of latent space?")
def exp_010(ds: Dataset) -> dict:
    light = ds.light_aggregate()
    post = light["post"]
    if post.size == 0 or post.shape[1] < 2:
        return finding({}, 0, "need ≥2 agents", status="skipped")
    N, A, m, D = post.shape
    X = post.reshape(N * A * m, D)
    y = np.repeat(np.arange(A), m); y = np.tile(y, N)
    # compute centroid separation vs within-agent spread
    centroids = np.stack([X[y == a].mean(0) for a in range(A)])
    within = np.mean([np.linalg.norm(X[y == a] - centroids[a], axis=1).mean()
                      for a in range(A)])
    between = np.mean([np.linalg.norm(centroids[i] - centroids[j])
                       for i in range(A) for j in range(i + 1, A)])
    fisher = between / (within + 1e-9)
    return finding(
        metrics=dict(within=float(within), between=float(between),
                     fisher=float(fisher), agents=int(A)),
        score=min(1.0, fisher / 2),
        msg=f"Agent separability (between/within) = {fisher:.3f}. "
            f"{'Strong' if fisher > 1 else 'Weak'} role specialisation."
    )


@experiment(34, "Cross-Task Latent Clustering",
            "Do task-independent latent clusters emerge?")
def exp_034(ds: Dataset) -> dict:
    light = ds.light_aggregate()
    post = light["post"]
    tasks = light["tasks"]
    if post.size == 0 or len(set(tasks)) < 2:
        return finding({}, 0, "need multiple tasks", status="skipped")
    # PCA to 8, then measure overlap between per-task distributions
    X = post.mean(axis=(1, 2))  # [N, D]
    V, _, mu = pca(X, 8)
    Z = (X - mu) @ V.T
    # per-task centroid separation
    tset = sorted(set(tasks))
    cents = np.stack([Z[tasks == t].mean(0) for t in tset])
    within = np.mean([np.linalg.norm(Z[tasks == t] - cents[i], axis=1).mean()
                      for i, t in enumerate(tset)])
    between = np.mean([np.linalg.norm(cents[i] - cents[j])
                       for i in range(len(tset)) for j in range(i + 1, len(tset))])
    task_sep = between / (within + 1e-9)
    # universal = 1 - task_separation (in [0,1] rough)
    universal = float(1 / (1 + task_sep))
    return finding(
        metrics=dict(task_separation=float(task_sep),
                     universal_cluster_score=universal,
                     n_tasks=len(tset)),
        score=universal,
        msg=f"Tasks separate with ratio {task_sep:.2f}. Universal-primitive "
            f"score {universal:.2f}."
    )


@experiment(35, "Dead Dimensions",
            "What fraction of hidden dims carry near-zero variance?")
def exp_035(ds: Dataset) -> dict:
    light = ds.light_aggregate()
    post = light["post"]
    if post.size == 0:
        return finding({}, 0, "no data", status="skipped")
    v = post.reshape(-1, post.shape[-1]).var(axis=0)
    thr = v.max() * 1e-3
    dead = float((v < thr).mean())
    return finding(
        metrics=dict(dim=int(len(v)), dead_frac=dead,
                     mean_var=float(v.mean()),
                     max_var=float(v.max())),
        score=dead,
        msg=f"{dead*100:.1f}% of hidden dims are effectively dead."
    )


@experiment(36, "Prompt vs latent alignment",
            "How aligned are prompt_hidden and latent_thoughts representations?")
def exp_036(ds: Dataset) -> dict:
    sample = ds.all_examples()[:60]
    ckas = []
    for ex in sample:
        try:
            _, post, agents = ex.latents()
            ph = ex.prompt()
            for i, a in enumerate(agents):
                if a not in ph:
                    continue
                ckas.append(cka_linear(post[i], ph[a]))
        except Exception:
            pass
        ex.drop_heavy()
    if not ckas:
        return finding({}, 0, "no data", status="skipped")
    m = float(np.mean(ckas))
    return finding(
        metrics=dict(cka_mean=m, cka_std=float(np.std(ckas)), n=len(ckas)),
        score=m,
        msg=f"CKA(prompt, latent) = {m:.3f}."
    )


@experiment(37, "Task Identity Decodable",
            "Can we decode task from latent thought alone?")
def exp_037(ds: Dataset) -> dict:
    light = ds.light_aggregate()
    post = light["post"]; tasks = light["tasks"]
    if post.size == 0 or len(set(tasks)) < 2:
        return finding({}, 0, "need ≥2 tasks", status="skipped")
    X = post.mean(axis=(1, 2))
    # LDA-ish: nearest-centroid CV accuracy
    tset = sorted(set(tasks))
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(X))
    split = int(0.7 * len(X))
    tr, te = perm[:split], perm[split:]
    cents = np.stack([X[tr][tasks[tr] == t].mean(0) for t in tset])
    dists = np.linalg.norm(X[te][:, None, :] - cents[None, :, :], axis=-1)
    pred = np.array(tset)[dists.argmin(axis=1)]
    acc = float((pred == tasks[te]).mean())
    return finding(
        metrics=dict(accuracy=acc, n_classes=len(tset), chance=1 / len(tset)),
        score=max(0.0, acc - 1 / len(tset)) / (1 - 1 / len(tset)),
        msg=f"Task-decoding accuracy {acc:.3f} (chance {1/len(tset):.3f})."
    )


@experiment(38, "First-Agent Anchoring",
            "Does agent 0's latent disproportionately anchor later agents?")
def exp_038(ds: Dataset) -> dict:
    light = ds.light_aggregate()
    post = light["post"]
    if post.shape[1] < 3:
        return finding({}, 0, "need ≥3 agents", status="skipped")
    # cosine sim of agent_t's first thought with agent_0's vs agent_{t-1}'s
    N, A, m, D = post.shape
    a0_cos, prev_cos = [], []
    for n in range(N):
        for t in range(1, A):
            a0_cos.append(cosine(post[n, 0, 0], post[n, t, 0]))
            prev_cos.append(cosine(post[n, t - 1, 0], post[n, t, 0]))
    a0 = float(np.mean(a0_cos)); pr = float(np.mean(prev_cos))
    anchoring = a0 - pr
    return finding(
        metrics=dict(mean_cos_with_agent0=a0, mean_cos_with_prev=pr,
                     anchoring=anchoring),
        score=min(1.0, max(0.0, anchoring * 3)),
        msg=f"Agent-0 anchor sim {a0:.3f} vs prev-agent sim {pr:.3f} "
            f"(Δ={anchoring:+.3f})."
    )


@experiment(39, "Semantic Direction Inventory",
            "Are there consistent 'semantic axes' in latent space?")
def exp_039(ds: Dataset) -> dict:
    # direction = mean(correct) - mean(incorrect), then measure projection
    # stability per example
    light = ds.light_aggregate()
    post = light["post"]; corr = light["correct"]
    if post.size == 0 or corr.sum() < 5 or (~corr).sum() < 5:
        return finding({}, 0, "need both classes", status="skipped")
    x = post.mean(axis=(1, 2))
    axis_correct = x[corr].mean(0) - x[~corr].mean(0)
    axis_correct /= np.linalg.norm(axis_correct) + 1e-9
    # separate by task
    tasks = light["tasks"]
    per_task = {}
    for t in sorted(set(tasks)):
        mask = tasks == t
        if corr[mask].sum() < 2 or (~corr[mask]).sum() < 2:
            continue
        ax = x[mask & corr].mean(0) - x[mask & ~corr].mean(0)
        ax /= np.linalg.norm(ax) + 1e-9
        per_task[t] = cosine(ax, axis_correct)
    mean_stab = float(np.mean(list(per_task.values()))) if per_task else 0.0
    return finding(
        metrics=dict(per_task_correct_axis_cosine=per_task,
                     mean_stability=mean_stab),
        score=max(0.0, mean_stab),
        msg=f"'Correctness' semantic axis cross-task stability {mean_stab:.3f}."
    )


@experiment(40, "Correct vs Incorrect Geometry",
            "Where in latent space do right/wrong trajectories diverge?")
def exp_040(ds: Dataset) -> dict:
    light = ds.light_aggregate()
    post = light["post"]; corr = light["correct"]
    if post.size == 0 or corr.sum() < 5 or (~corr).sum() < 5:
        return finding({}, 0, "need both classes", status="skipped")
    N, A, m, D = post.shape
    # per round (t), compute centroid distance between classes
    seps = []
    for t in range(m):
        for a in range(A):
            c = post[:, a, t, :]
            d = np.linalg.norm(c[corr].mean(0) - c[~corr].mean(0))
            within = np.linalg.norm(c - c.mean(0), axis=1).mean()
            seps.append((a, t, float(d / (within + 1e-9))))
    seps.sort(key=lambda r: -r[2])
    best = seps[0]
    return finding(
        metrics=dict(best_agent=best[0], best_round=best[1],
                     best_separation_ratio=best[2],
                     top5=[{"agent": a, "round": t, "sep": s}
                           for a, t, s in seps[:5]]),
        score=min(1.0, best[2] / 2),
        msg=f"Max correct/incorrect separation at agent {best[0]} round "
            f"{best[1]} (ratio {best[2]:.3f})."
    )


# ===========================================================================
# TRACK 3 — information theory
# ===========================================================================
def _mi_gaussian(X: np.ndarray, Y: np.ndarray) -> float:
    """Gaussian-assumption MI lower bound between [N,Dx] and [N,Dy] via
    log-det covariances (nats)."""
    if len(X) < 4:
        return 0.0
    Z = np.concatenate([X, Y], axis=1)
    def _logdet(C):
        s = np.linalg.svd(C, compute_uv=False)
        s = s[s > 1e-10]
        return float(np.log(s).sum())
    Cx = np.cov(X, rowvar=False); Cy = np.cov(Y, rowvar=False)
    Cz = np.cov(Z, rowvar=False)
    # regularise
    for C in (Cx, Cy, Cz):
        C += 1e-4 * np.eye(C.shape[0])
    return float(0.5 * (_logdet(Cx) + _logdet(Cy) - _logdet(Cz)))


@experiment(5, "Information Compression",
            "How much is preserved when thought is compressed into m tokens?")
def exp_005(ds: Dataset) -> dict:
    light = ds.light_aggregate()
    pre = light["pre"]; post = light["post"]
    if pre.size == 0:
        return finding({}, 0, "no data", status="skipped")
    # MI-proxy: CKA between pre and post
    pre_flat = pre.reshape(-1, pre.shape[-1])
    post_flat = post.reshape(-1, post.shape[-1])
    k = min(2000, len(pre_flat))
    rng = np.random.default_rng(0)
    sel = rng.choice(len(pre_flat), k, replace=False)
    c = cka_linear(pre_flat[sel], post_flat[sel])
    # info density: explained variance of top-1 PCA component of post
    V, ev, _ = pca(post_flat[sel], 16)
    total = post_flat[sel].var(axis=0).sum()
    pc1 = float(ev[0] / total)
    return finding(
        metrics=dict(cka_pre_post=c, pca1_var_ratio=pc1),
        score=c,
        msg=f"Pre→post CKA {c:.3f}; 1st PC holds {pc1*100:.1f}% of latent variance."
    )


@experiment(6, "Information Propagation / Decay",
            "Does useful info amplify or decay across rounds?")
def exp_006(ds: Dataset) -> dict:
    light = ds.light_aggregate()
    post = light["post"]
    if post.shape[2] < 2:
        return finding({}, 0, "need ≥2 rounds", status="skipped")
    m = post.shape[2]
    norms_per_t = [float(np.linalg.norm(post[:, :, t, :], axis=-1).mean())
                   for t in range(m)]
    dec = (norms_per_t[-1] - norms_per_t[0]) / norms_per_t[0]
    return finding(
        metrics=dict(norm_per_step=norms_per_t,
                     first_to_last_delta=dec),
        score=min(1.0, abs(dec)),
        msg=f"Norm change across rounds: {dec*100:+.1f}%."
    )


@experiment(41, "Channel capacity of latent comm",
            "Empirical bits/token through the inter-agent latent channel.")
def exp_041(ds: Dataset) -> dict:
    light = ds.light_aggregate()
    post = light["post"]
    if post.shape[1] < 2:
        return finding({}, 0, "need ≥2 agents", status="skipped")
    # approximate MI between agent_0 first latent and agent_1 first latent in
    # a low-D PCA space
    X = post[:, 0, 0, :]; Y = post[:, 1, 0, :]
    _, _, mu = pca(np.concatenate([X, Y], 0), 32)
    Vx, _, mx = pca(X, 8); Vy, _, my = pca(Y, 8)
    Xp = (X - mx) @ Vx.T; Yp = (Y - my) @ Vy.T
    mi_nat = _mi_gaussian(Xp, Yp)
    mi_bits = mi_nat / math.log(2)
    return finding(
        metrics=dict(mi_bits_per_token=mi_bits),
        score=min(1.0, mi_bits / 32),
        msg=f"Gaussian MI proxy ≈ {mi_bits:.2f} bits/token."
    )


@experiment(42, "Redundancy coding across agents",
            "How much is the same info encoded by multiple agents?")
def exp_042(ds: Dataset) -> dict:
    light = ds.light_aggregate()
    post = light["post"]; corr = light["correct"].astype(int)
    if post.shape[1] < 2:
        return finding({}, 0, "need ≥2 agents", status="skipped")
    N, A, m, D = post.shape
    # pairwise cosine similarity between agents averaged over steps
    sims = []
    for i in range(A):
        for j in range(i + 1, A):
            for n in range(N):
                sims.append(cosine(post[n, i].ravel(), post[n, j].ravel()))
    sims = np.array(sims)
    mean_red = float(sims.mean())
    # correlation redundancy <-> correctness
    per_ex_red = np.zeros(N)
    k = 0
    for n in range(N):
        vals = []
        for i in range(A):
            for j in range(i + 1, A):
                vals.append(cosine(post[n, i].ravel(), post[n, j].ravel()))
        per_ex_red[n] = np.mean(vals)
    rho = pearson(per_ex_red, corr.astype(float))
    return finding(
        metrics=dict(mean_pairwise_cos=mean_red,
                     redundancy_correct_pearson=rho),
        score=min(1.0, abs(rho) * 4),
        msg=f"Cross-agent redundancy r={mean_red:.3f}; "
            f"pearson(redundancy, correct)={rho:+.3f}."
    )


@experiment(43, "Rate-distortion tradeoff",
            "Where does accuracy fall as we throw away PCA components?")
def exp_043(ds: Dataset) -> dict:
    light = ds.light_aggregate()
    post = light["post"]; corr = light["correct"].astype(int)
    if post.size == 0 or corr.sum() < 5 or (~corr.astype(bool)).sum() < 5:
        return finding({}, 0, "need both classes", status="skipped")
    X = post.mean(axis=(1, 2))
    V, _, mu = pca(X, 64)
    Z = (X - mu) @ V.T
    curve = {}
    for k in [1, 2, 4, 8, 16, 32, 64]:
        k = min(k, Z.shape[1])
        Zk = Z[:, :k]
        # linear AUC using dot with class-diff axis
        ax = Zk[corr == 1].mean(0) - Zk[corr == 0].mean(0)
        scores = Zk @ ax
        curve[k] = auc_roc(scores, corr)
    # find knee
    vals = list(curve.values())
    knee = max(curve.items(), key=lambda kv: kv[1])
    return finding(
        metrics=dict(auc_by_k=curve, knee_k=knee[0], knee_auc=knee[1]),
        score=float(knee[1]),
        msg=f"Rate-distortion: best discrimination AUC={knee[1]:.3f} at "
            f"k={knee[0]} PCA components."
    )


@experiment(44, "Fisher information by dim",
            "Which single dims discriminate correct vs incorrect the most?")
def exp_044(ds: Dataset) -> dict:
    light = ds.light_aggregate()
    post = light["post"]; corr = light["correct"]
    if corr.sum() < 5 or (~corr).sum() < 5:
        return finding({}, 0, "need both classes", status="skipped")
    X = post.mean(axis=(1, 2))
    mu0 = X[~corr].mean(0); mu1 = X[corr].mean(0)
    s2 = X.var(axis=0) + 1e-9
    fisher = (mu1 - mu0) ** 2 / s2
    top = np.argsort(-fisher)[:20]
    return finding(
        metrics=dict(max_fisher=float(fisher.max()),
                     top20_dims=top.tolist(),
                     top20_fisher=[float(fisher[i]) for i in top],
                     mean_fisher=float(fisher.mean())),
        score=min(1.0, float(fisher.max()) / 5),
        msg=f"Max per-dim Fisher {fisher.max():.3f}; "
            f"top dim {int(top[0])}."
    )


@experiment(45, "Kolmogorov-proxy via gzip",
            "Are simpler (more compressible) thoughts more often correct?")
def exp_045(ds: Dataset) -> dict:
    import gzip
    light = ds.light_aggregate()
    post = light["post"]; corr = light["correct"]
    if post.size == 0:
        return finding({}, 0, "no data", status="skipped")
    # quantize to int8
    q = np.clip(np.round(post / (np.abs(post).max() + 1e-9) * 127), -127, 127).astype(np.int8)
    sizes = []
    for n in range(len(q)):
        sizes.append(len(gzip.compress(q[n].tobytes())))
    sizes = np.array(sizes, float)
    r = pearson(sizes, corr.astype(float))
    return finding(
        metrics=dict(mean_size=float(sizes.mean()),
                     pearson_size_correct=r),
        score=min(1.0, abs(r) * 4),
        msg=f"gzip(size) vs correct pearson={r:+.3f}."
    )


@experiment(46, "Mutual information between rounds",
            "Working-memory half-life across rounds")
def exp_046(ds: Dataset) -> dict:
    light = ds.light_aggregate()
    post = light["post"]
    if post.shape[2] < 2:
        return finding({}, 0, "need ≥2 rounds", status="skipped")
    N, A, m, D = post.shape
    # cosine sim between round t and t+k, averaged across agents
    decays = []
    for k in range(1, m):
        cs = []
        for n in range(N):
            for a in range(A):
                cs.append(cosine(post[n, a, 0], post[n, a, k]))
        decays.append((k, float(np.mean(cs))))
    return finding(
        metrics=dict(decay=[{"lag": k, "cos": v} for k, v in decays]),
        score=1.0 - min(1.0, decays[-1][1]) if decays else 0.0,
        msg=f"Cos(round0, round{decays[-1][0]})={decays[-1][1]:.3f}; "
            f"higher ⇒ longer memory."
    )


# ===========================================================================
# TRACK 4 — geometry / topology
# ===========================================================================
@experiment(26, "Universal Conceptual Geometry (W_a manifold)",
            "Does W_a preserve relational / manifold structure?")
def exp_026(ds: Dataset) -> dict:
    light = ds.light_aggregate()
    pre = light["pre"]; post = light["post"]
    if pre.size == 0:
        return finding({}, 0, "no data", status="skipped")
    X = pre.reshape(-1, pre.shape[-1])
    Y = post.reshape(-1, post.shape[-1])
    k = min(1000, len(X))
    rng = np.random.default_rng(0)
    sel = rng.choice(len(X), k, replace=False)
    # compare pairwise distance matrices
    Dx = np.linalg.norm(X[sel][:, None, :] - X[sel][None, :, :], axis=-1)
    Dy = np.linalg.norm(Y[sel][:, None, :] - Y[sel][None, :, :], axis=-1)
    mask = ~np.eye(k, dtype=bool)
    r = pearson(Dx[mask], Dy[mask])
    return finding(
        metrics=dict(distance_pearson=r, n=k),
        score=max(0.0, r),
        msg=f"Pre↔post pairwise-distance correlation r={r:.3f} (→ "
            f"{'preserved' if r > 0.8 else 'partially distorted'} geometry)."
    )


@experiment(47, "Geodesics in concept space",
            "Do trajectories follow direct paths?")
def exp_047(ds: Dataset) -> dict:
    light = ds.light_aggregate()
    post = light["post"]
    if post.shape[2] < 3:
        return finding({}, 0, "need ≥3 rounds", status="skipped")
    # direct distance vs path length
    N, A, m, D = post.shape
    ratios = []
    for n in range(N):
        for a in range(A):
            pts = post[n, a]
            path = sum(np.linalg.norm(pts[t + 1] - pts[t]) for t in range(m - 1))
            direct = np.linalg.norm(pts[-1] - pts[0]) + 1e-9
            ratios.append(direct / path)
    r = np.array(ratios)
    return finding(
        metrics=dict(straightness_mean=float(r.mean()),
                     straightness_std=float(r.std())),
        score=float(r.mean()),
        msg=f"Trajectory straightness {r.mean():.3f} (1=geodesic)."
    )


@experiment(48, "Persistent homology signature",
            "Topological features separating correct vs incorrect")
def exp_048(ds: Dataset) -> dict:
    light = ds.light_aggregate()
    post = light["post"]; corr = light["correct"]
    if post.size == 0:
        return finding({}, 0, "no data", status="skipped")
    # cheap proxy: "number of 0-dim persistent components" ≈ #clusters
    # separable via threshold on pairwise distances
    def n_components_at(th, pts):
        # count pieces in epsilon-graph
        n = len(pts)
        D = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1)
        parent = list(range(n))
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]; x = parent[x]
            return x
        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb: parent[ra] = rb
        for i in range(n):
            for j in range(i + 1, n):
                if D[i, j] < th: union(i, j)
        return len(set(find(i) for i in range(n)))
    N, A, m, D = post.shape
    flat = post.reshape(N, A * m, D)
    correct_pi, incorrect_pi = [], []
    ths = np.linspace(0.5, 5, 8)
    for n in range(min(N, 100)):
        pts = flat[n]
        traj_pi = [n_components_at(t, pts) for t in ths]
        (correct_pi if corr[n] else incorrect_pi).append(np.mean(traj_pi))
    if not correct_pi or not incorrect_pi:
        return finding({}, 0, "single class", status="skipped")
    diff = abs(np.mean(correct_pi) - np.mean(incorrect_pi))
    return finding(
        metrics=dict(mean_components_correct=float(np.mean(correct_pi)),
                     mean_components_incorrect=float(np.mean(incorrect_pi)),
                     topo_gap=float(diff)),
        score=min(1.0, diff / 3),
        msg=f"Avg 0-dim component count: correct={np.mean(correct_pi):.2f}, "
            f"incorrect={np.mean(incorrect_pi):.2f}."
    )


@experiment(49, "Fractal dimension of trajectories",
            "Box-counting dim of latent trajectory")
def exp_049(ds: Dataset) -> dict:
    light = ds.light_aggregate()
    post = light["post"]
    if post.shape[2] < 2:
        return finding({}, 0, "need ≥2 rounds", status="skipped")
    # Higuchi-like: length at different scales
    N, A, m, D = post.shape
    lens = []
    for n in range(N):
        for a in range(A):
            pts = post[n, a]
            lens.append(sum(np.linalg.norm(pts[t+1]-pts[t]) for t in range(m-1)))
    lens = np.array(lens)
    return finding(
        metrics=dict(mean_path_len=float(lens.mean()),
                     std_path_len=float(lens.std())),
        score=min(1.0, lens.std() / (lens.mean() + 1e-9)),
        msg=f"Path length mean={lens.mean():.2f} std={lens.std():.2f}."
    )


@experiment(50, "Voronoi partitioning",
            "Do agents partition space into balanced cells?")
def exp_050(ds: Dataset) -> dict:
    light = ds.light_aggregate()
    post = light["post"]
    if post.shape[1] < 2:
        return finding({}, 0, "need ≥2 agents", status="skipped")
    N, A, m, D = post.shape
    cents = post.mean(axis=(0, 2))  # [A, D]
    X = post.reshape(-1, D)
    dists = np.linalg.norm(X[:, None, :] - cents[None, :, :], axis=-1)
    owners = dists.argmin(axis=1)
    sizes = np.bincount(owners, minlength=A) / len(owners)
    balance = 1.0 - float(sizes.std())
    return finding(
        metrics=dict(cell_fracs=sizes.tolist(), balance=balance),
        score=balance,
        msg=f"Voronoi balance {balance:.3f} (1=equal)."
    )


@experiment(51, "Curvature of thought manifold",
            "Estimated curvature per task")
def exp_051(ds: Dataset) -> dict:
    light = ds.light_aggregate()
    post = light["post"]; tasks = light["tasks"]
    if post.shape[2] < 3:
        return finding({}, 0, "need ≥3 rounds", status="skipped")
    N, A, m, D = post.shape
    per_task = {}
    for t in sorted(set(tasks)):
        sel = np.where(tasks == t)[0]
        cs = []
        for n in sel:
            for a in range(A):
                pts = post[n, a]
                for i in range(1, m - 1):
                    v1 = pts[i] - pts[i - 1]; v2 = pts[i + 1] - pts[i]
                    cs.append(1 - cosine(v1, v2))  # deviation from straight
        per_task[t] = float(np.mean(cs)) if cs else 0.0
    vals = list(per_task.values())
    spread = float(np.std(vals))
    return finding(
        metrics=dict(per_task_curvature=per_task, cross_task_std=spread),
        score=min(1.0, spread * 4),
        msg="Per-task curvature: " + ", ".join(f"{k}={v:.3f}" for k, v in per_task.items())
    )


@experiment(52, "Dimensionality scaling",
            "Intrinsic dim vs task difficulty proxy (accuracy)")
def exp_052(ds: Dataset) -> dict:
    light = ds.light_aggregate()
    post = light["post"]; corr = light["correct"]; tasks = light["tasks"]
    if post.size == 0:
        return finding({}, 0, "no data", status="skipped")
    per_task = {}
    for t in sorted(set(tasks)):
        sel = tasks == t
        X = post[sel].reshape(-1, post.shape[-1])
        if len(X) < 50: continue
        V, ev, _ = pca(X, min(64, len(X) - 1))
        pr = participation_ratio(ev)
        acc = float(corr[sel].mean())
        per_task[t] = dict(pr=float(pr), acc=acc)
    if len(per_task) < 2:
        return finding(dict(per_task=per_task), 0, "single task", status="skipped")
    prs = [v["pr"] for v in per_task.values()]
    accs = [v["acc"] for v in per_task.values()]
    r = pearson(np.array(prs), np.array(accs))
    return finding(
        metrics=dict(per_task=per_task, pearson_pr_acc=r),
        score=min(1.0, abs(r)),
        msg=f"Intrinsic dim / acc correlation r={r:+.3f}."
    )


# ===========================================================================
# TRACK 5 — physics
# ===========================================================================
@experiment(15, "Flocking in thought space",
            "Alignment / cohesion / separation across agents")
def exp_015(ds: Dataset) -> dict:
    light = ds.light_aggregate()
    post = light["post"]
    if post.shape[1] < 2:
        return finding({}, 0, "need ≥2 agents", status="skipped")
    N, A, m, D = post.shape
    align, coh, sep = [], [], []
    for n in range(N):
        for t in range(m):
            vecs = post[n, :, t, :]
            ctr = vecs.mean(0)
            align.append(float(np.mean([cosine(vecs[i], vecs[j])
                              for i in range(A) for j in range(i + 1, A)])))
            coh.append(float(np.linalg.norm(vecs - ctr, axis=1).mean()))
            sep.append(float(np.min([np.linalg.norm(vecs[i] - vecs[j])
                             for i in range(A) for j in range(i + 1, A)])))
    return finding(
        metrics=dict(alignment=float(np.mean(align)),
                     cohesion=float(np.mean(coh)),
                     separation=float(np.mean(sep))),
        score=min(1.0, np.mean(align)),
        msg=f"align={np.mean(align):.3f}, cohesion={np.mean(coh):.2f}, "
            f"sep={np.mean(sep):.2f}."
    )


@experiment(16, "Latent entanglement",
            "Are agent-pair correlations stronger than direct-channel prediction?")
def exp_016(ds: Dataset) -> dict:
    light = ds.light_aggregate()
    post = light["post"]
    if post.shape[1] < 2:
        return finding({}, 0, "need ≥2 agents", status="skipped")
    N, A, m, D = post.shape
    # shuffle-null: correlation between agent0 and agent1 vs shuffled baseline
    def _corr(x, y):
        return pearson(x.ravel(), y.ravel())
    real = []; shuf = []
    rng = np.random.default_rng(0)
    for n in range(N):
        real.append(_corr(post[n, 0], post[n, 1]))
        j = rng.integers(0, N)
        while j == n and N > 1:
            j = rng.integers(0, N)
        shuf.append(_corr(post[n, 0], post[j, 1]))
    ent = float(np.mean(real) - np.mean(shuf))
    return finding(
        metrics=dict(mean_real_corr=float(np.mean(real)),
                     mean_shuffled_corr=float(np.mean(shuf)),
                     entanglement_excess=ent),
        score=min(1.0, max(0.0, ent) * 4),
        msg=f"Corr(real)−Corr(shuffled) = {ent:+.3f}."
    )


@experiment(19, "Superposition of reasoning paths",
            "Does one latent encode multiple candidate answers?")
def exp_019(ds: Dataset) -> dict:
    light = ds.light_aggregate()
    post = light["post"]
    if post.size == 0:
        return finding({}, 0, "no data", status="skipped")
    N, A, m, D = post.shape
    # measure 'superposition' as participation ratio of singular-value
    # decomposition of each latent thought viewed as matrix [m, D]
    prs = []
    for n in range(N):
        for a in range(A):
            s = np.linalg.svd(post[n, a], compute_uv=False)
            prs.append(participation_ratio(s ** 2))
    m_pr = float(np.mean(prs))
    return finding(
        metrics=dict(mean_PR=m_pr, max_PR=float(np.max(prs))),
        score=min(1.0, m_pr / max(1, post.shape[2])),
        msg=f"Mean per-thought participation ratio {m_pr:.3f} "
            f"(of m={post.shape[2]})."
    )


@experiment(25, "Attractor / entropy decrease",
            "Does entropy decrease across rounds like dissipation?")
def exp_025(ds: Dataset) -> dict:
    light = ds.light_aggregate()
    post = light["post"]
    if post.shape[2] < 2:
        return finding({}, 0, "need ≥2 rounds", status="skipped")
    # proxy entropy: log-det of covariance per-round over agents
    N, A, m, D = post.shape
    ents = []
    for t in range(m):
        X = post[:, :, t, :].reshape(-1, D)
        v = X.var(axis=0)
        ents.append(float(np.log(v[v > 1e-9]).sum()))
    drop = ents[0] - ents[-1]
    return finding(
        metrics=dict(entropy_per_round=ents, entropy_drop=float(drop)),
        score=min(1.0, max(0.0, drop) / 500),
        msg=f"Entropy proxy: {ents[0]:.1f} → {ents[-1]:.1f} (Δ={drop:+.1f})."
    )


@experiment(53, "Phase transitions in collaboration",
            "Does coherence jump past a critical agent count?")
def exp_053(ds: Dataset) -> dict:
    light = ds.light_aggregate()
    post = light["post"]
    if post.shape[1] < 2:
        return finding({}, 0, "need ≥2 agents", status="skipped")
    N, A, m, D = post.shape
    cohs = []
    for k in range(1, A + 1):
        sub = post[:, :k, :, :]
        mean_vec = sub.mean(axis=1, keepdims=True)
        coh = float(np.mean([cosine(sub[n, a].ravel(), mean_vec[n, 0].ravel())
                             for n in range(N) for a in range(k)]))
        cohs.append((k, coh))
    diffs = np.diff([v for _, v in cohs])
    big = float(diffs.max()) if len(diffs) else 0.0
    return finding(
        metrics=dict(coherence_curve=[dict(k=k, coh=v) for k, v in cohs],
                     max_jump=big),
        score=min(1.0, big * 2),
        msg=f"Coherence vs #agents max step {big:+.3f}."
    )


@experiment(54, "Diffusion dynamics",
            "Does concept spread follow Fickian diffusion?")
def exp_054(ds: Dataset) -> dict:
    light = ds.light_aggregate()
    post = light["post"]
    if post.shape[2] < 2:
        return finding({}, 0, "need ≥2 rounds", status="skipped")
    # variance of ensemble across rounds — Fickian: var ~ t
    N, A, m, D = post.shape
    vars_t = []
    for t in range(m):
        X = post[:, :, t, :].reshape(-1, D)
        vars_t.append(float(X.var(axis=0).sum()))
    ts = np.arange(1, m + 1)
    slope = float(np.polyfit(np.log(ts), np.log(vars_t), 1)[0])
    return finding(
        metrics=dict(var_per_step=vars_t, log_log_slope=slope),
        score=min(1.0, abs(slope - 1) < 0.3 and 1.0 or abs(1.0 / (abs(slope - 1) + 0.3))),
        msg=f"var ∝ t^{slope:.2f} (Fickian if ≈1)."
    )


@experiment(55, "Percolation threshold",
            "Masking dims until accuracy proxy collapses")
def exp_055(ds: Dataset) -> dict:
    light = ds.light_aggregate()
    post = light["post"]; corr = light["correct"]
    if post.size == 0 or corr.sum() < 5 or (~corr).sum() < 5:
        return finding({}, 0, "need both classes", status="skipped")
    X = post.mean(axis=(1, 2))
    ax = X[corr].mean(0) - X[~corr].mean(0)
    rng = np.random.default_rng(0)
    curve = {}
    for drop in [0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95]:
        mask = rng.random(X.shape[1]) > drop
        scores = (X * mask) @ ax
        curve[drop] = auc_roc(scores, corr.astype(int))
    # find threshold
    thresh = None
    last = curve[0.0]
    for d, v in sorted(curve.items()):
        if last - v > 0.1:
            thresh = d; break
        last = v
    return finding(
        metrics=dict(auc_by_drop=curve, percolation_threshold=thresh),
        score=1.0 if thresh else 0.0,
        msg=f"Percolation threshold ≈ {thresh}."
    )


@experiment(56, "Spin-glass: metastable failures",
            "Do failed examples cluster into local minima?")
def exp_056(ds: Dataset) -> dict:
    light = ds.light_aggregate()
    post = light["post"]; corr = light["correct"]
    if corr.sum() < 5 or (~corr).sum() < 5:
        return finding({}, 0, "need both classes", status="skipped")
    X = post.mean(axis=(1, 2))
    # cluster incorrect: k-means then see if clusters are tight
    Xw = X[~corr]
    k = min(8, len(Xw) // 3)
    if k < 2: return finding({}, 0, "too few", status="skipped")
    rng = np.random.default_rng(0)
    idx = rng.choice(len(Xw), k, replace=False)
    cent = Xw[idx].copy()
    xw_sq = (Xw ** 2).sum(1)
    for _ in range(8):
        c_sq = (cent ** 2).sum(1)
        d2 = xw_sq[:, None] + c_sq[None, :] - 2 * (Xw @ cent.T)
        a = d2.argmin(1)
        for c in range(k):
            if (a == c).any(): cent[c] = Xw[a == c].mean(0)
    within = float(np.mean([np.linalg.norm(Xw[a == c] - cent[c], axis=1).mean()
                            for c in range(k) if (a == c).any()]))
    overall = float(np.linalg.norm(Xw - Xw.mean(0), axis=1).mean())
    tight = 1 - within / (overall + 1e-9)
    return finding(
        metrics=dict(within=within, overall=overall, glass_tightness=tight, k=k),
        score=max(0.0, tight),
        msg=f"Failure clusters tightness {tight:.3f}."
    )


@experiment(57, "Maxwell's demon dims",
            "Which dims best sort signal from noise?")
def exp_057(ds: Dataset) -> dict:
    light = ds.light_aggregate()
    post = light["post"]; corr = light["correct"]
    if corr.sum() < 5 or (~corr).sum() < 5:
        return finding({}, 0, "need both", status="skipped")
    X = post.mean(axis=(1, 2))
    # per-dim AUC
    aucs = np.array([auc_roc(X[:, d], corr.astype(int)) for d in range(X.shape[1])])
    gate = np.abs(aucs - 0.5)
    top = np.argsort(-gate)[:10]
    # minimum gatekeeper set for AUC ≥ 0.7
    order = np.argsort(-gate)
    cum = 0
    k_needed = None
    for k in [1, 2, 4, 8, 16, 32, 64, 128]:
        k = min(k, X.shape[1])
        ax_dims = order[:k]
        scores = X[:, ax_dims] @ (X[corr][:, ax_dims].mean(0) - X[~corr][:, ax_dims].mean(0))
        a = auc_roc(scores, corr.astype(int))
        if a >= 0.7 and k_needed is None: k_needed = k
    return finding(
        metrics=dict(top10_dims=top.tolist(),
                     top10_abs_auc_minus_half=[float(gate[i]) for i in top],
                     gatekeepers_for_auc_07=k_needed),
        score=float(gate.max() * 2),
        msg=f"{k_needed} gatekeeper dims reach AUC ≥ 0.7."
    )


@experiment(59, "Renormalization across layers",
            "Does info coarse-grain like an RG flow through transformer layers?")
def exp_059(ds: Dataset) -> dict:
    sample = ds.all_examples()[:20]
    if not sample:
        return finding({}, 0, "no data", status="skipped")
    ent_by_layer = None; counts = None
    for ex in sample:
        pl = ex.per_layer()   # [A, m, L+1, D]
        if ent_by_layer is None:
            ent_by_layer = np.zeros(pl.shape[2]); counts = 0
        for l in range(pl.shape[2]):
            v = pl[:, :, l, :].reshape(-1, pl.shape[-1]).var(axis=0)
            ent_by_layer[l] += float(np.log(v[v > 1e-9]).sum())
        counts += 1
        ex.drop_heavy()
    ent_by_layer /= max(1, counts)
    drop = float(ent_by_layer[0] - ent_by_layer[-1])
    return finding(
        metrics=dict(entropy_profile=[float(x) for x in ent_by_layer],
                     first_to_last=drop),
        score=min(1.0, abs(drop) / 500),
        msg=f"log-var layer profile: {ent_by_layer[0]:.0f}→{ent_by_layer[-1]:.0f}."
    )


# ===========================================================================
# TRACK 6 — biology / ecology
# ===========================================================================
@experiment(17, "Evolutionary selection of patterns",
            "Do some latent patterns get amplified across rounds?")
def exp_017(ds: Dataset) -> dict:
    light = ds.light_aggregate()
    post = light["post"]
    if post.shape[2] < 2:
        return finding({}, 0, "need ≥2 rounds", status="skipped")
    # top PCA components at round 0 vs last: do the top directions' energies grow?
    X0 = post[:, :, 0, :].reshape(-1, post.shape[-1])
    XL = post[:, :, -1, :].reshape(-1, post.shape[-1])
    V0, ev0, _ = pca(X0, 8)
    ev_last = ((XL - XL.mean(0)) @ V0.T).var(axis=0)
    growth = float((ev_last / (ev0 + 1e-9)).mean())
    return finding(
        metrics=dict(top_dirs_energy_growth=growth,
                     ev0=[float(x) for x in ev0],
                     ev_last=[float(x) for x in ev_last]),
        score=min(1.0, abs(growth - 1)),
        msg=f"Top-8 directions energy ×{growth:.2f} from round 0 → last."
    )


@experiment(18, "Symbiotic vs parasitic exchanges",
            "Directional benefit between agent pairs")
def exp_018(ds: Dataset) -> dict:
    light = ds.light_aggregate()
    post = light["post"]; corr = light["correct"]
    if post.shape[1] < 2:
        return finding({}, 0, "need ≥2 agents", status="skipped")
    N, A, m, D = post.shape
    # parasitic: high cos between i,j but decreases norm of j over rounds
    pairs = {}
    for i in range(A):
        for j in range(A):
            if i == j: continue
            cs = [cosine(post[n, i, 0], post[n, j, -1]) for n in range(N)]
            dn = [(np.linalg.norm(post[n, j, -1]) - np.linalg.norm(post[n, j, 0]))
                  for n in range(N)]
            pairs[f"{i}->{j}"] = dict(mean_cos=float(np.mean(cs)),
                                       mean_norm_change=float(np.mean(dn)))
    return finding(
        metrics=dict(pairs=pairs),
        score=0.5,
        msg="Pairwise directional exchange stats recorded."
    )


@experiment(21, "Hebbian strengthening",
            "Do patterns that co-fire consistently across rounds get stronger?")
def exp_021(ds: Dataset) -> dict:
    light = ds.light_aggregate()
    post = light["post"]
    if post.shape[2] < 2:
        return finding({}, 0, "need ≥2 rounds", status="skipped")
    N, A, m, D = post.shape
    norms = np.linalg.norm(post, axis=-1)  # [N,A,m]
    growth = norms[:, :, -1] - norms[:, :, 0]
    return finding(
        metrics=dict(mean_norm_growth=float(growth.mean()),
                     pos_growth_frac=float((growth > 0).mean())),
        score=min(1.0, abs(growth.mean()) / max(1e-6, norms.mean())),
        msg=f"Norm grows {float((growth>0).mean())*100:.1f}% of the time."
    )


@experiment(22, "Predator-prey dynamics",
            "Asymmetric extraction between agents")
def exp_022(ds: Dataset) -> dict:
    light = ds.light_aggregate()
    post = light["post"]
    if post.shape[1] < 2:
        return finding({}, 0, "need ≥2 agents", status="skipped")
    N, A, m, D = post.shape
    # predator: agent whose downstream targets lose norm
    asym = np.zeros((A, A))
    for i in range(A):
        for j in range(A):
            if i == j: continue
            dn = (np.linalg.norm(post[:, j, -1], axis=-1) -
                  np.linalg.norm(post[:, j, 0], axis=-1))
            cs = np.array([cosine(post[n, i, 0], post[n, j, -1]) for n in range(N)])
            asym[i, j] = float(pearson(cs, dn))
    return finding(
        metrics=dict(predation_matrix=asym.tolist()),
        score=float(min(1.0, abs(asym).max())),
        msg=f"Max |predation| signal {abs(asym).max():.3f}."
    )


@experiment(61, "Quorum sensing threshold",
            "Minimum agent consensus before answer crystallises")
def exp_061(ds: Dataset) -> dict:
    light = ds.light_aggregate()
    post = light["post"]; corr = light["correct"]
    if post.shape[1] < 2 or corr.sum() < 5 or (~corr).sum() < 5:
        return finding({}, 0, "insufficient", status="skipped")
    # mean pairwise cos in final round vs correctness
    N, A, m, D = post.shape
    cos_last = []
    for n in range(N):
        vals = [cosine(post[n, i, -1], post[n, j, -1])
                for i in range(A) for j in range(i + 1, A)]
        cos_last.append(np.mean(vals))
    cos_last = np.array(cos_last)
    auc = auc_roc(cos_last, corr.astype(int))
    return finding(
        metrics=dict(auc_final_consensus_vs_correct=auc,
                     median_cos=float(np.median(cos_last))),
        score=abs(auc - 0.5) * 2,
        msg=f"Final-round consensus AUC vs correct={auc:.3f}."
    )


@experiment(62, "Immune response",
            "Do subsequent agents shift away from wrong intros?")
def exp_062(ds: Dataset) -> dict:
    light = ds.light_aggregate()
    post = light["post"]; corr = light["correct"]
    if post.shape[1] < 2 or post.shape[2] < 2:
        return finding({}, 0, "need ≥2A,2m", status="skipped")
    N, A, m, D = post.shape
    # for failed examples: how much does agent t's direction diverge from agent 0?
    divs_wrong, divs_right = [], []
    for n in range(N):
        dif = 1 - cosine(post[n, 0, 0], post[n, -1, -1])
        (divs_wrong if not corr[n] else divs_right).append(dif)
    d = float(np.mean(divs_wrong) - np.mean(divs_right)) if divs_wrong and divs_right else 0.0
    return finding(
        metrics=dict(divergence_wrong=float(np.mean(divs_wrong)) if divs_wrong else 0,
                     divergence_right=float(np.mean(divs_right)) if divs_right else 0,
                     gap=d),
        score=min(1.0, abs(d) * 4),
        msg=f"Divergence(wrong)−Divergence(right) = {d:+.3f}."
    )


@experiment(63, "Morphogenesis / role differentiation",
            "Do identical seeds diverge into specialized roles?")
def exp_063(ds: Dataset) -> dict:
    # re-use role separation metric across time
    light = ds.light_aggregate()
    post = light["post"]
    if post.shape[1] < 2 or post.shape[2] < 2:
        return finding({}, 0, "insufficient", status="skipped")
    N, A, m, D = post.shape
    # role-fisher at round 0 vs last
    def fisher(t):
        X = post[:, :, t, :]
        cents = X.mean(axis=0)  # [A, D]
        within = np.mean([np.linalg.norm(X[:, a] - cents[a], axis=1).mean()
                          for a in range(A)])
        between = np.mean([np.linalg.norm(cents[i] - cents[j])
                           for i in range(A) for j in range(i + 1, A)])
        return float(between / (within + 1e-9))
    f0 = fisher(0); fL = fisher(m - 1)
    return finding(
        metrics=dict(fisher_first=f0, fisher_last=fL,
                     differentiation_gain=fL - f0),
        score=min(1.0, max(0.0, fL - f0) / 2),
        msg=f"Role-Fisher {f0:.3f}→{fL:.3f}."
    )


@experiment(64, "Niche construction",
            "Do agents re-sculpt shared memory to favor themselves?")
def exp_064(ds: Dataset) -> dict:
    light = ds.light_aggregate()
    post = light["post"]
    if post.shape[2] < 2:
        return finding({}, 0, "need ≥2 rounds", status="skipped")
    # correlation between agent i's initial thought and agent i's last
    N, A, m, D = post.shape
    self_corr = []
    for n in range(N):
        for a in range(A):
            self_corr.append(cosine(post[n, a, 0], post[n, a, -1]))
    return finding(
        metrics=dict(mean_self_persistence=float(np.mean(self_corr))),
        score=float(max(0.0, np.mean(self_corr))),
        msg=f"Self-persistence {np.mean(self_corr):.3f}."
    )


@experiment(65, "GRN-like interaction topology",
            "Motif counts in the interaction graph")
def exp_065(ds: Dataset) -> dict:
    light = ds.light_aggregate()
    post = light["post"]
    if post.shape[1] < 3:
        return finding({}, 0, "need ≥3 agents", status="skipped")
    N, A, m, D = post.shape
    # edge weight i->j = mean cosine(post[i,0], post[j,-1]) - shuffled baseline
    edges = np.zeros((A, A))
    for i in range(A):
        for j in range(A):
            if i == j: continue
            edges[i, j] = float(np.mean([cosine(post[n, i, 0], post[n, j, -1])
                                         for n in range(N)]))
    # count feed-forward triples
    th = float(np.mean(edges[edges > 0]))
    bin_e = (edges > th).astype(int)
    ff = 0; fb = 0
    for i in range(A):
        for j in range(A):
            for k in range(A):
                if len({i, j, k}) < 3: continue
                if bin_e[i, j] and bin_e[j, k] and bin_e[i, k]: ff += 1
                if bin_e[i, j] and bin_e[j, i]: fb += 1
    return finding(
        metrics=dict(edge_matrix=edges.tolist(),
                     feed_forward_triangles=int(ff),
                     bidirectional_pairs=int(fb / 2)),
        score=min(1.0, ff / max(1, A * (A - 1) * (A - 2))),
        msg=f"{ff} feed-forward triangles, {fb//2} bidirectional pairs."
    )


@experiment(66, "Preferential adoption",
            "Do some agents get imitated more?")
def exp_066(ds: Dataset) -> dict:
    light = ds.light_aggregate()
    post = light["post"]
    if post.shape[1] < 3:
        return finding({}, 0, "need ≥3 agents", status="skipped")
    N, A, m, D = post.shape
    # for each agent: average cosine of their first thought with *later*
    # agents' first thought, normalised by cross-agent baseline
    pull = np.zeros(A)
    for i in range(A):
        cs = []
        for n in range(N):
            for j in range(A):
                if j <= i: continue
                cs.append(cosine(post[n, i, 0], post[n, j, 0]))
        pull[i] = float(np.mean(cs)) if cs else 0.0
    var = float(pull.std())
    return finding(
        metrics=dict(per_agent_pull=pull.tolist(), std=var),
        score=min(1.0, var * 4),
        msg=f"Pull variance across agents {var:.3f}."
    )


# ===========================================================================
# TRACK 7 — neuroscience / cognitive
# ===========================================================================
@experiment(67, "Cortical column specialization by layer",
            "Layer-wise specialisation pattern")
def exp_067(ds: Dataset) -> dict:
    sample = ds.all_examples()[:30]
    if not sample: return finding({}, 0, "no data", status="skipped")
    # per-layer variance profile averaged
    profile = None; cnt = 0
    for ex in sample:
        pl = ex.per_layer()
        v = pl.reshape(pl.shape[2], -1).var(axis=1)
        if profile is None: profile = np.zeros_like(v)
        profile += v; cnt += 1
        ex.drop_heavy()
    profile /= cnt
    # location of max
    peak = int(np.argmax(profile))
    return finding(
        metrics=dict(layer_variance_profile=[float(x) for x in profile],
                     peak_layer=peak),
        score=min(1.0, float(profile[peak]) / (profile.mean() + 1e-9) / 4),
        msg=f"Variance peaks at layer {peak}/{len(profile)-1}."
    )


@experiment(68, "Working-memory capacity limit",
            "How many semantic axes are simultaneously active?")
def exp_068(ds: Dataset) -> dict:
    light = ds.light_aggregate()
    post = light["post"]
    if post.size == 0: return finding({}, 0, "no data", status="skipped")
    # participation ratio of PCA eigenvalues of a single example's latent cloud
    prs = []
    N, A, m, D = post.shape
    for n in range(N):
        X = post[n].reshape(-1, D)
        _, ev, _ = pca(X, min(32, len(X) - 1))
        prs.append(participation_ratio(ev))
    return finding(
        metrics=dict(mean_PR=float(np.mean(prs)),
                     p10=float(np.percentile(prs, 10)),
                     p90=float(np.percentile(prs, 90))),
        score=min(1.0, float(np.mean(prs)) / 10),
        msg=f"Working-memory PR ≈ {np.mean(prs):.2f} (Miller limit ≈ 7±2)."
    )


@experiment(69, "Episodic vs semantic memory",
            "Are prompt_hidden and latent_thoughts geometrically distinct?")
def exp_069(ds: Dataset) -> dict:
    sample = ds.all_examples()[:60]
    ckas = []
    for ex in sample:
        try:
            _, post, agents = ex.latents()
            ph = ex.prompt()
            for i, a in enumerate(agents):
                if a in ph:
                    ckas.append(cka_linear(post[i], ph[a]))
        except Exception:
            pass
        ex.drop_heavy()
    if not ckas: return finding({}, 0, "no data", status="skipped")
    sep = 1.0 - float(np.mean(ckas))
    return finding(
        metrics=dict(cka_episodic_vs_semantic=float(np.mean(ckas)),
                     separation=sep),
        score=sep,
        msg=f"Prompt/latent separation {sep:.3f} (1=fully distinct subspaces)."
    )


@experiment(70, "Prediction-error / surprise",
            "Large agent disagreement → corrective update?")
def exp_070(ds: Dataset) -> dict:
    light = ds.light_aggregate()
    post = light["post"]
    if post.shape[1] < 2 or post.shape[2] < 2:
        return finding({}, 0, "insufficient", status="skipped")
    N, A, m, D = post.shape
    surprise = []; correction = []
    for n in range(N):
        for a in range(1, A):
            s = 1 - cosine(post[n, a - 1, 0], post[n, a, 0])
            c = np.linalg.norm(post[n, a, -1] - post[n, a, 0])
            surprise.append(s); correction.append(c)
    r = pearson(np.array(surprise), np.array(correction))
    return finding(
        metrics=dict(pearson_surprise_correction=r),
        score=min(1.0, abs(r) * 2),
        msg=f"Surprise↔correction pearson r={r:+.3f}."
    )


@experiment(71, "Mirror neurons",
            "Does agent B mirror agent A's latent direction?")
def exp_071(ds: Dataset) -> dict:
    light = ds.light_aggregate()
    post = light["post"]
    if post.shape[1] < 2: return finding({}, 0, "need ≥2 agents", status="skipped")
    N, A, m, D = post.shape
    cs = [cosine(post[n, 0, 0], post[n, 1, 0]) for n in range(N)]
    rng = np.random.default_rng(0)
    shuf = [cosine(post[n, 0, 0], post[rng.integers(0, N), 1, 0]) for n in range(N)]
    excess = float(np.mean(cs) - np.mean(shuf))
    return finding(
        metrics=dict(real=float(np.mean(cs)), shuf=float(np.mean(shuf)),
                     excess=excess),
        score=min(1.0, max(0.0, excess) * 4),
        msg=f"Mirroring excess {excess:+.3f}."
    )


@experiment(72, "Default mode on easy tasks",
            "Idle pattern on trivially-easy examples?")
def exp_072(ds: Dataset) -> dict:
    light = ds.light_aggregate()
    post = light["post"]; corr = light["correct"]
    if corr.sum() < 5: return finding({}, 0, "insufficient", status="skipped")
    # proxy: correct examples cluster tightly → "DMN"
    Xc = post[corr].reshape(corr.sum(), -1)
    Xi = post[~corr].reshape((~corr).sum(), -1) if (~corr).sum() > 0 else None
    sd_c = float(Xc.std(0).mean())
    sd_i = float(Xi.std(0).mean()) if Xi is not None else 0.0
    return finding(
        metrics=dict(std_correct=sd_c, std_incorrect=sd_i,
                     dmn_signature=float(sd_i - sd_c)),
        score=min(1.0, max(0.0, sd_i - sd_c) / (sd_c + 1e-9)),
        msg=f"std(correct)={sd_c:.3f}, std(incorrect)={sd_i:.3f}."
    )


@experiment(73, "Sleep / consolidation across rounds",
            "Do rounds integrate earlier info?")
def exp_073(ds: Dataset) -> dict:
    light = ds.light_aggregate()
    post = light["post"]
    if post.shape[2] < 3: return finding({}, 0, "need ≥3 rounds", status="skipped")
    # MI proxy between round 0 and round t increases?
    N, A, m, D = post.shape
    ckas = []
    for t in range(m):
        X0 = post[:, :, 0, :].reshape(-1, D)
        Xt = post[:, :, t, :].reshape(-1, D)
        k = min(1000, len(X0))
        rng = np.random.default_rng(0)
        sel = rng.choice(len(X0), k, replace=False)
        ckas.append(cka_linear(X0[sel], Xt[sel]))
    slope = float(np.polyfit(range(m), ckas, 1)[0])
    return finding(
        metrics=dict(cka_to_first=ckas, slope=slope),
        score=min(1.0, abs(slope) * 4),
        msg=f"CKA(round0, round t) linear slope {slope:+.3f}."
    )


@experiment(74, "Cognitive load direction",
            "Does a direction track reasoning complexity?")
def exp_074(ds: Dataset) -> dict:
    light = ds.light_aggregate()
    post = light["post"]; corr = light["correct"]
    if post.size == 0: return finding({}, 0, "no data", status="skipped")
    # proxy difficulty: norm of first-agent first latent vs task (tasks w/
    # lower accuracy = harder)
    tasks = light["tasks"]
    diff = np.zeros(len(tasks))
    for t in sorted(set(tasks)):
        diff[tasks == t] = 1 - float(corr[tasks == t].mean())
    norm = np.linalg.norm(post[:, 0, 0, :], axis=-1)
    r = pearson(norm, diff)
    return finding(
        metrics=dict(pearson_norm_difficulty=r),
        score=min(1.0, abs(r) * 4),
        msg=f"Pearson(norm, task-difficulty) = {r:+.3f}."
    )


@experiment(75, "Confirmation bias",
            "Do later rounds amplify initial direction?")
def exp_075(ds: Dataset) -> dict:
    light = ds.light_aggregate()
    post = light["post"]; corr = light["correct"]
    if post.shape[2] < 2: return finding({}, 0, "need ≥2 rounds", status="skipped")
    N, A, m, D = post.shape
    amp_wrong, amp_right = [], []
    for n in range(N):
        for a in range(A):
            axis = post[n, a, 0]; axis = axis / (np.linalg.norm(axis) + 1e-9)
            proj = post[n, a, -1] @ axis / (np.linalg.norm(post[n, a, -1]) + 1e-9)
            (amp_right if corr[n] else amp_wrong).append(float(proj))
    gap = float(np.mean(amp_wrong) - np.mean(amp_right)) if amp_wrong and amp_right else 0.0
    return finding(
        metrics=dict(amp_correct=float(np.mean(amp_right)) if amp_right else 0,
                     amp_wrong=float(np.mean(amp_wrong)) if amp_wrong else 0,
                     bias_gap=gap),
        score=min(1.0, abs(gap) * 4),
        msg=f"Bias gap (wrong−right) = {gap:+.3f}."
    )


@experiment(76, "Binding problem",
            "Orthogonal vs superposed multi-concept binding")
def exp_076(ds: Dataset) -> dict:
    light = ds.light_aggregate()
    post = light["post"]
    if post.shape[2] < 2: return finding({}, 0, "need ≥2 steps", status="skipped")
    # per thought, measure how "sparse" it is — low participation ratio
    # -> sparse (orthogonal) binding; high -> dense superposition
    N, A, m, D = post.shape
    prs = []
    for n in range(N):
        for a in range(A):
            # take m thoughts as rows
            X = post[n, a]
            G = X @ X.T  # m x m Gram
            ev = np.linalg.eigvalsh(G)
            prs.append(participation_ratio(np.clip(ev, 0, None)))
    m_pr = float(np.mean(prs))
    return finding(
        metrics=dict(mean_PR=m_pr),
        score=min(1.0, m_pr / max(1, post.shape[2])),
        msg=f"Binding PR {m_pr:.3f} / max {post.shape[2]}."
    )


# ===========================================================================
# TRACK 8 — social / psychology
# ===========================================================================
@experiment(8, "Faithfulness gap",
            "Latent vs text divergence of answer")
def exp_008(ds: Dataset) -> dict:
    # can't directly decode latents without model; use per-agent text from
    # text_outputs to compare short strings from different agents as a proxy
    gap = 0; total = 0
    for ex in ds.all_examples():
        txt = ex.text()
        pred = str(ex.meta.get("prediction", "")).strip().lower()
        # proxy: does any per-agent text contain a differing short token?
        per_agent = []
        for k, v in txt.items():
            if isinstance(v, str):
                per_agent.append(v.lower())
        if per_agent and pred:
            diverge = any(pred not in s and s for s in per_agent)
            gap += int(diverge); total += 1
    frac = gap / max(1, total)
    return finding(
        metrics=dict(faithfulness_gap_frac=frac, n=total),
        score=frac,
        msg=f"Proxy faithfulness-gap frequency {frac:.3f}."
    )


@experiment(23, "Bargaining / power asymmetries",
            "Does one agent dominate shared memory direction?")
def exp_023(ds: Dataset) -> dict:
    light = ds.light_aggregate()
    post = light["post"]
    if post.shape[1] < 2: return finding({}, 0, "need ≥2 agents", status="skipped")
    N, A, m, D = post.shape
    # per example, which agent's direction is closest to the centroid?
    wins = np.zeros(A)
    for n in range(N):
        ctr = post[n].reshape(-1, D).mean(0)
        sims = [cosine(post[n, a, -1], ctr) for a in range(A)]
        wins[int(np.argmax(sims))] += 1
    wins /= N
    dom = float(wins.max())
    return finding(
        metrics=dict(win_fraction_per_agent=wins.tolist(),
                     most_dominant=int(wins.argmax()),
                     dominance=dom),
        score=max(0.0, dom - 1 / A) * A / (A - 1),
        msg=f"Agent {int(wins.argmax())} dominates {dom*100:.1f}% of the time."
    )


@experiment(24, "Catastrophe precursors",
            "Entropy/norm changes before text failure")
def exp_024(ds: Dataset) -> dict:
    light = ds.light_aggregate()
    post = light["post"]; corr = light["correct"]
    if post.shape[2] < 2: return finding({}, 0, "need ≥2 rounds", status="skipped")
    # look at early-round norm variance as predictor of failure
    early_var = post[:, :, 0, :].var(axis=(1, 2))
    auc = auc_roc(early_var, (~corr).astype(int))
    return finding(
        metrics=dict(auc_var0_vs_failure=auc),
        score=abs(auc - 0.5) * 2,
        msg=f"Round-0 variance AUC vs failure={auc:.3f}."
    )


@experiment(77, "Groupthink detector",
            "Does early convergence predict wrong answers?")
def exp_077(ds: Dataset) -> dict:
    light = ds.light_aggregate()
    post = light["post"]; corr = light["correct"]
    if post.shape[1] < 2: return finding({}, 0, "need ≥2 agents", status="skipped")
    N, A, m, D = post.shape
    early_conv = []
    for n in range(N):
        vals = [cosine(post[n, i, 0], post[n, j, 0])
                for i in range(A) for j in range(i + 1, A)]
        early_conv.append(np.mean(vals))
    early_conv = np.array(early_conv)
    auc = auc_roc(early_conv, (~corr).astype(int))
    return finding(
        metrics=dict(auc_early_consensus_vs_failure=auc),
        score=abs(auc - 0.5) * 2,
        msg=f"Early-round consensus AUC vs failure = {auc:.3f} "
            f"(>0.5 = groupthink evidence)."
    )


@experiment(78, "Wisdom of crowds",
            "Centroid predicts better than individuals?")
def exp_078(ds: Dataset) -> dict:
    light = ds.light_aggregate()
    post = light["post"]; corr = light["correct"]
    if post.size == 0 or corr.sum() < 5 or (~corr).sum() < 5:
        return finding({}, 0, "insufficient", status="skipped")
    N, A, m, D = post.shape
    # class axis from centroids
    X = post.mean(axis=(1, 2))  # centroid
    ax = X[corr].mean(0) - X[~corr].mean(0)
    auc_c = auc_roc(X @ ax, corr.astype(int))
    aucs_i = []
    for a in range(A):
        Xa = post[:, a, :, :].mean(1)
        auc_a = auc_roc(Xa @ ax, corr.astype(int))
        aucs_i.append(auc_a)
    return finding(
        metrics=dict(auc_centroid=auc_c, auc_per_agent=aucs_i,
                     centroid_minus_best=float(auc_c - max(aucs_i))),
        score=max(0.0, auc_c - max(aucs_i)) * 4,
        msg=f"Centroid AUC {auc_c:.3f} vs best-agent AUC {max(aucs_i):.3f}."
    )


@experiment(79, "Social contagion of errors",
            "How fast do errors spread across agents?")
def exp_079(ds: Dataset) -> dict:
    # proxy: similarity between agent_0 first thought and last agent last
    # thought, split by correctness
    light = ds.light_aggregate()
    post = light["post"]; corr = light["correct"]
    if post.shape[1] < 2 or post.shape[2] < 2:
        return finding({}, 0, "insufficient", status="skipped")
    N, A, m, D = post.shape
    spread = np.array([cosine(post[n, 0, 0], post[n, -1, -1]) for n in range(N)])
    gap = float(spread[~corr].mean() - spread[corr].mean()) if corr.any() and (~corr).any() else 0.0
    return finding(
        metrics=dict(mean_spread_correct=float(spread[corr].mean()) if corr.any() else 0,
                     mean_spread_incorrect=float(spread[~corr].mean()) if (~corr).any() else 0,
                     contagion_gap=gap),
        score=min(1.0, abs(gap) * 4),
        msg=f"Error-contagion gap {gap:+.3f}."
    )


@experiment(80, "Status hierarchy & deference",
            "Whose direction gets preserved downstream?")
def exp_080(ds: Dataset) -> dict:
    light = ds.light_aggregate()
    post = light["post"]
    if post.shape[1] < 2: return finding({}, 0, "need ≥2 agents", status="skipped")
    N, A, m, D = post.shape
    preserve = np.zeros(A)
    for a in range(A):
        cs = []
        for n in range(N):
            for b in range(A):
                if b == a: continue
                cs.append(cosine(post[n, a, 0], post[n, b, -1]))
        preserve[a] = float(np.mean(cs))
    return finding(
        metrics=dict(deference_per_agent=preserve.tolist()),
        score=min(1.0, float(preserve.std()) * 4),
        msg=f"Deference std {preserve.std():.3f}."
    )


@experiment(81, "Minority opinion preservation",
            "Is outlier agent amplified or suppressed?")
def exp_081(ds: Dataset) -> dict:
    light = ds.light_aggregate()
    post = light["post"]; corr = light["correct"]
    if post.shape[1] < 3 or post.shape[2] < 2:
        return finding({}, 0, "insufficient", status="skipped")
    N, A, m, D = post.shape
    # outlier = agent whose first latent is farthest from others
    suppressed_right, suppressed_wrong = [], []
    for n in range(N):
        first = post[n, :, 0, :]
        ctr = first.mean(0)
        dists0 = np.linalg.norm(first - ctr, axis=1)
        outlier = int(dists0.argmax())
        last = post[n, :, -1, :]; ctrL = last.mean(0)
        dL = float(np.linalg.norm(last[outlier] - ctrL))
        d0 = float(dists0[outlier])
        suppression = (d0 - dL) / (d0 + 1e-9)  # >0 ⇒ minority moved toward crowd
        (suppressed_right if corr[n] else suppressed_wrong).append(suppression)
    gap = float(np.mean(suppressed_wrong) - np.mean(suppressed_right)) if suppressed_wrong and suppressed_right else 0.0
    return finding(
        metrics=dict(mean_suppression_correct=float(np.mean(suppressed_right)) if suppressed_right else 0,
                     mean_suppression_wrong=float(np.mean(suppressed_wrong)) if suppressed_wrong else 0,
                     gap=gap),
        score=min(1.0, abs(gap) * 4),
        msg=f"Minority-suppression gap (wrong−right) {gap:+.3f}."
    )


# ===========================================================================
# TRACK 9 — game theory
# ===========================================================================
@experiment(82, "Shapley-like agent attribution",
            "Marginal contribution of each agent to predicted correctness")
def exp_082(ds: Dataset) -> dict:
    light = ds.light_aggregate()
    post = light["post"]; corr = light["correct"]
    if post.shape[1] < 2: return finding({}, 0, "need ≥2 agents", status="skipped")
    N, A, m, D = post.shape
    from itertools import combinations
    # linear probe on subset-mean latent vs correctness, leave-one-agent-out
    X_full = post.mean(axis=(1, 2))
    ax_full = X_full[corr].mean(0) - X_full[~corr].mean(0)
    auc_full = auc_roc(X_full @ ax_full, corr.astype(int))
    contrib = {}
    for a in range(A):
        idxs = [i for i in range(A) if i != a]
        Xw = post[:, idxs, :, :].mean(axis=(1, 2))
        ax = Xw[corr].mean(0) - Xw[~corr].mean(0)
        auc_w = auc_roc(Xw @ ax, corr.astype(int))
        contrib[f"agent_{a}"] = float(auc_full - auc_w)
    return finding(
        metrics=dict(marginal_auc_loss=contrib, auc_full=float(auc_full)),
        score=min(1.0, max(abs(v) for v in contrib.values()) * 4),
        msg="Per-agent leave-one-out AUC drop: "
            + ", ".join(f"{k}={v:+.3f}" for k, v in contrib.items())
    )


@experiment(83, "Information asymmetry",
            "Does prompt-hidden entropy predict latent quality?")
def exp_083(ds: Dataset) -> dict:
    sample = ds.all_examples()[:120]
    xs, ys = [], []
    for ex in sample:
        try:
            _, post, agents = ex.latents()
            ph = ex.prompt()
            for i, a in enumerate(agents):
                if a not in ph: continue
                v = ph[a].var(axis=0)
                H = float(np.log(v[v > 1e-9]).sum())
                q = float(np.linalg.norm(post[i]))
                xs.append(H); ys.append(q)
        except Exception: pass
        ex.drop_heavy()
    if len(xs) < 10: return finding({}, 0, "too few", status="skipped")
    r = pearson(np.array(xs), np.array(ys))
    return finding(
        metrics=dict(pearson_promptH_latentNorm=r, n=len(xs)),
        score=min(1.0, abs(r) * 2),
        msg=f"Prompt-entropy ↔ latent-norm pearson {r:+.3f}."
    )


@experiment(84, "Diminishing returns to rounds",
            "Does accuracy-proxy saturate?")
def exp_084(ds: Dataset) -> dict:
    light = ds.light_aggregate()
    post = light["post"]; corr = light["correct"]
    if post.shape[2] < 2 or corr.sum() < 5 or (~corr).sum() < 5:
        return finding({}, 0, "insufficient", status="skipped")
    aucs = []
    for t in range(post.shape[2]):
        X = post[:, :, t, :].mean(axis=1)
        ax = X[corr].mean(0) - X[~corr].mean(0)
        aucs.append(auc_roc(X @ ax, corr.astype(int)))
    gains = np.diff(aucs)
    return finding(
        metrics=dict(auc_per_round=aucs,
                     marginal_gains=gains.tolist(),
                     saturation_round=int(np.argmin(np.abs(gains))) + 1),
        score=float(aucs[-1] - aucs[0]),
        msg=f"AUC across rounds: " + ", ".join(f"{v:.3f}" for v in aucs)
    )


@experiment(85, "Pareto efficiency of exchanges",
            "Does pairwise exchange improve both agents?")
def exp_085(ds: Dataset) -> dict:
    light = ds.light_aggregate()
    post = light["post"]
    if post.shape[1] < 2 or post.shape[2] < 2:
        return finding({}, 0, "insufficient", status="skipped")
    # improvement proxy: norm growth per agent
    growth = (np.linalg.norm(post[:, :, -1, :], axis=-1) -
              np.linalg.norm(post[:, :, 0, :], axis=-1)).mean(0)
    pareto_frac = float((growth > 0).mean())
    return finding(
        metrics=dict(per_agent_growth=growth.tolist(),
                     pareto_frac=pareto_frac),
        score=pareto_frac,
        msg=f"{pareto_frac*100:.1f}% of agents gain norm (Pareto-ish)."
    )


# ===========================================================================
# TRACK 10 — chemistry / materials
# ===========================================================================
@experiment(86, "Catalytic agents",
            "Which agent's presence lowers 'activation energy' for others?")
def exp_086(ds: Dataset) -> dict:
    light = ds.light_aggregate()
    post = light["post"]; corr = light["correct"]
    if post.shape[1] < 2: return finding({}, 0, "need ≥2 agents", status="skipped")
    N, A, m, D = post.shape
    cat = {}
    for a in range(A):
        norm_a = np.linalg.norm(post[:, a, 0, :], axis=-1)
        # correctness for examples with high a-norm
        hi = norm_a > np.median(norm_a)
        acc_hi = float(corr[hi].mean()); acc_lo = float(corr[~hi].mean())
        cat[f"agent_{a}"] = float(acc_hi - acc_lo)
    return finding(
        metrics=dict(catalytic_gain=cat),
        score=min(1.0, max(abs(v) for v in cat.values()) * 4),
        msg="Catalytic Δacc: " + ", ".join(f"{k}={v:+.3f}" for k, v in cat.items())
    )


@experiment(87, "Entropy of mixing",
            "Is combined memory more or less ordered than parts?")
def exp_087(ds: Dataset) -> dict:
    light = ds.light_aggregate()
    post = light["post"]
    if post.shape[1] < 2: return finding({}, 0, "need ≥2 agents", status="skipped")
    N, A, m, D = post.shape
    parts_H = []
    joint_H = []
    for n in range(N):
        for a in range(A):
            v = post[n, a].var(axis=0)
            parts_H.append(float(np.log(v[v > 1e-9]).sum()))
        vJ = post[n].reshape(-1, D).var(axis=0)
        joint_H.append(float(np.log(vJ[vJ > 1e-9]).sum()))
    excess = float(np.mean(joint_H) - np.mean(parts_H))
    return finding(
        metrics=dict(parts_H=float(np.mean(parts_H)),
                     joint_H=float(np.mean(joint_H)),
                     mixing_excess=excess),
        score=min(1.0, abs(excess) / 300),
        msg=f"Mixing excess-entropy {excess:+.1f}."
    )


@experiment(88, "Reaction kinetics",
            "Rate of concept transfer between agents")
def exp_088(ds: Dataset) -> dict:
    light = ds.light_aggregate()
    post = light["post"]
    if post.shape[1] < 2 or post.shape[2] < 2:
        return finding({}, 0, "insufficient", status="skipped")
    N, A, m, D = post.shape
    # similarity between agent0-first and agent1-round-t
    rates = []
    for n in range(N):
        ys = [cosine(post[n, 0, 0], post[n, 1, t]) for t in range(m)]
        rates.append(np.polyfit(range(m), ys, 1)[0])
    k = float(np.mean(rates))
    return finding(
        metrics=dict(mean_rate=k, std_rate=float(np.std(rates))),
        score=min(1.0, abs(k) * 4),
        msg=f"Mean transfer rate {k:+.3f} per round."
    )


@experiment(89, "Phase diagram (rounds × agents)",
            "Heatmap of accuracy-proxy over (#agents, #rounds)")
def exp_089(ds: Dataset) -> dict:
    light = ds.light_aggregate()
    post = light["post"]; corr = light["correct"]
    if corr.sum() < 5 or (~corr).sum() < 5:
        return finding({}, 0, "need both", status="skipped")
    N, A, m, D = post.shape
    grid = {}
    for k in range(1, A + 1):
        for t in range(1, m + 1):
            X = post[:, :k, :t, :].mean(axis=(1, 2))
            ax = X[corr].mean(0) - X[~corr].mean(0)
            grid[f"A{k}_m{t}"] = float(auc_roc(X @ ax, corr.astype(int)))
    best = max(grid.items(), key=lambda kv: kv[1])
    return finding(
        metrics=dict(auc_grid=grid, best_regime=best[0], best_auc=best[1]),
        score=float(best[1] - 0.5) * 2,
        msg=f"Best regime {best[0]} AUC={best[1]:.3f}."
    )


# ===========================================================================
# TRACK 11 — network science
# ===========================================================================
@experiment(90, "Transfer-entropy graph motifs",
            "Directed info-flow topology")
def exp_090(ds: Dataset) -> dict:
    light = ds.light_aggregate()
    post = light["post"]
    if post.shape[1] < 3: return finding({}, 0, "need ≥3 agents", status="skipped")
    N, A, m, D = post.shape
    # directed weight i->j: correlation between post[i,0] and post[j,-1]
    W = np.zeros((A, A))
    for i in range(A):
        for j in range(A):
            if i == j: continue
            W[i, j] = pearson(post[:, i, 0, :].ravel(), post[:, j, -1, :].ravel())
    return finding(
        metrics=dict(flow_matrix=W.tolist(),
                     asymmetry=float(np.abs(W - W.T).mean())),
        score=min(1.0, float(np.abs(W - W.T).mean()) * 4),
        msg=f"Mean asymmetry of info flow {np.abs(W - W.T).mean():.3f}."
    )


@experiment(91, "Critical agents (node removal)",
            "Whose removal kills the predictive signal most?")
def exp_091(ds: Dataset) -> dict:
    light = ds.light_aggregate()
    post = light["post"]; corr = light["correct"]
    if post.shape[1] < 2 or corr.sum() < 5 or (~corr).sum() < 5:
        return finding({}, 0, "insufficient", status="skipped")
    N, A, m, D = post.shape
    auc_full = auc_roc(post.mean(axis=(1, 2)) @ (post[corr].mean((0, 1, 2)) - post[~corr].mean((0, 1, 2))), corr.astype(int))
    drop = {}
    for a in range(A):
        keep = [i for i in range(A) if i != a]
        X = post[:, keep, :, :].mean(axis=(1, 2))
        ax = X[corr].mean(0) - X[~corr].mean(0)
        drop[f"agent_{a}"] = float(auc_full - auc_roc(X @ ax, corr.astype(int)))
    return finding(
        metrics=dict(per_agent_auc_loss=drop, auc_full=float(auc_full)),
        score=min(1.0, max(abs(v) for v in drop.values()) * 4),
        msg="AUC drop if removed: " + ", ".join(f"{k}={v:+.3f}" for k, v in drop.items())
    )


@experiment(92, "Power law in info flow",
            "Scale-free distribution of edge weights?")
def exp_092(ds: Dataset) -> dict:
    light = ds.light_aggregate()
    post = light["post"]
    if post.shape[1] < 3: return finding({}, 0, "need ≥3 agents", status="skipped")
    N, A, m, D = post.shape
    edges = []
    for i in range(A):
        for j in range(A):
            if i == j: continue
            edges.append(abs(pearson(post[:, i, 0, :].ravel(), post[:, j, -1, :].ravel())))
    edges = np.sort(np.array(edges))[::-1]
    ranks = np.arange(1, len(edges) + 1)
    # fit log-log slope
    slope = float(np.polyfit(np.log(ranks), np.log(edges + 1e-9), 1)[0])
    return finding(
        metrics=dict(edge_weights_desc=edges.tolist(),
                     log_log_slope=slope),
        score=min(1.0, abs(slope) / 2),
        msg=f"Edge-weight rank-frequency slope {slope:.2f} (~-1 for power law)."
    )


@experiment(93, "Clustering coefficient",
            "Triangle density in info-flow graph")
def exp_093(ds: Dataset) -> dict:
    light = ds.light_aggregate()
    post = light["post"]
    if post.shape[1] < 3: return finding({}, 0, "need ≥3 agents", status="skipped")
    N, A, m, D = post.shape
    W = np.zeros((A, A))
    for i in range(A):
        for j in range(A):
            if i == j: continue
            W[i, j] = abs(pearson(post[:, i, 0, :].ravel(), post[:, j, -1, :].ravel()))
    mean = W[W > 0].mean()
    B = (W > mean).astype(int)
    # triangle count
    tri = int(np.trace(B @ B @ B))
    possible = A * (A - 1) * (A - 2) / 6
    cc = tri / max(1, possible)
    return finding(
        metrics=dict(triangles=tri, clustering_coef=float(cc)),
        score=float(cc),
        msg=f"Clustering coefficient {cc:.3f}."
    )


# ===========================================================================
# TRACK 12 — linguistics
# ===========================================================================
@experiment(94, "Zipf law in latent patterns",
            "Do cluster frequencies follow 1/rank?")
def exp_094(ds: Dataset) -> dict:
    light = ds.light_aggregate()
    post = light["post"]
    if post.size == 0: return finding({}, 0, "no data", status="skipped")
    N, A, m, D = post.shape
    X = post.reshape(-1, D)
    k = 128
    rng = np.random.default_rng(0)
    idx = rng.choice(len(X), k, replace=False)
    cent = X[idx].copy()
    x_sq = (X ** 2).sum(1)
    for _ in range(8):
        c_sq = (cent ** 2).sum(1)
        # ||x-c||^2 = x^2 + c^2 - 2 x·c
        d2 = x_sq[:, None] + c_sq[None, :] - 2 * (X @ cent.T)
        a = d2.argmin(1)
        for c in range(k):
            if (a == c).any(): cent[c] = X[a == c].mean(0)
    sizes = np.bincount(a, minlength=k).astype(float)
    sizes = np.sort(sizes)[::-1]
    sizes = sizes[sizes > 0]
    ranks = np.arange(1, len(sizes) + 1)
    slope = float(np.polyfit(np.log(ranks), np.log(sizes), 1)[0])
    return finding(
        metrics=dict(log_log_slope=slope, n_clusters=int(len(sizes))),
        score=min(1.0, 1 - abs(abs(slope) - 1)),
        msg=f"Frequency-rank slope {slope:.2f} (Zipf slope ≈ -1)."
    )


@experiment(95, "Semantic drift across tasks",
            "Same concept representation drifts by task?")
def exp_095(ds: Dataset) -> dict:
    light = ds.light_aggregate()
    post = light["post"]; tasks = light["tasks"]
    if len(set(tasks)) < 2: return finding({}, 0, "need ≥2 tasks", status="skipped")
    # centroid per task
    cents = {t: post[tasks == t].mean(axis=(0, 1, 2)) for t in set(tasks)}
    ts = list(cents)
    drifts = [float(np.linalg.norm(cents[ts[i]] - cents[ts[j]]))
              for i in range(len(ts)) for j in range(i + 1, len(ts))]
    return finding(
        metrics=dict(per_task_drift=drifts, tasks=ts),
        score=min(1.0, float(np.mean(drifts)) / 10),
        msg=f"Pairwise task-centroid distance mean {np.mean(drifts):.3f}."
    )


@experiment(96, "Compositionality",
            "Do multi-step thoughts = sums of simpler primitives?")
def exp_096(ds: Dataset) -> dict:
    light = ds.light_aggregate()
    post = light["post"]
    if post.shape[2] < 2: return finding({}, 0, "need ≥2 steps", status="skipped")
    # check if avg(step0 + step1) ≈ step_last (linearity of composition)
    residual = []
    N, A, m, D = post.shape
    for n in range(N):
        for a in range(A):
            pred = post[n, a, :m - 1].mean(0)
            act = post[n, a, -1]
            residual.append(cosine(pred, act))
    r = float(np.mean(residual))
    return finding(
        metrics=dict(mean_cos_pred_vs_last=r),
        score=max(0.0, r),
        msg=f"Composed-mean↔last cosine {r:.3f}."
    )


@experiment(97, "Code-switching by task",
            "Are task-dialect subspaces separable?")
def exp_097(ds: Dataset) -> dict:
    return exp_037(ds)  # same underlying measurement


# ===========================================================================
# TRACK 13 — safety / alignment
# ===========================================================================
@experiment(98, "Latent hallucination signature",
            "Geometric signature of hallucinations")
def exp_098(ds: Dataset) -> dict:
    light = ds.light_aggregate()
    post = light["post"]; corr = light["correct"]
    if corr.sum() < 5 or (~corr).sum() < 5:
        return finding({}, 0, "need both", status="skipped")
    # compare norm-variance distributions
    var = post.var(axis=(1, 2)).mean(axis=-1)
    auc = auc_roc(var, (~corr).astype(int))
    return finding(
        metrics=dict(auc_var_vs_wrong=auc),
        score=abs(auc - 0.5) * 2,
        msg=f"Latent variance AUC vs hallucination proxy {auc:.3f}."
    )


@experiment(99, "Agent-level responsibility",
            "Which agent/round introduces the error?")
def exp_099(ds: Dataset) -> dict:
    light = ds.light_aggregate()
    post = light["post"]; corr = light["correct"]
    if (~corr).sum() < 5: return finding({}, 0, "too few failures", status="skipped")
    N, A, m, D = post.shape
    # axis = mean correct - mean incorrect for round-last
    X_last = post[:, :, -1, :].mean(axis=1)
    ax = X_last[corr].mean(0) - X_last[~corr].mean(0)
    ax = ax / (np.linalg.norm(ax) + 1e-9)
    # per (agent,round), projection gap between correct and incorrect
    gaps = np.zeros((A, m))
    for a in range(A):
        for t in range(m):
            proj = post[:, a, t, :] @ ax
            gaps[a, t] = float(proj[corr].mean() - proj[~corr].mean())
    loc = np.unravel_index(np.argmax(np.abs(gaps)), gaps.shape)
    return finding(
        metrics=dict(gap_matrix=gaps.tolist(),
                     blame_agent=int(loc[0]), blame_round=int(loc[1]),
                     max_abs_gap=float(np.abs(gaps).max())),
        score=min(1.0, float(np.abs(gaps).max()) * 2),
        msg=f"Largest correct-vs-wrong projection gap at agent "
            f"{int(loc[0])} round {int(loc[1])}."
    )


@experiment(100, "Sycophancy in latent space",
            "Do later agents align with dominant agent when wrong?")
def exp_100(ds: Dataset) -> dict:
    light = ds.light_aggregate()
    post = light["post"]; corr = light["correct"]
    if post.shape[1] < 3: return finding({}, 0, "need ≥3 agents", status="skipped")
    N, A, m, D = post.shape
    sycop = []
    for n in range(N):
        # dominant = agent with largest first-round norm
        norms = np.linalg.norm(post[n, :, 0, :], axis=-1)
        dom = int(norms.argmax())
        others = [i for i in range(A) if i != dom]
        sycop.append(float(np.mean([cosine(post[n, dom, 0], post[n, j, -1]) for j in others])))
    sycop = np.array(sycop)
    gap = float(sycop[~corr].mean() - sycop[corr].mean()) if corr.any() and (~corr).any() else 0.0
    return finding(
        metrics=dict(sycophancy_gap=gap,
                     mean_correct=float(sycop[corr].mean()) if corr.any() else 0.0,
                     mean_wrong=float(sycop[~corr].mean()) if (~corr).any() else 0.0),
        score=min(1.0, abs(gap) * 4),
        msg=f"Sycophancy gap (wrong−right) = {gap:+.3f}."
    )


# ===========================================================================
# TRACK 14 — wild / high-upside
# ===========================================================================
@experiment(101, "Latent Turing test",
            "Can a classifier identify which agent produced a thought?")
def exp_101(ds: Dataset) -> dict:
    light = ds.light_aggregate()
    post = light["post"]
    if post.shape[1] < 2: return finding({}, 0, "need ≥2 agents", status="skipped")
    N, A, m, D = post.shape
    X = post.reshape(N * A * m, D)
    y = np.repeat(np.repeat(np.arange(A)[None, :], N, axis=0)[..., None], m, axis=2).ravel()
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(X))
    split = int(0.7 * len(X))
    tr, te = perm[:split], perm[split:]
    cents = np.stack([X[tr][y[tr] == a].mean(0) for a in range(A)])
    d = np.linalg.norm(X[te][:, None, :] - cents[None, :, :], axis=-1)
    pred = d.argmin(1)
    acc = float((pred == y[te]).mean())
    return finding(
        metrics=dict(accuracy=acc, chance=1 / A),
        score=max(0.0, (acc - 1 / A) / (1 - 1 / A)),
        msg=f"Agent-id accuracy {acc:.3f} (chance {1/A:.3f})."
    )


@experiment(102, "Cocktail party / selective attention",
            "Which dims does each agent pull from shared memory?")
def exp_102(ds: Dataset) -> dict:
    light = ds.light_aggregate()
    post = light["post"]
    if post.shape[1] < 2: return finding({}, 0, "need ≥2 agents", status="skipped")
    N, A, m, D = post.shape
    pref = []
    for a in range(A):
        v = post[:, a, :, :].reshape(-1, D).var(axis=0)
        pref.append(np.argsort(-v)[:10].tolist())
    # overlap between agents' top dims
    overlap = float(np.mean([
        len(set(pref[i]) & set(pref[j])) / 10
        for i in range(A) for j in range(i + 1, A)
    ])) if A > 1 else 0.0
    return finding(
        metrics=dict(top10_dims_per_agent=pref, mean_overlap=overlap),
        score=1.0 - overlap,
        msg=f"Mean top-10 dim overlap across agents {overlap:.3f} (low = selective)."
    )


@experiment(103, "Turing-completeness / expressiveness",
            "Evidence of recursive / compositional structure")
def exp_103(ds: Dataset) -> dict:
    # crude proxy: if post-latent i is approximable by linear combination of
    # post-latents < i, then the channel is at least linear-expressive
    light = ds.light_aggregate()
    post = light["post"]
    if post.shape[2] < 3: return finding({}, 0, "need ≥3 steps", status="skipped")
    N, A, m, D = post.shape
    resid = []
    for n in range(N):
        for a in range(A):
            X = post[n, a]
            # predict last from preceding (least squares)
            A_ = X[:-1]; b_ = X[-1]
            w, *_ = np.linalg.lstsq(A_.T @ A_ + 1e-3 * np.eye(A_.shape[0]),
                                    A_ @ b_, rcond=None)
            pred = w @ A_
            resid.append(float(np.linalg.norm(pred - b_) / (np.linalg.norm(b_) + 1e-9)))
    r = float(np.mean(resid))
    return finding(
        metrics=dict(mean_residual=r),
        score=max(0.0, 1 - r),
        msg=f"Residual (lower=more compositional) {r:.3f}."
    )


@experiment(105, "Minority report — dissent under consensus",
            "When all agents agree but answer is wrong, is there a hidden dissent axis?")
def exp_105(ds: Dataset) -> dict:
    light = ds.light_aggregate()
    post = light["post"]; corr = light["correct"]
    if post.shape[1] < 2: return finding({}, 0, "need ≥2 agents", status="skipped")
    N, A, m, D = post.shape
    # isolate "high consensus" examples
    cos_last = np.array([
        np.mean([cosine(post[n, i, -1], post[n, j, -1])
                 for i in range(A) for j in range(i + 1, A)])
        for n in range(N)])
    hi = cos_last > np.percentile(cos_last, 75)
    if (hi & ~corr).sum() < 5 or (hi & corr).sum() < 5:
        return finding({}, 0, "too few", status="skipped")
    # find axis separating hi-confidence-wrong vs hi-confidence-right
    Xw = post[hi & ~corr].mean(axis=(1, 2))
    Xr = post[hi & corr].mean(axis=(1, 2))
    ax = Xr.mean(0) - Xw.mean(0)
    s = np.linalg.norm(ax)
    return finding(
        metrics=dict(axis_norm=float(s),
                     n_hi_wrong=int((hi & ~corr).sum()),
                     n_hi_right=int((hi & corr).sum())),
        score=min(1.0, float(s) / 10),
        msg=f"Hidden-dissent axis norm {s:.3f}."
    )


# ---------------------------------------------------------------------------
# runner
# ---------------------------------------------------------------------------
def run(args):
    root = Path(args.activations)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    ds = Dataset(root=root, tasks=args.tasks)
    ds.load(max_examples=args.max_examples)
    print(f"Loaded {sum(len(v) for v in ds.examples.values())} examples across "
          f"{len(ds.tasks)} tasks.")

    ids = sorted(Registry.keys())
    if args.only:
        only = set(args.only)
        ids = [i for i in ids if i in only]

    results = []
    for i in ids:
        name, question, fn = Registry[i]
        out_path = out / f"exp_{i:03d}.json"
        if out_path.exists() and not args.overwrite:
            with open(out_path) as f:
                r = json.load(f)
            results.append(r)
            print(f"  [skip] exp_{i:03d} ({name}) — cached")
            continue
        print(f"  [run ] exp_{i:03d} ({name}) …", flush=True)
        try:
            res = fn(ds)
            status = res.get("status", "ok")
        except Exception as e:
            traceback.print_exc()
            res = dict(metrics={}, score=0.0, finding=f"error: {e}",
                       notes="", status="error")
            status = "error"
        record = dict(id=i, name=name, question=question, **res)
        with open(out_path, "w") as f:
            json.dump(record, f, indent=2, default=str)
        results.append(record)
        print(f"    ↳ status={status} score={record.get('score',0):.3f}  "
              f"{record.get('finding','')[:120]}")

    results.sort(key=lambda r: -r.get("score", 0))
    summary = dict(
        n_experiments=len(results),
        n_ok=sum(1 for r in results if r.get("status") == "ok"),
        n_skipped=sum(1 for r in results if r.get("status") == "skipped"),
        n_error=sum(1 for r in results if r.get("status") == "error"),
        ranked=[{
            "id": r["id"], "name": r["name"],
            "score": r.get("score", 0),
            "status": r.get("status"),
            "finding": r.get("finding", "")} for r in results],
    )
    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nwrote {out / 'summary.json'}. "
          f"{summary['n_ok']} ok / {summary['n_skipped']} skipped / "
          f"{summary['n_error']} errored.")
    print("\nTop 10 by score:")
    for r in summary["ranked"][:10]:
        print(f"  exp_{r['id']:03d}  score={r['score']:.3f}  "
              f"{r['name']}  —  {r['finding']}")


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--activations", default="./activations")
    p.add_argument("--out", default="./results")
    p.add_argument("--tasks", nargs="+",
                   default=["gsm8k", "arc_challenge", "mbppplus"])
    p.add_argument("--max-examples", type=int, default=150,
                   help="per task; 0 = use all")
    p.add_argument("--only", type=int, nargs="*", default=None,
                   help="run only these experiment IDs")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    run(parse())
