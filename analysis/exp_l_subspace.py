"""Exp L — communication subspace.

L1. Identify subspace via two methods:
    A) PCA on residualized post-W_a hiddens (residualize agent_id + round_id)
       Pool ALL discovery-split examples (not just Bucket 1).
    B) L1-regularized logistic regression to predict Bucket-1 membership.
    Compare cosine similarity between top-k bases.
L2/L3. (sufficiency / necessity) NOTE: these require running the model
       with subspace projection. final_run.py topk_gated handles L2;
       L3 (orthogonal complement) would need a new condition. We aggregate
       topk_gated results here vs vanilla LMAS for the L2 sweep.
L4. Random k-d subspace control vs PCA-k.
L5. Dead dimensions (variance < 0.01 × max).
L6. Dimensionality vs TextMAS.

This script writes the subspace basis to results/exp_l/basis.pt for use
by Exp P / confidence_gated. It also writes the elbow-k value to
elbow_k.json so other scripts can reuse it.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

from analysis import common
from analysis.stats import bootstrap_ci, mcnemar


def _residualize(X: np.ndarray, agent_idx: np.ndarray, round_idx: np.ndarray) -> np.ndarray:
    """Subtract per-(agent, round) mean. X: [N, D]. agent_idx/round_idx: [N]."""
    out = X.copy()
    pairs = set(zip(agent_idx.tolist(), round_idx.tolist()))
    for a, r in pairs:
        m = (agent_idx == a) & (round_idx == r)
        if m.sum() > 1:
            out[m] -= X[m].mean(axis=0, keepdims=True)
    return out


def _elbow(variance: np.ndarray, threshold: float = 0.90) -> int:
    cum = np.cumsum(variance) / np.sum(variance)
    return int(np.searchsorted(cum, threshold) + 1)


def _build_pool(root: Path, condition: str, *, split: str = "discovery"):
    """Pool all post-aligned vectors from all tasks/all (agent, round).
    Falls back to all examples if the requested split is empty/unlocked."""
    H_list, agent_list, round_list, ex_id_list, task_list, correct_list = [], [], [], [], [], []
    for task in common.TASKS:
        X, ifo = common.stack_post_aligned(root, condition, task, split=split)
        if X.size == 0:
            # split not locked or empty — fall back to all examples
            X, ifo = common.stack_post_aligned(root, condition, task)
        if X.size == 0:
            continue
        N, A, M, D = X.shape
        for n in range(N):
            for a in range(A):
                for m in range(M):
                    H_list.append(X[n, a, m])
                    agent_list.append(a); round_list.append(m)
                    ex_id_list.append(ifo[n]["example_id"])
                    task_list.append(task)
                    correct_list.append(int(ifo[n]["correct"]))
    if not H_list:
        return None
    return {
        "H": np.stack(H_list, axis=0),
        "agent": np.array(agent_list), "round": np.array(round_list),
        "ex_id": np.array(ex_id_list), "task": np.array(task_list),
        "correct": np.array(correct_list),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--activations_dir", required=True, type=Path)
    p.add_argument("--energy_threshold", type=float, default=0.90)
    args = p.parse_args()
    common.setup_logging()
    out = common.results_dir(args.activations_dir, "exp_l")

    pool = _build_pool(args.activations_dir, "latent_mas", split="discovery")
    if pool is None:
        print("[exp_l] no LMAS discovery-split data — aborting")
        return
    H, agent_idx, round_idx = pool["H"], pool["agent"], pool["round"]
    H_res = _residualize(H, agent_idx, round_idx)
    D = H.shape[1]

    # Method A: PCA on residualized
    n_comp = min(D, H_res.shape[0] - 1, 512)
    pca = PCA(n_components=n_comp).fit(H_res)
    expvar = pca.explained_variance_ratio_
    elbow_k = _elbow(expvar, threshold=args.energy_threshold)
    basis_A = pca.components_[:elbow_k].T  # [D, k]

    # Method B: L1 logistic — Bucket-1 membership
    ex_ids = pool["ex_id"]
    correct = pool["correct"]
    # we need Bucket labels; load buckets and create per-vector label
    bucket1_labels = np.zeros(len(ex_ids), dtype=int)
    for task in common.TASKS:
        b = common.load_buckets(str(args.activations_dir), task)
        m_task = (pool["task"] == task)
        for i in np.where(m_task)[0]:
            bucket1_labels[i] = int(b.get(int(ex_ids[i]), 0) == 1)
    method_B_results = {}
    if bucket1_labels.sum() > 50:
        # use mean-pooled per-example to reduce dim
        # group by (task, ex_id) → average vector
        rows = {}
        for i in range(len(ex_ids)):
            key = (pool["task"][i], int(ex_ids[i]))
            rows.setdefault(key, []).append(H_res[i])
        keys = list(rows.keys())
        Hper = np.array([np.mean(rows[k], axis=0) for k in keys])
        ylabel = np.array([
            int(common.load_buckets(str(args.activations_dir), k[0]).get(int(k[1]), 0) == 1)
            for k in keys
        ])
        if ylabel.sum() >= 20:
            try:
                clf = LogisticRegression(C=0.1, penalty="l1", solver="liblinear",
                                         max_iter=2000).fit(Hper, ylabel)
                coefs = clf.coef_[0]
                top_dims = np.argsort(-np.abs(coefs))[:elbow_k]
                # build a basis from selected coordinates (one-hot)
                basis_B = np.zeros((D, len(top_dims)))
                for i, dim in enumerate(top_dims):
                    basis_B[dim, i] = 1.0
                # cosine sim between basis A and B (subspace overlap via Gram matrix)
                G = basis_A.T @ basis_B  # [k, k]
                _, sv, _ = np.linalg.svd(G, full_matrices=False)
                method_B_results = {
                    "n_selected_dims": int(np.sum(np.abs(coefs) > 0)),
                    "subspace_overlap_mean_singular_value": float(np.mean(sv)),
                    "coef_l1_norm": float(np.abs(coefs).sum()),
                }
            except Exception as e:
                method_B_results = {"error": str(e)}

    # L4: random k-d subspace control (compute alignment overlap)
    rng = np.random.default_rng(42)
    R = rng.normal(size=(D, elbow_k))
    R, _ = np.linalg.qr(R)
    G_rand = basis_A.T @ R
    _, sv_rand, _ = np.linalg.svd(G_rand, full_matrices=False)

    # L5: dead dimensions
    var_per_dim = H_res.var(axis=0)
    dead_frac = float((var_per_dim < 0.01 * var_per_dim.max()).mean())

    # L6: dimensionality vs TextMAS (PCA on TextMAS post-W_a if present)
    text_pool = _build_pool(args.activations_dir, "text_mas", split="discovery")
    text_dim = None
    if text_pool is not None:
        try:
            t_pca = PCA(n_components=min(text_pool["H"].shape[1] - 1, 256)).fit(text_pool["H"])
            text_dim = int(_elbow(t_pca.explained_variance_ratio_, args.energy_threshold))
        except Exception:
            text_dim = None

    # Save artifacts for Exp P + confidence_gated
    torch.save({"basis": torch.from_numpy(basis_A).float(),
                "k": int(elbow_k),
                "energy_threshold": args.energy_threshold},
               out / "basis.pt")

    # L2: aggregate topk_gated vs latent_mas accuracy
    l2_per_task = []
    for task in common.TASKS:
        l_meta = {ex.idx: ex.meta for ex in common.iter_examples(
            args.activations_dir, "latent_mas", task)}
        g_meta = {ex.idx: ex.meta for ex in common.iter_examples(
            args.activations_dir, "topk_gated", task)}
        shared = sorted(set(l_meta) & set(g_meta))
        if not shared:
            continue
        a = [bool(l_meta[i].get("correct")) for i in shared]
        b = [bool(g_meta[i].get("correct")) for i in shared]
        l2_per_task.append({
            "task": task,
            "n_paired": len(shared),
            "lmas_acc": float(np.mean(a)),
            "topk_gated_acc": float(np.mean(b)),
            "mcnemar": mcnemar(a, b),
        })

    results = {
        "elbow_k": int(elbow_k),
        "D": D,
        "n_discovery_vectors": int(H.shape[0]),
        "method_A_pca_explained_var_to_k": float(np.cumsum(expvar)[elbow_k - 1]),
        "method_A_top10_var": expvar[:10].tolist(),
        "method_B_predictive": method_B_results,
        "L4_random_subspace_overlap_mean_sv": float(np.mean(sv_rand)),
        "L5_dead_dim_fraction": dead_frac,
        "L6_text_mas_elbow_k": text_dim,
        "L2_topk_gated_vs_lmas": l2_per_task,
    }
    (out / "exp_l.json").write_text(json.dumps(results, indent=2))
    np.save(out / "expvar_curve.npy", expvar)
    print(f"[exp_l] elbow_k={elbow_k} (energy={args.energy_threshold}); "
          f"basis written to {out/'basis.pt'}")


if __name__ == "__main__":
    main()
