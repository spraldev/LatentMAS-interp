"""Exp A — W_a mechanism.

A1. SVD spectrum: σ-distribution, fraction within 5% of 1.0, condition number
A2. Identity ablation (post-hoc): cos(W_a h, h) on collected post-W_a hiddens
A3. Fixed-point analysis: top-10 unit-eigenvalue directions, projection by correctness
A4. Layer analogue: CKA(W_a row-space, layer weights) — skipped if model not loadable
A5. Manifold alignment: Spearman r between pairwise cos similarity matrices
    in agent A's space (pre-W_a) vs agent B's space (post-W_a)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy import stats

from analysis import common
from analysis.stats import bootstrap_ci, permutation_test_diff


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--activations_dir", required=True, type=Path)
    p.add_argument("--max_examples", type=int, default=300,
                   help="Per task; A2/A3/A5 use latent_mas activations.")
    args = p.parse_args()

    common.setup_logging()
    out = common.results_dir(args.activations_dir, "exp_a")
    results = {}

    wa = common.load_wa(args.activations_dir)
    if wa is None:
        print("[exp_a] wa_matrix.pt missing; aborting")
        return
    W = wa["W_a"].to(torch.float32).cpu().numpy()
    D = W.shape[0]

    # A1: SVD spectrum
    U, S, Vh = np.linalg.svd(W, full_matrices=False)
    frac_near_1 = float(np.mean(np.abs(S - 1.0) < 0.05))
    top10_var = float(np.sum(S[:10] ** 2) / np.sum(S ** 2))
    cond_num = float(S.max() / max(S.min(), 1e-12))
    frob_from_I = float(np.linalg.norm(W - np.eye(D)))
    results["A1"] = {
        "D": D,
        "sv_min": float(S.min()), "sv_max": float(S.max()),
        "sv_mean": float(S.mean()),
        "frac_sv_near_1": frac_near_1,
        "top10_variance_fraction": top10_var,
        "condition_number": cond_num,
        "frobenius_from_identity": frob_from_I,
        "is_identity_like": (cond_num < 1.1 and frob_from_I < 1.0),
    }
    np.save(out / "wa_singular_values.npy", S)

    # A2: identity-ablation cosine on collected pre-aligned hiddens
    cos_results = []
    for task in common.TASKS:
        X, _ = common.stack_pre_aligned(args.activations_dir, "latent_mas", task,
                                        limit=args.max_examples)
        if X.size == 0:
            continue
        # X: [N, A, M, D]; flatten over (N, A, M)
        H = X.reshape(-1, D)
        Wt = torch.from_numpy(W)
        Ht = torch.from_numpy(H)
        Wh = (Ht @ Wt.T).numpy()  # [N*, D]
        cos = (np.sum(H * Wh, axis=1) /
               (np.linalg.norm(H, axis=1) * np.linalg.norm(Wh, axis=1) + 1e-12))
        m, lo, hi = bootstrap_ci(cos.tolist())
        cos_results.append({"task": task, "n": int(len(cos)),
                            "mean_cos": m, "ci_lo": lo, "ci_hi": hi})
    results["A2"] = cos_results

    # A3: fixed-point analysis (eigvals nearest to 1)
    eigvals, eigvecs = np.linalg.eig(W)
    dists = np.abs(eigvals - 1.0)
    order = np.argsort(dists)
    top10_dirs = eigvecs[:, order[:10]].real    # [D, 10]
    top10_eig = eigvals[order[:10]]
    results["A3_eigvals_near_1"] = [complex(e).__repr__() for e in top10_eig.tolist()]

    # project all post-aligned by correctness
    a3_per_task = []
    for task in common.TASKS:
        Xc, ic = common.stack_post_aligned(args.activations_dir, "latent_mas", task,
                                            limit=args.max_examples)
        if Xc.size == 0:
            continue
        H = Xc.reshape(Xc.shape[0], -1, D).mean(axis=1)  # avg over (A, M) → [N, D]
        proj = np.linalg.norm(H @ top10_dirs, axis=1)
        correct_mask = np.array([info["correct"] for info in ic])
        if correct_mask.sum() < 10 or (~correct_mask).sum() < 10:
            continue
        perm = permutation_test_diff(proj[correct_mask].tolist(),
                                     proj[~correct_mask].tolist())
        a3_per_task.append({
            "task": task,
            "n_correct": int(correct_mask.sum()),
            "n_incorrect": int((~correct_mask).sum()),
            "mean_proj_correct": float(proj[correct_mask].mean()),
            "mean_proj_incorrect": float(proj[~correct_mask].mean()),
            **perm,
        })
    results["A3"] = a3_per_task

    # A5: manifold alignment (Spearman of pairwise cos matrices, pre vs post)
    a5_results = []
    for task in common.TASKS:
        Xpre, _ = common.stack_pre_aligned(args.activations_dir, "latent_mas", task,
                                           limit=args.max_examples)
        Xpost, _ = common.stack_post_aligned(args.activations_dir, "latent_mas", task,
                                              limit=args.max_examples)
        if Xpre.size == 0 or Xpost.size == 0:
            continue
        # use round 0, agent 0 → [N, D]; pairwise cos sim
        Hpre = Xpre[:, 0, 0, :]
        Hpost = Xpost[:, 0, 0, :]
        Hpre /= (np.linalg.norm(Hpre, axis=1, keepdims=True) + 1e-12)
        Hpost /= (np.linalg.norm(Hpost, axis=1, keepdims=True) + 1e-12)
        Mpre = Hpre @ Hpre.T
        Mpost = Hpost @ Hpost.T
        iu = np.triu_indices(Mpre.shape[0], k=1)
        r, p = stats.spearmanr(Mpre[iu], Mpost[iu])
        a5_results.append({"task": task, "n": int(Mpre.shape[0]),
                           "spearman_r": float(r), "p_value": float(p)})
    results["A5"] = a5_results

    (out / "exp_a.json").write_text(json.dumps(results, indent=2))
    print(f"[exp_a] wrote {out/'exp_a.json'}")
    print(f"  W_a is identity-like: {results['A1']['is_identity_like']}")
    print(f"  cond_num={results['A1']['condition_number']:.3f} "
          f"frob_from_I={results['A1']['frobenius_from_identity']:.3f}")


if __name__ == "__main__":
    main()
