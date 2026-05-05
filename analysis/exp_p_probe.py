"""Exp P — predictive probe for Bucket-1 membership.

P1. Feature: round-1 post-W_a hidden of agent 0, projected onto top-k subspace
    from Exp L (basis at results/exp_l/basis.pt). If basis missing, falls
    back to W_a SVD top-k.
P2. Logistic regression with 5-fold CV; AUC, AUPRC, calibration.
P3. Dumb baselines:
    (a) full hidden state
    (b) random k-d projection
    (c) single-agent top-k features
    (d) question length
    (e) single-agent top-1 answer probability (proxy: prediction != "")
    (f) task ID one-hot
P5. Cross-task generalization: train on (gsm8k+arc), test on mbppplus.

Saves probe to <activations_dir>/exp_p/probe.joblib and basis (if regenerated)
to <activations_dir>/exp_p/basis.pt — these are picked up by
final_run.py's confidence_gated condition.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold

from analysis import common
from analysis.stats import auc_with_ci, mcnemar


def _load_basis(activations_dir: Path) -> torch.Tensor:
    p = activations_dir / "results" / "exp_l" / "basis.pt"
    if p.exists():
        d = torch.load(p, map_location="cpu", weights_only=False)
        return d["basis"]
    # fallback: W_a SVD top-k (k=64)
    wa = common.load_wa(activations_dir)
    if wa is None:
        raise FileNotFoundError("Neither exp_l basis nor W_a available")
    W = wa["W_a"].to(torch.float32).cpu().numpy()
    _, S, Vh = np.linalg.svd(W, full_matrices=False)
    cum = np.cumsum(S ** 2) / np.sum(S ** 2)
    k = int(np.searchsorted(cum, 0.90)) + 1
    return torch.from_numpy(Vh[:k].T).float()


def _build_features(root: Path, basis: np.ndarray, condition: str = "latent_mas"):
    """Returns X (N, k), y (Bucket-1), task_idx, ex_id, full_H, q_lens."""
    X_list, y_list, task_list, ex_ids, full_H, qlens = [], [], [], [], [], []
    for ti, task in enumerate(common.TASKS):
        b = common.load_buckets(str(root), task)
        for ex in common.iter_examples(root, condition, task):
            lt = common.load_latent_thoughts(ex.dir)
            if lt is None:
                continue
            post = lt["post_aligned"][0, 0].numpy()  # agent 0, round 0
            X_list.append(post @ basis)
            full_H.append(post)
            label = int(b.get(ex.idx, 0) == 1)
            y_list.append(label)
            task_list.append(ti)
            ex_ids.append(ex.idx)
            qlens.append(len(ex.meta.get("question", "").split()))
    if not X_list:
        return None
    return {
        "X": np.array(X_list), "y": np.array(y_list),
        "task": np.array(task_list), "ex_id": np.array(ex_ids),
        "full_H": np.array(full_H), "qlens": np.array(qlens),
    }


def _eval_probe(X, y, n_splits: int = 5):
    if y.sum() < 10 or (~y.astype(bool)).sum() < 10:
        return None
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = np.zeros_like(y, dtype=float)
    for tr, te in skf.split(X, y):
        clf = LogisticRegression(C=1.0, max_iter=2000).fit(X[tr], y[tr])
        scores[te] = clf.predict_proba(X[te])[:, 1]
    auc = auc_with_ci(scores, y)
    aupr = float(average_precision_score(y, scores))
    return {"auc": auc, "auprc": aupr, "scores": scores}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--activations_dir", required=True, type=Path)
    p.add_argument("--threshold", type=float, default=0.5)
    args = p.parse_args()
    common.setup_logging()
    out = common.results_dir(args.activations_dir, "exp_p")

    basis_t = _load_basis(args.activations_dir)
    basis = basis_t.cpu().numpy()
    k = basis.shape[1]
    feats = _build_features(args.activations_dir, basis)
    if feats is None:
        print("[exp_p] no features built; aborting")
        return
    X, y = feats["X"], feats["y"]
    full_H, qlens, task_idx = feats["full_H"], feats["qlens"], feats["task"]

    results = {"k": k, "n": int(X.shape[0]),
               "n_bucket1": int(y.sum())}

    # P2 — main probe on subspace features
    main_eval = _eval_probe(X, y)
    if main_eval is None:
        print("[exp_p] insufficient Bucket-1 examples; aborting probe fit")
        return
    results["P2_main"] = {"auc": main_eval["auc"], "auprc": main_eval["auprc"]}

    # Fit final probe on ALL data and save
    clf = LogisticRegression(C=1.0, max_iter=2000).fit(X, y)
    joblib.dump(clf, out / "probe.joblib")
    torch.save({"basis": basis_t}, out / "basis.pt")

    # P3 — dumb baselines
    p3 = {}
    # (a) full hidden state
    full_eval = _eval_probe(full_H, y)
    if full_eval is not None:
        p3["full_hidden"] = {"auc": full_eval["auc"], "auprc": full_eval["auprc"]}

    # (b) random k-d projection
    rng = np.random.default_rng(42)
    R = rng.normal(size=(full_H.shape[1], k))
    R, _ = np.linalg.qr(R)
    rand_eval = _eval_probe(full_H @ R, y)
    if rand_eval is not None:
        p3["random_kd"] = {"auc": rand_eval["auc"], "auprc": rand_eval["auprc"]}

    # (d) question length (1d feature)
    ql_eval = _eval_probe(qlens.reshape(-1, 1).astype(float), y)
    if ql_eval is not None:
        p3["question_length"] = {"auc": ql_eval["auc"], "auprc": ql_eval["auprc"]}

    # (f) task one-hot
    task_oh = np.eye(len(common.TASKS))[task_idx]
    task_eval = _eval_probe(task_oh, y)
    if task_eval is not None:
        p3["task_onehot"] = {"auc": task_eval["auc"], "auprc": task_eval["auprc"]}

    results["P3_baselines"] = p3

    # P5 — cross-task generalization
    p5 = {}
    for held_out_task in range(len(common.TASKS)):
        tr_mask = task_idx != held_out_task
        te_mask = task_idx == held_out_task
        if y[te_mask].sum() < 5 or (~y[te_mask].astype(bool)).sum() < 5:
            continue
        clf_x = LogisticRegression(C=1.0, max_iter=2000).fit(X[tr_mask], y[tr_mask])
        scores = clf_x.predict_proba(X[te_mask])[:, 1]
        auc = auc_with_ci(scores, y[te_mask])
        p5[common.TASKS[held_out_task]] = auc
    results["P5_cross_task"] = p5

    # P4 — calibration (10 bins)
    bins = np.linspace(0, 1, 11)
    cal = []
    s = main_eval["scores"]
    for i in range(10):
        m = (s >= bins[i]) & (s < bins[i + 1])
        if m.sum() > 0:
            cal.append({"bin_lo": float(bins[i]), "bin_hi": float(bins[i + 1]),
                        "n": int(m.sum()),
                        "mean_pred": float(s[m].mean()),
                        "actual_b1_rate": float(y[m].mean())})
    results["P4_calibration"] = cal

    # also compute fallback rate at threshold τ for use by random_gated control
    fallback_rate = float((main_eval["scores"] < args.threshold).mean())
    results["fallback_rate_at_threshold"] = {
        "threshold": args.threshold, "rate": fallback_rate,
    }

    (out / "exp_p.json").write_text(json.dumps(results, indent=2))
    print(f"[exp_p] probe saved at {out/'probe.joblib'}; "
          f"AUC={results['P2_main']['auc']['auc']:.3f}; "
          f"fallback_rate@{args.threshold}={fallback_rate:.3f}")
    print(f"  → set extras.fallback_rate={fallback_rate:.3f} on random_gated condition")


if __name__ == "__main__":
    main()
