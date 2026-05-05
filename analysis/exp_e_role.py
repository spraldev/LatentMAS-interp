"""Exp E — role specialization.

E1. Within-agent vs cross-agent cosine similarity (LMAS vs TextMAS)
E2. Agent-identity classifier (3-way) — accuracy >> 33% means roles are distinct
E3. Anchoring effect: cos(agent_1, agent_K) for K=2,3
E4. Layer-wise role emergence — only if latent_per_layer is saved
E5. Mirror neuron dynamics: cos(h_B^{m+1}, h_A^m) - cos(h_B^{m+1}, h_C^m)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from analysis import common
from analysis.stats import bootstrap_ci, mannwhitney_u


def _within_vs_cross(X: np.ndarray) -> dict:
    """X: [N, A, M, D]. Returns mean within-agent and cross-agent cos sim."""
    N, A, M, D = X.shape
    Xn = X / (np.linalg.norm(X, axis=-1, keepdims=True) + 1e-12)
    within, cross = [], []
    for n in range(N):
        for a in range(A):
            for b in range(A):
                # mean over rounds
                ca = Xn[n, a].mean(0)
                cb = Xn[n, b].mean(0)
                ca /= (np.linalg.norm(ca) + 1e-12)
                cb /= (np.linalg.norm(cb) + 1e-12)
                s = float(np.dot(ca, cb))
                if a == b:
                    within.append(s)
                else:
                    cross.append(s)
    wm, wlo, whi = bootstrap_ci(within)
    cm, clo, chi = bootstrap_ci(cross)
    return {"within_mean": wm, "within_ci": [wlo, whi],
            "cross_mean": cm, "cross_ci": [clo, chi],
            "ratio": wm / cm if cm > 0 else float("inf")}


def _agent_id_classifier(X: np.ndarray) -> dict:
    N, A, M, D = X.shape
    H = X.reshape(N * A * M, D)
    y = np.repeat(np.tile(np.arange(A), N), M)
    if H.shape[0] > 5000:
        idx = np.random.default_rng(42).choice(H.shape[0], 5000, replace=False)
        H = H[idx]; y = y[idx]
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    accs = []
    for tr, te in skf.split(H, y):
        clf = LogisticRegression(C=1.0, max_iter=2000).fit(H[tr], y[tr])
        accs.append(float((clf.predict(H[te]) == y[te]).mean()))
    m, lo, hi = bootstrap_ci(accs)
    return {"accuracy": m, "ci_lo": lo, "ci_hi": hi,
            "n_classes": A, "chance": 1 / A}


def _anchoring(X: np.ndarray) -> dict:
    N, A, M, D = X.shape
    if A < 3:
        return {}
    cos12, cos13 = [], []
    for n in range(N):
        h1 = X[n, 0, -1]; h1 /= (np.linalg.norm(h1) + 1e-12)
        h2 = X[n, 1, -1]; h2 /= (np.linalg.norm(h2) + 1e-12)
        h3 = X[n, 2, -1]; h3 /= (np.linalg.norm(h3) + 1e-12)
        cos12.append(float(np.dot(h1, h2)))
        cos13.append(float(np.dot(h1, h3)))
    return {"mean_cos_a1_a2": float(np.mean(cos12)),
            "mean_cos_a1_a3": float(np.mean(cos13)),
            "anchoring_ratio_a1a3_vs_a1a2": float(np.mean(cos13) / max(np.mean(cos12), 1e-9))}


def _mirror(X: np.ndarray) -> dict:
    N, A, M, D = X.shape
    if A < 3 or M < 2:
        return {}
    Xn = X / (np.linalg.norm(X, axis=-1, keepdims=True) + 1e-12)
    scores = []
    for n in range(N):
        for m in range(M - 1):
            for b in range(A):
                a = (b - 1) % A
                c = (b + 1) % A
                if a == b or c == b or a == c:
                    continue
                s_a = float(np.dot(Xn[n, b, m + 1], Xn[n, a, m]))
                s_c = float(np.dot(Xn[n, b, m + 1], Xn[n, c, m]))
                scores.append(s_a - s_c)
    if not scores:
        return {}
    m, lo, hi = bootstrap_ci(scores)
    return {"mean_mirror_score": m, "ci_lo": lo, "ci_hi": hi, "n": len(scores)}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--activations_dir", required=True, type=Path)
    args = p.parse_args()
    common.setup_logging()
    out = common.results_dir(args.activations_dir, "exp_e")

    results = {}
    for cond in ("latent_mas", "text_mas"):
        cond_results = {}
        for task in common.TASKS:
            X, _ = common.stack_post_aligned(args.activations_dir, cond, task)
            if X.size == 0 or X.shape[1] < 2:
                continue
            cond_results[task] = {
                "E1_within_vs_cross": _within_vs_cross(X),
                "E2_agent_id_classifier": _agent_id_classifier(X),
                "E3_anchoring": _anchoring(X),
                "E5_mirror": _mirror(X),
            }
        results[cond] = cond_results

    # E4 — layer-wise role emergence (if available)
    e4 = {}
    for task in common.TASKS:
        # collect per_layer activations
        by_ex = []
        for ex in common.iter_examples(args.activations_dir, "latent_mas", task):
            lp = common.load_latent_per_layer(ex.dir)
            if lp is None:
                continue
            by_ex.append(lp["hidden_per_layer"].numpy())  # [A, M, L+1, D]
            if len(by_ex) >= 100:
                break
        if not by_ex:
            continue
        X = np.stack(by_ex, axis=0)  # [N, A, M, L+1, D]
        N, A, M, L, D = X.shape
        per_layer_acc = []
        for lyr in range(L):
            Hl = X[:, :, :, lyr, :].reshape(N * A * M, D)
            y = np.repeat(np.tile(np.arange(A), N), M)
            if Hl.shape[0] > 4000:
                idx = np.random.default_rng(42).choice(Hl.shape[0], 4000, replace=False)
                Hl = Hl[idx]; y = y[idx]
            try:
                skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
                accs = []
                for tr, te in skf.split(Hl, y):
                    clf = LogisticRegression(C=1.0, max_iter=1000).fit(Hl[tr], y[tr])
                    accs.append(float((clf.predict(Hl[te]) == y[te]).mean()))
                per_layer_acc.append(float(np.mean(accs)))
            except Exception:
                per_layer_acc.append(0.0)
        e4[task] = per_layer_acc
    results["E4_layer_emergence"] = e4

    (out / "exp_e.json").write_text(json.dumps(results, indent=2))
    print(f"[exp_e] wrote {out/'exp_e.json'}")


if __name__ == "__main__":
    main()
