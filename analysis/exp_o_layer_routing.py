"""Exp O — layer-wise information routing.

Requires latent_per_layer.pt (which is NOT saved when --no_layer_hidden is
set on the full collection). If missing, this script will warn and skip.

O1. Per-layer answer-decodability AUC (logistic regression, per layer)
O2. Per-layer task-identity classifier accuracy
O3. CKA between adjacent layers
O4. W_a bridge layer (CKA(W_a row-space, layer))
O5. Episodic vs semantic alignment (prompt_hidden CKA per layer)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from analysis import common
from analysis.stats import auc_with_ci, benjamini_hochberg, linear_cka


def _gather_per_layer(root: Path, condition: str, task: str, limit: int = 200):
    by_ex = []
    correct = []
    for ex in common.iter_examples(root, condition, task):
        lp = common.load_latent_per_layer(ex.dir)
        if lp is None:
            continue
        by_ex.append(lp["hidden_per_layer"].numpy())  # [A, M, L+1, D]
        correct.append(int(ex.meta.get("correct")))
        if len(by_ex) >= limit:
            break
    if not by_ex:
        return None, None
    return np.stack(by_ex, axis=0), np.array(correct)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--activations_dir", required=True, type=Path)
    p.add_argument("--limit", type=int, default=200)
    args = p.parse_args()
    common.setup_logging()
    out = common.results_dir(args.activations_dir, "exp_o")

    results = {}
    for task in common.TASKS:
        X, y = _gather_per_layer(args.activations_dir, "latent_mas", task, args.limit)
        if X is None:
            print(f"[exp_o] no per-layer data for {task} (likely --no_layer_hidden)")
            continue
        N, A, M, L, D = X.shape
        # collapse agent×round → take last position
        H_per_layer = X[:, -1, -1, :, :]  # [N, L, D]
        per_layer = []
        raw_pvals = []
        keys = []
        if y.sum() > 10 and (~y.astype(bool)).sum() > 10:
            for lyr in range(L):
                Hl = H_per_layer[:, lyr, :]
                skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
                scores = np.zeros_like(y, dtype=float)
                for tr, te in skf.split(Hl, y):
                    scores[te] = LogisticRegression(C=1.0, max_iter=1000).fit(
                        Hl[tr], y[tr]).predict_proba(Hl[te])[:, 1]
                auc = auc_with_ci(scores, y)
                per_layer.append({"layer": lyr, **auc})
                # rough p-value via z-test against AUC=0.5
                z = (auc["auc"] - 0.5) / max(auc.get("se", 1.0), 1e-9)
                from scipy.stats import norm
                raw_pvals.append(2 * (1 - norm.cdf(abs(z))))
                keys.append(f"l{lyr}")
        rejected, qvals = benjamini_hochberg(raw_pvals)
        for entry, pv, qv, rej in zip(per_layer, raw_pvals, qvals, rejected):
            entry.update({"p_value": pv, "q_value": qv, "rejected_q05": bool(rej)})

        # O3 — CKA between adjacent layers (mean over examples)
        cka_pairs = []
        for lyr in range(L - 1):
            cka = linear_cka(H_per_layer[:, lyr, :], H_per_layer[:, lyr + 1, :])
            cka_pairs.append({"layer_pair": [lyr, lyr + 1], "cka": cka})

        # O4 — W_a bridge layer (CKA between W_a-applied vectors and per-layer)
        wa = common.load_wa(args.activations_dir)
        cka_wa = []
        if wa is not None:
            W = wa["W_a"].to(torch.float32).cpu().numpy()
            # average over (a, m) → [N, L, D]
            for lyr in range(L):
                if lyr == 0:
                    continue  # skip embedding (Cleanup 7)
                Hwa = H_per_layer[:, lyr, :] @ W.T
                cka_wa.append({"layer": lyr,
                               "cka_wa_vs_layer": linear_cka(Hwa, H_per_layer[:, lyr, :])})

        results[task] = {
            "n": int(N), "L": int(L), "D": int(D),
            "O1_per_layer_auc": per_layer,
            "O3_cka_adjacent": cka_pairs,
            "O4_wa_bridge_cka": cka_wa,
        }

    (out / "exp_o.json").write_text(json.dumps(results, indent=2))
    print(f"[exp_o] wrote {out/'exp_o.json'}")


if __name__ == "__main__":
    main()
