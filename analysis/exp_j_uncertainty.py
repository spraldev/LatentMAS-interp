"""Exp J — uncertainty encoding.

J1. Confidence direction (correct vs incorrect contrastive)
J2. Confidence-direction AUC vs:
    (a) text-confidence regex proxy
    (b) single-agent baseline confidence direction
J3. Cognitive load proxy: GSM8K equation count vs latent norm (Spearman)
J4. Idle-agent baseline: variance of late-agent thoughts on easy vs hard
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from analysis import common
from analysis.stats import auc_with_ci, delong_paired_test, permutation_test_diff


CONFIDENCE_REGEX = re.compile(
    r"\b(not sure|unsure|maybe|perhaps|i think|might be|possibly|guess)\b",
    re.IGNORECASE,
)


def _gsm8k_steps(solution: str) -> int:
    return len(re.findall(r"=", solution or ""))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--activations_dir", required=True, type=Path)
    args = p.parse_args()
    common.setup_logging()
    out = common.results_dir(args.activations_dir, "exp_j")
    results = {}

    for task in common.TASKS:
        X, ifo = common.stack_post_aligned(args.activations_dir, "latent_mas", task)
        if X.size == 0:
            continue
        y = np.array([1 if i["correct"] else 0 for i in ifo])
        if y.sum() < 10 or (~y.astype(bool)).sum() < 10:
            continue
        H = X[:, :, -1, :].mean(axis=1)  # avg over agents at last round
        # J1+J2: confidence direction via cross-validated logistic regression
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores_lat = np.zeros_like(y, dtype=float)
        for tr, te in skf.split(H, y):
            clf = LogisticRegression(C=1.0, max_iter=2000).fit(H[tr], y[tr])
            scores_lat[te] = clf.predict_proba(H[te])[:, 1]
        auc_lat = auc_with_ci(scores_lat, y)

        # J2a: text-confidence regex
        regex_score = []
        for ex in common.iter_examples(args.activations_dir, "latent_mas", task):
            txt = ex.meta.get("raw_prediction", "") or ""
            regex_score.append(0 if CONFIDENCE_REGEX.search(txt) else 1)
        regex_score = np.array(regex_score[:len(y)])
        auc_text = auc_with_ci(regex_score, y)

        # J2b: single-agent baseline confidence
        Xs, ifos = common.stack_post_aligned(args.activations_dir,
                                              "single_agent_latent_sampled", task)
        auc_sa = {}
        if Xs.size > 0:
            ys = np.array([1 if i["correct"] else 0 for i in ifos])
            if ys.sum() > 10 and (~ys.astype(bool)).sum() > 10:
                Hs = Xs[:, :, -1, :].mean(axis=1)
                scores_s = np.zeros_like(ys, dtype=float)
                for tr, te in skf.split(Hs, ys):
                    scores_s[te] = LogisticRegression(C=1.0, max_iter=2000).fit(
                        Hs[tr], ys[tr]).predict_proba(Hs[te])[:, 1]
                auc_sa = auc_with_ci(scores_s, ys)

        results.setdefault("J2", {})[task] = {
            "auc_latent": auc_lat,
            "auc_text_regex": auc_text,
            "auc_single_agent": auc_sa,
        }

        # J3: cognitive load (GSM8K only)
        if task == "gsm8k":
            steps = []
            norms = []
            for i, ex in enumerate(common.iter_examples(
                    args.activations_dir, "latent_mas", task)):
                if i >= len(H):
                    break
                steps.append(_gsm8k_steps(ex.meta.get("raw_prediction", "")))
                norms.append(float(np.linalg.norm(H[i])))
            if steps and norms:
                rho, pv = spearmanr(steps, norms)
                results.setdefault("J3_cognitive_load", {})[task] = {
                    "spearman_r": float(rho), "p_value": float(pv), "n": len(steps),
                }

        # J4: idle-agent variance comparison
        if task == "gsm8k" and X.shape[1] >= 2 and X.shape[2] >= 2:
            steps_arr = np.array([
                _gsm8k_steps(ex.meta.get("raw_prediction", ""))
                for ex in common.iter_examples(args.activations_dir, "latent_mas", task)
            ])[:X.shape[0]]
            if len(steps_arr) > 20:
                easy_thresh = np.percentile(steps_arr, 10)
                hard_thresh = np.percentile(steps_arr, 90)
                easy = steps_arr <= easy_thresh
                hard = steps_arr >= hard_thresh
                # variance of last-agent last-round vector across examples
                Xl = X[:, -1, -1, :]
                if easy.sum() > 5 and hard.sum() > 5:
                    var_easy = float(np.mean(np.var(Xl[easy], axis=0)))
                    var_hard = float(np.mean(np.var(Xl[hard], axis=0)))
                    perm = permutation_test_diff(
                        np.var(Xl[easy], axis=0).tolist(),
                        np.var(Xl[hard], axis=0).tolist(),
                    )
                    results.setdefault("J4_idle_agent", {})[task] = {
                        "variance_easy": var_easy,
                        "variance_hard": var_hard,
                        "perm_test": perm,
                    }

    (out / "exp_j.json").write_text(json.dumps(results, indent=2))
    print(f"[exp_j] wrote {out/'exp_j.json'}")


if __name__ == "__main__":
    main()
