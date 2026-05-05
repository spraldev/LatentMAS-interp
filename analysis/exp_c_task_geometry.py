"""Exp C — task-domain geometry (linear probing).

C1. 3-way linear SVM on post-W_a hidden states (gsm8k vs arc vs mbppplus)
C2. Same on single_agent_latent_sampled — comparison
C3. Same on text_mas — comparison
C4. Per-(agent, round) breakdown (does decodability rise across rounds?)
C5. UMAP figure colored by (task, agent, round, correctness)
C6. K-means clustering, ARI vs task labels (cross-agent vocabulary)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from analysis import common
from analysis.stats import bootstrap_ci, mcnemar


def _build_task_dataset(root: Path, condition: str, mode: str = "post",
                        max_per_task: int = 500) -> tuple:
    Xs, ys, info = [], [], []
    for ti, task in enumerate(common.TASKS):
        if mode == "post":
            X, ifo = common.stack_post_aligned(root, condition, task, limit=max_per_task)
        else:
            X, ifo = common.stack_pre_aligned(root, condition, task, limit=max_per_task)
        if X.size == 0:
            continue
        # mean over agents+rounds → [N, D]
        H = X.mean(axis=(1, 2))
        Xs.append(H)
        ys.extend([ti] * H.shape[0])
        info.extend(ifo)
    if not Xs:
        return None, None, None
    return np.concatenate(Xs, axis=0), np.array(ys), info


def _eval_3way(X: np.ndarray, y: np.ndarray, seed: int = 42) -> dict:
    if X is None or X.shape[0] < 30:
        return {"accuracy": 0.5, "n": 0}
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    accs, preds_all, labels_all = [], [], []
    for tr, te in skf.split(X, y):
        scaler = StandardScaler().fit(X[tr])
        clf = LinearSVC(C=1.0, max_iter=2000).fit(scaler.transform(X[tr]), y[tr])
        pred = clf.predict(scaler.transform(X[te]))
        accs.append(float((pred == y[te]).mean()))
        preds_all.append(pred); labels_all.append(y[te])
    preds_all = np.concatenate(preds_all); labels_all = np.concatenate(labels_all)
    cm = confusion_matrix(labels_all, preds_all).tolist()
    m, lo, hi = bootstrap_ci(accs)
    return {"accuracy": m, "ci_lo": lo, "ci_hi": hi, "n": int(X.shape[0]),
            "confusion_matrix": cm, "per_fold": accs}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--activations_dir", required=True, type=Path)
    p.add_argument("--max_per_task", type=int, default=500)
    args = p.parse_args()

    common.setup_logging()
    out = common.results_dir(args.activations_dir, "exp_c")

    results = {}
    # C1 — LatentMAS
    Xl, yl, _ = _build_task_dataset(args.activations_dir, "latent_mas",
                                    max_per_task=args.max_per_task)
    results["C1_latent_mas"] = _eval_3way(Xl, yl)

    # C2 — single-agent latent (sampled)
    Xs, ys, _ = _build_task_dataset(args.activations_dir, "single_agent_latent_sampled",
                                    max_per_task=args.max_per_task)
    results["C2_single_agent_latent_sampled"] = _eval_3way(Xs, ys)

    # C3 — TextMAS — note: text_mas may not save latent_thoughts.pt
    Xt, yt, _ = _build_task_dataset(args.activations_dir, "text_mas",
                                    max_per_task=args.max_per_task)
    results["C3_text_mas"] = _eval_3way(Xt, yt)

    # C4 — per-(agent, round) breakdown on LatentMAS
    c4 = {}
    if Xl is not None:
        # need per-position arrays
        Xs_per = []
        ys_per = []
        for ti, task in enumerate(common.TASKS):
            X, _ = common.stack_post_aligned(args.activations_dir, "latent_mas", task,
                                             limit=args.max_per_task)
            if X.size == 0:
                continue
            Xs_per.append(X)
            ys_per.extend([ti] * X.shape[0])
        if Xs_per:
            X_all = np.concatenate(Xs_per, axis=0)  # [N, A, M, D]
            y_all = np.array(ys_per)
            for a in range(X_all.shape[1]):
                for r in range(X_all.shape[2]):
                    res = _eval_3way(X_all[:, a, r, :], y_all)
                    c4[f"agent_{a}_round_{r}"] = res
    results["C4_per_position"] = c4

    # C6 — k-means on LatentMAS, ARI vs task
    if Xl is not None and Xl.shape[0] > 50:
        km = KMeans(n_clusters=50, n_init=10, random_state=42).fit(Xl)
        results["C6_kmeans_ari_vs_task"] = float(adjusted_rand_score(yl, km.labels_))

    # C5 — UMAP figure (optional, requires umap-learn)
    try:
        import umap
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        if Xl is not None:
            emb = umap.UMAP(n_components=2, random_state=42).fit_transform(Xl)
            fig, ax = plt.subplots(figsize=(6, 5))
            for ti, task in enumerate(common.TASKS):
                m = (yl == ti)
                ax.scatter(emb[m, 0], emb[m, 1], s=4, alpha=0.5, label=task)
            ax.legend(loc="best", fontsize=8)
            ax.set_title("UMAP of LatentMAS post-W_a hiddens (mean over agents/rounds)")
            fig.tight_layout()
            fig.savefig(out / "umap_latent_mas.png", dpi=150)
            plt.close(fig)
    except ImportError:
        pass
    except Exception as e:
        print(f"[exp_c] UMAP skipped: {e}")

    (out / "exp_c.json").write_text(json.dumps(results, indent=2))
    print(f"[exp_c] LatentMAS task-decode acc = "
          f"{results['C1_latent_mas']['accuracy']:.3f}")


if __name__ == "__main__":
    main()
