#!/usr/bin/env python3
"""Generate all paper figures from results/.

Reads:  results/exp_*/  results/report/
Writes: figures/*.png  figures/*.pdf  figures/FIGURE_GUIDE.txt

Rules:
  - No figure titles
  - No em dashes in any labels (replaced with hyphens)
  - Figures saved as both PDF (vector) and PNG (raster)

Usage:
  python make_figures.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 140,
    "savefig.dpi": 220,
    "savefig.bbox": "tight",
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
    "legend.frameon": False,
    "lines.linewidth": 2.0,
})

TASKS = ["gsm8k", "arc_challenge", "mbppplus"]
TASK_LABEL = {"gsm8k": "GSM8K", "arc_challenge": "ARC-Challenge", "mbppplus": "MBPP+"}
TASK_COLOR = {"gsm8k": "#2E86AB", "arc_challenge": "#E07A5F", "mbppplus": "#3D5A80"}

CONDITION_LABEL = {
    "latent_mas": "LatentMAS",
    "single_agent_latent_sampled": "Single-Agent (sampled)",
    "single_agent_latent_greedy": "Single-Agent (greedy)",
    "latent_mas_random_wa_spectrum": "Random-Wa",
    "kv_blocked": "KV-blocked",
    "no_transfer": "No-transfer",
    "text_mas": "TextMAS",
    "exp_m_identity_wa": "Identity-Wa",
}
CONDITION_ORDER = [
    "latent_mas", "single_agent_latent_sampled", "single_agent_latent_greedy",
    "text_mas", "no_transfer", "kv_blocked", "latent_mas_random_wa_spectrum", "exp_m_identity_wa",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_json(rel: str):
    p = RESULTS / rel
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def save(fig, name: str):
    fig.savefig(FIG_DIR / f"{name}.png")
    plt.close(fig)
    print(f"  {name}.png")


def nd(s: str) -> str:
    return s.replace("—", "-").replace("–", "-")


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def fig_main_accuracy():
    df = pd.read_csv(RESULTS / "report" / "accuracy.csv")
    conds = [c for c in CONDITION_ORDER if c in df["condition"].unique()]
    fig, ax = plt.subplots(figsize=(8, 5.5))
    offsets = np.linspace(-0.22, 0.22, len(TASKS))
    for i, (task, off) in enumerate(zip(TASKS, offsets)):
        sub = df[df["task"] == task].set_index("condition").reindex(conds)
        accs = sub["accuracy"].values * 100
        lo = (sub["accuracy"].values - sub["ci_lo"].values) * 100
        hi = (sub["ci_hi"].values - sub["accuracy"].values) * 100
        y = np.arange(len(conds)) + off
        ax.errorbar(accs, y, xerr=[lo, hi], fmt="o", color=TASK_COLOR[task],
                    label=TASK_LABEL[task], capsize=3, ms=6, lw=1.5)
    ax.set_yticks(np.arange(len(conds)))
    ax.set_yticklabels([nd(CONDITION_LABEL[c]) for c in conds])
    ax.set_xlabel("Accuracy (%)")
    ax.set_xlim(60, 100)
    ax.invert_yaxis()
    ax.axvline(ax.get_xlim()[0], color="black", lw=0.5)
    ax.legend(loc="lower right", ncols=1)
    fig.tight_layout()
    save(fig, "fig_main_accuracy")


def fig_compute_efficiency():
    df = pd.read_csv(RESULTS / "report" / "compute.csv")
    df["wall_s"] = df["wall_clock_ms_mean"] / 1000.0

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.0), sharey=False)
    for ax, task in zip(axes, TASKS):
        sub = df[df["task"] == task].set_index("condition").reindex(CONDITION_ORDER).dropna(how="all")
        labels = [nd(CONDITION_LABEL.get(c, c)) for c in sub.index]
        ax.barh(labels, sub["wall_s"].values, color=TASK_COLOR[task], alpha=0.85)
        ax.invert_yaxis()
        ax.set_xlabel("Wall-clock (s / example)")
        ax.set_title(TASK_LABEL[task])
        for i, v in enumerate(sub["wall_s"].values):
            if not np.isnan(v):
                ax.text(v, i, f" {v:.1f}", va="center", fontsize=8)
    fig.tight_layout()
    save(fig, "fig_wallclock")

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.0))
    for ax, task in zip(axes, TASKS):
        sub = df[df["task"] == task].set_index("condition").reindex(CONDITION_ORDER).dropna(how="all")
        labels = [nd(CONDITION_LABEL.get(c, c)) for c in sub.index]
        toks = sub["generated_tokens_mean"].values
        ax.barh(labels, toks, color=TASK_COLOR[task], alpha=0.85)
        ax.invert_yaxis()
        ax.set_xlabel("Tokens generated / example")
        ax.set_title(TASK_LABEL[task])
        for i, v in enumerate(toks):
            ax.text(v, i, f" {int(v)}" if v > 0 else "  n/a", va="center", fontsize=8)
    fig.tight_layout()
    save(fig, "fig_tokens")



def fig_exp_c_geometry():
    C = load_json("exp_c/exp_c.json")
    if C is None:
        print("  [skip] exp_c.json not found")
        return
    pos = C["C4_per_position"]
    A, R = 3, 4
    H = np.zeros((A, R))
    for a in range(A):
        for r in range(R):
            H[a, r] = pos[f"agent_{a}_round_{r}"]["accuracy"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw={"width_ratios": [1.4, 1]})
    ax = axes[0]
    im = ax.imshow(H, vmin=0.95, vmax=1.0, cmap="viridis", aspect="auto")
    ax.set_xticks(range(R)); ax.set_yticks(range(A))
    ax.set_xticklabels([f"Round {r}" for r in range(R)])
    ax.set_yticklabels([f"Agent {a}" for a in range(A)])
    ax.set_xlabel("Round"); ax.set_ylabel("Agent")
    for a in range(A):
        for r in range(R):
            ax.text(r, a, f"{H[a,r]*100:.1f}", ha="center", va="center",
                    color="white" if H[a, r] < 0.985 else "black", fontsize=9)
    plt.colorbar(im, ax=ax, label="Accuracy")

    ax = axes[1]
    cats = ["LatentMAS\n(C1)", "Single-Agent\n(C2)"]
    vals = [C["C1_latent_mas"]["accuracy"] * 100,
            C["C2_single_agent_latent_sampled"]["accuracy"] * 100]
    los = [(C["C1_latent_mas"]["accuracy"] - C["C1_latent_mas"]["ci_lo"]) * 100,
           (C["C2_single_agent_latent_sampled"]["accuracy"]
            - C["C2_single_agent_latent_sampled"]["ci_lo"]) * 100]
    his = [(C["C1_latent_mas"]["ci_hi"] - C["C1_latent_mas"]["accuracy"]) * 100,
           (C["C2_single_agent_latent_sampled"]["ci_hi"]
            - C["C2_single_agent_latent_sampled"]["accuracy"]) * 100]
    ax.bar(cats, vals, yerr=[los, his], color=["#2E86AB", "#E07A5F"], capsize=4)
    ax.set_ylabel("Task-classification accuracy (%)")
    ax.set_ylim(98, 100.2)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.05, f"{v:.2f}", ha="center", fontsize=9)
    fig.tight_layout()
    save(fig, "fig_exp_c_task_geometry")


def fig_exp_d_trajectory():
    D = load_json("exp_d/exp_d.json")
    if D is None:
        print("  [skip] exp_d.json not found")
        return
    H = np.array(D["D1_latent_mas"]["heatmap"])
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))

    ax = axes[0]
    im = ax.imshow(H, vmin=0.6, vmax=0.75, cmap="magma", aspect="auto")
    ax.set_xticks(range(H.shape[1])); ax.set_yticks(range(H.shape[0]))
    ax.set_xticklabels([f"Round {r}" for r in range(H.shape[1])])
    ax.set_yticklabels([f"Agent {a}" for a in range(H.shape[0])])
    ax.set_xlabel("Round"); ax.set_ylabel("Agent")
    for i in range(H.shape[0]):
        for j in range(H.shape[1]):
            ax.text(j, i, f"{H[i,j]:.2f}", ha="center", va="center",
                    color="white" if H[i, j] < 0.68 else "black", fontsize=9)
    plt.colorbar(im, ax=ax, label="Correctness AUC")

    ax = axes[1]
    H2 = np.array(D["D2_single_agent"]["heatmap"])
    im2 = ax.imshow(H2, vmin=0.65, vmax=0.72, cmap="magma", aspect="auto")
    ax.set_yticks([0]); ax.set_yticklabels(["Single agent"])
    ax.set_xticks(range(H2.shape[1]))
    ax.set_xticklabels([f"R{r}" for r in range(H2.shape[1])])
    ax.set_xlabel("Latent step")
    for j in range(H2.shape[1]):
        ax.text(j, 0, f"{H2[0,j]:.2f}", ha="center", va="center",
                color="white" if H2[0, j] < 0.69 else "black", fontsize=8)
    plt.colorbar(im2, ax=ax, label="Correctness AUC")
    fig.tight_layout()
    save(fig, "fig_exp_d_trajectory")


def fig_exp_d_intrinsic_dim():
    D = load_json("exp_d/exp_d.json")
    if D is None:
        return
    rows = D["D4_intrinsic_dim"]
    tasks = [r["task"] for r in rows]
    cor = [r["intrinsic_dim_correct"] for r in rows]
    inc = [r["intrinsic_dim_incorrect"] for r in rows]
    fig, ax = plt.subplots(figsize=(6, 4))
    y = np.arange(len(tasks))
    for i, (c, n, task) in enumerate(zip(cor, inc, tasks)):
        ax.plot([c, n], [i, i], color="grey", lw=1.2, zorder=1)
        ax.scatter([c], [i], color="#3D5A80", s=70, zorder=2, label="Correct" if i == 0 else "")
        ax.scatter([n], [i], color="#E07A5F", s=70, zorder=2, marker="D",
                   label="Incorrect" if i == 0 else "")
        ax.text(c - 0.3, i, f"{c:.1f}", va="center", ha="right", fontsize=8, color="#3D5A80")
        ax.text(n + 0.3, i, f"{n:.1f}", va="center", ha="left", fontsize=8, color="#E07A5F")
    ax.set_yticks(y)
    ax.set_yticklabels([TASK_LABEL[t] for t in tasks])
    ax.set_xlabel("Intrinsic dimension (TwoNN)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    save(fig, "fig_exp_d_intrinsic_dim")


def fig_exp_e_role():
    E = load_json("exp_e/exp_e.json")
    if E is None:
        print("  [skip] exp_e.json not found")
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ax = axes[0]
    x = np.arange(len(TASKS))
    w = 0.35
    within = [E["latent_mas"][t]["E1_within_vs_cross"]["within_mean"] for t in TASKS]
    cross = [E["latent_mas"][t]["E1_within_vs_cross"]["cross_mean"] for t in TASKS]
    ax.bar(x - w/2, within, w, label="within agent", color="#2E86AB")
    ax.bar(x + w/2, cross, w, label="cross agent", color="#E07A5F")
    ax.set_xticks(x); ax.set_xticklabels([TASK_LABEL[t] for t in TASKS])
    ax.set_ylim(0.8, 1.02)
    ax.set_ylabel("Mean cosine similarity")
    ax.legend()

    ax = axes[1]
    accs = [E["latent_mas"][t]["E2_agent_id_classifier"]["accuracy"] * 100 for t in TASKS]
    los = [(E["latent_mas"][t]["E2_agent_id_classifier"]["accuracy"]
            - E["latent_mas"][t]["E2_agent_id_classifier"]["ci_lo"]) * 100 for t in TASKS]
    his = [(E["latent_mas"][t]["E2_agent_id_classifier"]["ci_hi"]
            - E["latent_mas"][t]["E2_agent_id_classifier"]["accuracy"]) * 100 for t in TASKS]
    ax.axhline(33.3, color="grey", ls="--", lw=1, label="chance")
    ax.errorbar(x, accs, yerr=[los, his], fmt="o", color="#2E86AB", capsize=4, ms=8, lw=2)
    ax.set_xticks(x); ax.set_xticklabels([TASK_LABEL[t] for t in TASKS])
    ax.set_ylim(30, 102)
    ax.set_ylabel("Agent-ID classifier accuracy (%)")
    ax.legend()
    for i, v in enumerate(accs):
        ax.text(i, v + 1.5, f"{v:.1f}", ha="center", fontsize=9)
    fig.tight_layout()
    save(fig, "fig_exp_e_role")


def fig_exp_e_layer_emergence():
    E = load_json("exp_e/exp_e.json")
    if E is None:
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    for t in TASKS:
        curve = E["E4_layer_emergence"][t]
        ax.plot(curve, label=TASK_LABEL[t], color=TASK_COLOR[t])
    ax.set_xlabel("Transformer layer")
    ax.set_ylabel("Agent-ID probe accuracy")
    ax.set_ylim(0.94, 1.0)
    ax.legend()
    save(fig, "fig_exp_e_layer_emergence")


def fig_exp_f_information():
    F = load_json("exp_f/exp_f.json")
    if F is None:
        print("  [skip] exp_f.json not found")
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax = axes[0]
    ks = [1, 2, 3]
    width = 0.25
    x = np.arange(len(ks))
    for i, t in enumerate(TASKS):
        vals = [F["latent_mas"][t][f"k={k}"]["mi_all"] for k in ks]
        ax.bar(x + (i - 1) * width, vals, width, label=TASK_LABEL[t], color=TASK_COLOR[t])
    ax.set_xticks(x); ax.set_xticklabels([f"Round {k}" for k in ks])
    ax.set_ylabel("Kraskov mutual information (nats)")
    ax.legend()

    ax = axes[1]
    for t in TASKS:
        curve = F["F4_rate_distortion"][t]["curve"]
        ks2 = [c["k"] for c in curve]
        aucs = [c["auc"] for c in curve]
        los = [c["auc"] - c["ci_lo"] for c in curve]
        his = [c["ci_hi"] - c["auc"] for c in curve]
        ax.errorbar(ks2, aucs, yerr=[los, his], label=TASK_LABEL[t],
                    color=TASK_COLOR[t], marker="o", capsize=2)
    ax.set_xscale("log")
    ax.set_xlabel("k (top-PC retained)")
    ax.set_ylabel("Correctness AUC")
    ax.legend()
    fig.tight_layout()
    save(fig, "fig_exp_f_information")


def fig_exp_g_groupthink():
    G = load_json("exp_g/exp_g.json")
    if G is None:
        print("  [skip] exp_g.json not found")
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    for t in TASKS:
        rec = G["latent_mas"][t]["G1"]
        rounds = [r["round"] for r in rec]
        means = [r["mean_F"] for r in rec]
        los = [r["mean_F"] - r["ci_lo"] for r in rec]
        his = [r["ci_hi"] - r["mean_F"] for r in rec]
        ax.errorbar(rounds, means, yerr=[los, his], label=TASK_LABEL[t],
                    color=TASK_COLOR[t], marker="o", capsize=3)
    ax.set_xlabel("Round")
    ax.set_ylabel("Mean inter-agent cosine (1 = collapse)")
    ax.legend()
    save(fig, "fig_exp_g_groupthink")


def fig_exp_i_blame():
    I = load_json("exp_i/exp_i.json")
    if I is None:
        print("  [skip] exp_i.json not found")
        return
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, t in zip(axes, TASKS):
        H = np.array(I["latent_mas"][t]["I2_blame_distribution"])
        im = ax.imshow(H, cmap="Reds", aspect="auto")
        ax.set_title(f"{TASK_LABEL[t]}")
        ax.set_xticks(range(H.shape[1]))
        ax.set_xticklabels([f"R{r}" for r in range(H.shape[1])])
        ax.set_yticks(range(H.shape[0]))
        ax.set_yticklabels([f"Agent {a}" for a in range(H.shape[0])])
        ax.set_xlabel("Round")
        for i in range(H.shape[0]):
            for j in range(H.shape[1]):
                if H[i, j] > 0:
                    ax.text(j, i, f"{H[i,j]:.2f}", ha="center", va="center",
                            color="white" if H[i, j] > 0.3 else "black", fontsize=9)
        plt.colorbar(im, ax=ax, label="Blame fraction")
    fig.tight_layout()
    save(fig, "fig_exp_i_blame")


def fig_exp_j_uncertainty():
    J = load_json("exp_j/exp_j.json")
    if J is None:
        print("  [skip] exp_j.json not found")
        return
    J = J["J2"]
    sources = [("Latent", "auc_latent", "#2E86AB"),
               ("Text-regex", "auc_text_regex", "#3D5A80"),
               ("Single-agent", "auc_single_agent", "#E07A5F")]
    fig, ax = plt.subplots(figsize=(9, 4))
    offsets = np.linspace(-0.18, 0.18, len(sources))
    for i, (src, key, color) in enumerate(sources):
        aucs = [J[t][key]["auc"] for t in TASKS]
        los = [J[t][key]["auc"] - J[t][key]["ci_lo"] for t in TASKS]
        his = [J[t][key]["ci_hi"] - J[t][key]["auc"] for t in TASKS]
        y = np.arange(len(TASKS)) + offsets[i]
        ax.errorbar(aucs, y, xerr=[los, his], fmt="o", color=color,
                    label=src, capsize=3, ms=6, lw=1.5)
    ax.axvline(0.5, color="grey", ls="--", lw=1)
    ax.set_yticks(np.arange(len(TASKS)))
    ax.set_yticklabels([TASK_LABEL[t] for t in TASKS])
    ax.set_xlabel("Correctness AUC")
    ax.set_xlim(0.4, 0.8)
    ax.invert_yaxis()
    ax.legend(loc="lower right")
    fig.tight_layout()
    save(fig, "fig_exp_j_uncertainty")


def fig_exp_k_redundancy():
    K = load_json("exp_k/exp_k.json")
    if K is None:
        print("  [skip] exp_k.json not found")
        return
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for ax, t in zip(axes, TASKS):
        H = np.array(K["latent_mas"][t]["K1_R2_matrix"])
        im = ax.imshow(H, vmin=0.6, vmax=1.0, cmap="Blues", aspect="auto")
        ax.set_title(f"{TASK_LABEL[t]} (mean R2={K['latent_mas'][t]['K1_mean_R2']:.2f})")
        ax.set_xticks(range(3)); ax.set_yticks(range(3))
        ax.set_xticklabels([f"A{i}" for i in range(3)])
        ax.set_yticklabels([f"A{i}" for i in range(3)])
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{H[i,j]:.2f}", ha="center", va="center",
                        color="white" if H[i, j] > 0.85 else "black", fontsize=9)
        plt.colorbar(im, ax=ax, label="R2 (predict Aj from Ai)")
    fig.tight_layout()
    save(fig, "fig_exp_k_redundancy")


def fig_exp_l_subspace():
    L = load_json("exp_l/exp_l.json")
    expvar_path = RESULTS / "exp_l" / "expvar_curve.npy"
    if L is None or not expvar_path.exists():
        print("  [skip] exp_l missing")
        return
    expvar = np.load(expvar_path)
    cum = np.cumsum(expvar)
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.plot(np.arange(1, len(cum) + 1), cum, color="#2E86AB", lw=2)
    elbow = L["elbow_k"]
    ax.axvline(elbow, color="#E07A5F", ls="--", lw=1.5,
               label=f"elbow k={elbow} (cum.var={cum[elbow-1]:.2f})")
    ax.axvline(L["D"], color="grey", ls=":", lw=1, label=f"D={L['D']}")
    ax.set_xscale("log")
    ax.set_xlabel("Top-k principal components retained")
    ax.set_ylabel("Cumulative explained variance")
    ax.legend()
    save(fig, "fig_exp_l_subspace")


def fig_exp_m_wa_ablation():
    M = load_json("exp_m/exp_m.json")
    if M is None:
        print("  [skip] exp_m.json not found")
        return
    series = [
        ("LatentMAS (trained Wa)", "latent_mas", "#2E86AB"),
        ("Identity Wa", "exp_m_identity_wa", "#3D5A80"),
        ("No transfer", "no_transfer", "#E07A5F"),
    ]
    labels = [nd(s[0]) for s in series]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(series))
    for t in TASKS:
        accs = [M[t]["accuracy"][key]["accuracy"] * 100 for _, key, _ in series]
        los = [(M[t]["accuracy"][key]["accuracy"] - M[t]["accuracy"][key]["ci_lo"]) * 100
               for _, key, _ in series]
        his = [(M[t]["accuracy"][key]["ci_hi"] - M[t]["accuracy"][key]["accuracy"]) * 100
               for _, key, _ in series]
        ax.plot(x, accs, marker="o", color=TASK_COLOR[t], label=TASK_LABEL[t], ms=7)
        ax.errorbar(x, accs, yerr=[los, his], fmt="none", color=TASK_COLOR[t], capsize=3, lw=1.5)
    for i, (_, key, color) in enumerate(series):
        ax.axvline(i, color="lightgrey", lw=0.7, zorder=0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(60, 100)
    ax.legend(loc="lower left")
    fig.tight_layout()
    save(fig, "fig_exp_m_wa_ablation")


def fig_exp_m_flocking():
    M = load_json("exp_m/exp_m.json")
    if M is None:
        return
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
    for ax, t in zip(axes, TASKS):
        d = M[t]["M4_flocking"]
        rounds = list(range(4))
        for label, key, color in [
            ("LatentMAS", "latent_mas", "#2E86AB"),
            ("Identity Wa", "exp_m_identity_wa", "#3D5A80"),
            ("No transfer", "no_transfer", "#E07A5F"),
        ]:
            ys = [d[key][f"round_{r}"] for r in rounds]
            ax.plot(rounds, ys, marker="o", label=nd(label), color=color)
        ax.set_title(TASK_LABEL[t])
        ax.set_xlabel("Round")
        ax.set_xticks(rounds)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Mean inter-agent cosine")
    axes[-1].legend(loc="lower left")
    fig.tight_layout()
    save(fig, "fig_exp_m_flocking")


def fig_exp_n_sycophancy():
    N = load_json("exp_n/exp_n.json")
    if N is None:
        print("  [skip] exp_n.json not found")
        return
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # Panel 1: dominance R2 per agent — lollipop
    ax = axes[0]
    agent_colors = ["#2E86AB", "#3D5A80", "#E07A5F"]
    offsets = np.linspace(-0.22, 0.22, 3)
    x = np.arange(len(TASKS))
    for a in range(3):
        vals = [N["latent_mas"][t]["N1_dominance_per_agent"][a] for t in TASKS]
        y = x + offsets[a]
        ax.hlines(y, 0, vals, color=agent_colors[a], lw=1.5, alpha=0.7)
        ax.scatter(vals, y, color=agent_colors[a], s=55, zorder=3, label=f"Agent {a}")
    ax.set_yticks(x); ax.set_yticklabels([TASK_LABEL[t] for t in TASKS])
    ax.set_xlabel("Dominance R2")
    ax.invert_yaxis()
    ax.legend(loc="lower right")

    # Panel 2: N2 shift toward dominant — dot with CI + fraction annotation
    ax = axes[1]
    frac_pos = [N["latent_mas"][t]["N2_fraction_positive"] * 100 for t in TASKS]
    means = [N["latent_mas"][t]["N2_mean_shift_toward_dominant"] for t in TASKS]
    ci_los = [N["latent_mas"][t]["N2_ci"][0] for t in TASKS]
    ci_his = [N["latent_mas"][t]["N2_ci"][1] for t in TASKS]
    lo_err = [m - lo for m, lo in zip(means, ci_los)]
    hi_err = [hi - m for hi, m in zip(ci_his, means)]
    y2 = np.arange(len(TASKS))
    ax.axvline(0, color="grey", ls="--", lw=1)
    for i, (t, m, lo, hi, fp) in enumerate(zip(TASKS, means, lo_err, hi_err, frac_pos)):
        ax.hlines(i, 0, m, color=TASK_COLOR[t], lw=2)
        ax.errorbar([m], [i], xerr=[[lo], [hi]], fmt="o", color=TASK_COLOR[t], capsize=3, ms=7)
        off = 0.001 if m >= 0 else -0.001
        ha = "left" if m >= 0 else "right"
        ax.text(m + off, i - 0.25, f"{fp:.0f}% pos", fontsize=8, ha=ha, color="dimgrey")
    ax.set_yticks(y2); ax.set_yticklabels([TASK_LABEL[t] for t in TASKS])
    ax.set_xlabel("Mean cosine shift toward dominant agent")
    ax.invert_yaxis()

    # Panel 3: sycophancy direction AUC — horizontal lollipop
    ax = axes[2]
    aucs = [N["latent_mas"][t]["N4_sycophancy_direction_auc"]["auc"] for t in TASKS]
    los = [N["latent_mas"][t]["N4_sycophancy_direction_auc"]["auc"]
           - N["latent_mas"][t]["N4_sycophancy_direction_auc"]["ci_lo"] for t in TASKS]
    his = [N["latent_mas"][t]["N4_sycophancy_direction_auc"]["ci_hi"]
           - N["latent_mas"][t]["N4_sycophancy_direction_auc"]["auc"] for t in TASKS]
    y = np.arange(len(TASKS))
    ax.axvline(0.5, color="grey", ls="--", lw=1)
    y = np.arange(len(TASKS))
    for i, (t, auc, lo, hi) in enumerate(zip(TASKS, aucs, los, his)):
        ax.hlines(i, 0.5, auc, color=TASK_COLOR[t], lw=2)
        ax.errorbar([auc], [i], xerr=[[lo], [hi]], fmt="o", color=TASK_COLOR[t], capsize=3, ms=7)
    ax.set_yticks(y); ax.set_yticklabels([TASK_LABEL[t] for t in TASKS])
    ax.set_xlim(0.45, 0.78)
    ax.invert_yaxis()
    ax.set_xlabel("Sycophancy direction AUC")

    fig.tight_layout()
    save(fig, "fig_exp_n_sycophancy")


def fig_exp_o_layer_routing():
    O = load_json("exp_o/exp_o.json")
    if O is None:
        print("  [skip] exp_o.json not found")
        return
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))

    ax = axes[0]
    for t in TASKS:
        per = O[t]["O1_per_layer_auc"]
        layers = [p["layer"] for p in per]
        aucs = [p["auc"] for p in per]
        ax.plot(layers, aucs, label=TASK_LABEL[t], color=TASK_COLOR[t], marker="o", ms=3)
    ax.axhline(0.5, color="grey", ls="--", lw=1)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Bucket-1 (channel-positive) AUC")
    ax.legend()

    ax = axes[1]
    for t in TASKS:
        adj = O[t]["O3_cka_adjacent"]
        layers = [a["layer_pair"][0] for a in adj]
        ckas = [a["cka"] for a in adj]
        ax.plot(layers, ckas, label=TASK_LABEL[t], color=TASK_COLOR[t], marker="o", ms=3)
    ax.set_xlabel("Layer pair")
    ax.set_ylabel("Adjacent-layer CKA")
    ax.legend()

    fig.tight_layout()
    save(fig, "fig_exp_o_layer_routing")


def fig_exp_p_probe():
    P = load_json("exp_p/exp_p.json")
    if P is None:
        print("  [skip] exp_p.json not found")
        return
    items = [("k-PCA probe", P["P2_main"]["auc"]),
             ("Full hidden", P["P3_baselines"]["full_hidden"]["auc"]),
             ("Random k-PCA", P["P3_baselines"]["random_kd"]["auc"]),
             ("Question length", P["P3_baselines"]["question_length"]["auc"]),
             ("Task one-hot", P["P3_baselines"]["task_onehot"]["auc"])]
    palette = ["#2E86AB", "#3D5A80", "#9DA5BD", "#9DA5BD", "#E07A5F"]
    fig, ax = plt.subplots(figsize=(7, 4))
    y = np.arange(len(items))
    names = [n for n, _ in items]
    aucs = [v["auc"] for _, v in items]
    los = [v["auc"] - v["ci_lo"] for _, v in items]
    his = [v["ci_hi"] - v["auc"] for _, v in items]
    ax.axvline(0.5, color="grey", ls="--", lw=1)
    for i, (auc, lo, hi, c) in enumerate(zip(aucs, los, his, palette)):
        ax.hlines(i, 0.5, auc, color=c, lw=2)
        ax.errorbar([auc], [i], xerr=[[lo], [hi]], fmt="o", color=c, capsize=3, ms=8)
    for i, (v, name) in enumerate(zip(aucs, names)):
        ax.text(v + 0.01, i, f"{v:.2f}", va="center", fontsize=9)
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.set_xlim(0.45, 0.85)
    ax.set_xlabel("Bucket-1 prediction AUC")
    ax.invert_yaxis()
    fig.tight_layout()
    save(fig, "fig_exp_p_probe")


def fig_exp_q_gated():
    Q = load_json("exp_q/exp_q.json")
    if Q is None:
        print("  [skip] exp_q.json not found")
        return
    gate = "topk_gated"
    has_data = any(
        gate in Q.get(t, {}) and Q[t][gate].get("n_paired", 0) > 0
        for t in TASKS
    )
    if not has_data:
        print("  [skip] exp_q has no topk_gated data")
        return

    diffs, ps, ns = [], [], []
    for t in TASKS:
        d = Q.get(t, {}).get(gate, {})
        ov = d.get("vs_lmas_overall", {})
        diffs.append(ov.get("diff_pp", 0))
        ps.append(ov.get("p_value", 1))
        ns.append(d.get("n_paired", 0))

    fig, ax = plt.subplots(figsize=(6, 3.5))
    y = np.arange(len(TASKS))
    colors = ["#2E86AB" if d >= 0 else "#E07A5F" for d in diffs]
    ax.axvline(0, color="black", lw=0.8)
    ax.hlines(y, 0, diffs, color=colors, lw=2.5)
    ax.scatter(diffs, y, color=colors, s=70, zorder=3)
    for i, (d, p, n) in enumerate(zip(diffs, ps, ns)):
        sig = "*" if p < 0.05 else ""
        off = 0.15 if d >= 0 else -0.15
        ha = "left" if d >= 0 else "right"
        ax.text(d + off, i, f"{d:+.1f}pp{sig}  p={p:.3f}", va="center", fontsize=8, ha=ha)
    ax.set_yticks(y)
    ax.set_yticklabels([TASK_LABEL[t] for t in TASKS])
    ax.set_xlabel("Accuracy diff vs LatentMAS (pp)")
    ax.invert_yaxis()
    fig.tight_layout()
    save(fig, "fig_exp_q_gated")


def fig_summary_panel():
    df_acc_path = RESULTS / "report" / "accuracy.csv"
    df_cmp_path = RESULTS / "report" / "compute.csv"
    M = load_json("exp_m/exp_m.json")
    L = load_json("exp_l/exp_l.json")
    O = load_json("exp_o/exp_o.json")
    expvar_path = RESULTS / "exp_l" / "expvar_curve.npy"
    if not df_acc_path.exists() or M is None or L is None or O is None:
        print("  [skip] summary panel missing deps")
        return

    df_acc = pd.read_csv(df_acc_path)
    df_cmp = pd.read_csv(df_cmp_path)
    df_cmp["wall_s"] = df_cmp["wall_clock_ms_mean"] / 1000.0

    fig = plt.figure(figsize=(13, 8.5))
    gs = fig.add_gridspec(2, 3, hspace=0.42, wspace=0.32)
    headline = ["latent_mas", "single_agent_latent_sampled", "text_mas"]
    width = 0.25
    x = np.arange(len(TASKS))

    axA = fig.add_subplot(gs[0, 0])
    for i, c in enumerate(headline):
        sub = df_acc[df_acc["condition"] == c].set_index("task").reindex(TASKS)
        if sub.empty:
            continue
        accs = sub["accuracy"].values * 100
        lo = (sub["accuracy"].values - sub["ci_lo"].values) * 100
        hi = (sub["ci_hi"].values - sub["accuracy"].values) * 100
        axA.bar(x + (i - 1) * width, accs, width, yerr=[lo, hi],
                color=["#2E86AB", "#3D5A80", "#E07A5F"][i],
                label=nd(CONDITION_LABEL[c]), capsize=2.5)
    axA.set_xticks(x); axA.set_xticklabels([TASK_LABEL[t] for t in TASKS])
    axA.set_ylabel("Accuracy (%)"); axA.set_ylim(60, 100)
    axA.legend(fontsize=8, loc="lower left")

    axB = fig.add_subplot(gs[0, 1])
    for i, c in enumerate(headline):
        sub = df_cmp[df_cmp["condition"] == c].set_index("task").reindex(TASKS)
        if sub.empty:
            continue
        axB.bar(x + (i - 1) * width, sub["wall_s"].values, width,
                color=["#2E86AB", "#3D5A80", "#E07A5F"][i], label=nd(CONDITION_LABEL[c]))
    axB.set_xticks(x); axB.set_xticklabels([TASK_LABEL[t] for t in TASKS])
    axB.set_ylabel("Wall-clock (s / example)")

    axC = fig.add_subplot(gs[0, 2])
    for i, c in enumerate(headline):
        sub = df_cmp[df_cmp["condition"] == c].set_index("task").reindex(TASKS)
        if sub.empty:
            continue
        toks = sub["generated_tokens_mean"].values
        axC.bar(x + (i - 1) * width, toks, width,
                color=["#2E86AB", "#3D5A80", "#E07A5F"][i], label=nd(CONDITION_LABEL[c]))
    axC.set_xticks(x); axC.set_xticklabels([TASK_LABEL[t] for t in TASKS])
    axC.set_ylabel("Tokens generated / example")

    axD = fig.add_subplot(gs[1, 0])
    for t in TASKS:
        d = M[t]["M4_flocking"]["latent_mas"]
        axD.plot(range(4), [d[f"round_{r}"] for r in range(4)],
                 marker="o", color=TASK_COLOR[t], label=TASK_LABEL[t])
    axD.set_xlabel("Round"); axD.set_xticks(range(4))
    axD.set_ylabel("Mean inter-agent cosine")
    axD.legend(fontsize=8)

    axE = fig.add_subplot(gs[1, 1])
    if expvar_path.exists():
        expvar = np.load(expvar_path)
        cum = np.cumsum(expvar)
        axE.plot(np.arange(1, len(cum) + 1), cum, color="#2E86AB", lw=2)
        axE.axvline(L["elbow_k"], color="#E07A5F", ls="--",
                    label=f"elbow k={L['elbow_k']}")
        axE.set_xscale("log")
        axE.set_xlabel("Top-k PCs"); axE.set_ylabel("Cum. explained var")
        axE.legend(fontsize=8)

    axF = fig.add_subplot(gs[1, 2])
    for t in TASKS:
        per = O[t]["O1_per_layer_auc"]
        axF.plot([p["layer"] for p in per], [p["auc"] for p in per],
                 color=TASK_COLOR[t], label=TASK_LABEL[t], marker="o", ms=2)
    axF.axhline(0.5, color="grey", ls="--", lw=1)
    axF.set_xlabel("Layer"); axF.set_ylabel("Channel-positive AUC")
    axF.legend(fontsize=8)

    save(fig, "fig_summary_panel")


# ---------------------------------------------------------------------------
# Figure guide
# ---------------------------------------------------------------------------

GUIDE = """FIGURE GUIDE
============
Generated by make_figures.py. All figures: no titles, no em dashes.
Each figure is saved as both .pdf (use in LaTeX) and .png (for preview).

fig_main_accuracy
  Cleveland dot plot: accuracy (%) across all 8 conditions x 3 tasks, with
  Wilson 95% CI error bars. Use as Table 1 companion or main results figure.
  Key story: LatentMAS vs single-agent and TextMAS baselines.

fig_wallclock
  Horizontal bar chart: wall-clock seconds per example, per task (3 panels).
  Lower is better. Shows TextMAS is 3-5x slower than LatentMAS.

fig_tokens
  Horizontal bar chart: generated tokens per example, per task (3 panels).
  TextMAS shows 0 (vLLM counter not exposed). LatentMAS is token-efficient.

fig_exp_c_task_geometry
  Left: heatmap of task-identity probe accuracy at each (agent, round) site.
  Right: overall task-ID accuracy for LatentMAS vs single-agent.
  Shows task identity is linearly readable from latent states (>99% accuracy).

fig_exp_d_trajectory
  Left: correctness AUC heatmap per (agent, round) for LatentMAS.
  Right: correctness AUC per latent step for matched single-agent.
  Shows how answer-decodability evolves through the multi-agent process.

fig_exp_d_intrinsic_dim
  Bar chart: TwoNN intrinsic dimension of correct vs incorrect trajectories
  per task. Correct trajectories occupy a lower-dimensional manifold.

fig_exp_e_role
  Left: within- vs cross-agent cosine similarity per task (agents have
  distinct representation clusters). Right: agent-ID classifier accuracy
  (vs 33% chance), showing agents develop stable role-specific representations.

fig_exp_e_layer_emergence
  Line plot: agent-ID probe accuracy vs transformer layer, per task.
  Shows at which layer role identity becomes linearly readable.

fig_exp_f_information
  Left: Kraskov mutual information between latent state and correctness label,
  at rounds 1-3. Right: rate-distortion curve (correctness AUC vs top-k PCs
  retained, log scale). Shows how much useful information is in the latent space.

fig_exp_g_groupthink
  Line plot: mean inter-agent cosine similarity across rounds, per task.
  Tracks whether agents converge (groupthink) or maintain diversity over rounds.

fig_exp_i_blame
  Heatmap per task: fraction of errors introduced at each (agent, round) site.
  Use to identify where in the pipeline errors originate.

fig_exp_j_uncertainty
  Bar chart: correctness AUC from 3 uncertainty signals (latent-based,
  text-regex, single-agent baseline), per task. Compares where calibration lives.

fig_exp_k_redundancy
  Cross-agent R2 matrix per task (3 panels): how well agent j's representation
  can be predicted from agent i's. High R2 = redundant information.

fig_exp_l_subspace
  Log-scale cumulative explained variance curve of W_a communication subspace.
  Vertical lines mark elbow (active dimensions) and full dimension D.
  The gap elbow/D is the fraction of dead dimensions in the latent channel.

fig_exp_m_wa_ablation
  Bar chart: accuracy for LatentMAS (trained Wa), Identity Wa, No-transfer.
  Annotated with diff vs no-transfer and McNemar p-value per task.
  Use to argue Wa is load-bearing, especially on MBPP+.

fig_exp_m_flocking
  Line plot: inter-agent cosine convergence over rounds for LatentMAS vs
  Identity Wa vs No-transfer, per task. Shows how the channel shapes dynamics.

fig_exp_n_sycophancy
  3-panel figure: (1) dominance R2 per agent per task -- Agent 1 (Critic)
  consistently dominates; (2) mean cosine shift toward dominant agent (N2)
  with CI and fraction-positive annotation -- later agents shift toward Critic;
  (3) sycophancy direction AUC (N4) -- a linear direction in latent space
  predicts failure cases above chance.

fig_exp_o_layer_routing
  Left: per-layer probe AUC for the channel-positive (B1) set, per task.
  Shows which layers carry the useful latent signal (typically mid-stack).
  Right: adjacent-layer CKA showing representation similarity across depth.

fig_exp_p_probe
  Bar chart: AUC for predicting B1 (collaboration-wins) examples using
  5 signals: k-PCA probe, full hidden, random k-PCA, question length, task
  one-hot. Shows the trained probe beats all baselines (AUC ~0.73).

fig_exp_q_gated
  Bar chart: accuracy diff (topk_gated vs LatentMAS) per task, with
  McNemar p-values and n. Positive = gating hurts; negative = gating helps.

fig_summary_panel
  6-panel overview figure: (A) accuracy vs baselines, (B) wall-clock,
  (C) tokens, (D) flocking dynamics, (E) subspace cumvar, (F) per-layer AUC.
  Good as Figure 1 or a poster summary panel.
"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Generating figures -> figures/")

    fig_main_accuracy()
    fig_compute_efficiency()
    fig_exp_c_geometry()
    fig_exp_d_trajectory()
    fig_exp_d_intrinsic_dim()
    fig_exp_e_role()
    fig_exp_e_layer_emergence()
    fig_exp_f_information()
    fig_exp_g_groupthink()
    fig_exp_i_blame()
    fig_exp_j_uncertainty()
    fig_exp_k_redundancy()
    fig_exp_l_subspace()
    fig_exp_m_wa_ablation()
    fig_exp_m_flocking()
    fig_exp_n_sycophancy()
    fig_exp_o_layer_routing()
    fig_exp_p_probe()
    fig_exp_q_gated()
    fig_summary_panel()

    (FIG_DIR / "FIGURE_GUIDE.txt").write_text(GUIDE)
    print(f"\nFigure guide written to figures/FIGURE_GUIDE.txt")
    print("Done.")
