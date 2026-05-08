"""Compile paper deliverables (tables + figures) from analysis JSONs.

Reads:  workspace/activations/results/{exp_*,report}/*
Writes: paper/figures/*.pdf,*.png  paper/tables/*.{md,tex,csv}
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "workspace" / "activations" / "results"
FIG_DIR = ROOT / "paper" / "figures"
TAB_DIR = ROOT / "paper" / "tables"
FIG_DIR.mkdir(parents=True, exist_ok=True)
TAB_DIR.mkdir(parents=True, exist_ok=True)

# ----- Style ---------------------------------------------------------------
plt.rcParams.update(
    {
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
    }
)

TASKS = ["gsm8k", "arc_challenge", "mbppplus"]
TASK_LABEL = {"gsm8k": "GSM8K", "arc_challenge": "ARC-Challenge", "mbppplus": "MBPP+"}
TASK_COLOR = {"gsm8k": "#2E86AB", "arc_challenge": "#E07A5F", "mbppplus": "#3D5A80"}

CONDITION_LABEL = {
    "latent_mas": "LatentMAS",
    "single_agent_latent_sampled": "Single-Agent (sampled)",
    "single_agent_latent_greedy": "Single-Agent (greedy)",
    "latent_mas_random_wa_spectrum": "Random-$W_a$",
    "kv_blocked": "KV-blocked",
    "no_transfer": "No-transfer",
    "text_mas": "TextMAS",
    "exp_m_identity_wa": "Identity-$W_a$",
}
CONDITION_ORDER = [
    "latent_mas",
    "single_agent_latent_sampled",
    "single_agent_latent_greedy",
    "text_mas",
    "no_transfer",
    "kv_blocked",
    "latent_mas_random_wa_spectrum",
    "exp_m_identity_wa",
]


def load_json(p: Path):
    with open(p) as fh:
        return json.load(fh)


def df_to_markdown(df: pd.DataFrame, index: bool = True) -> str:
    rows = []
    if index:
        header = [str(df.index.name or "")] + list(map(str, df.columns))
        rows.append("| " + " | ".join(header) + " |")
        rows.append("|" + "|".join(["---"] * len(header)) + "|")
        for idx, row in df.iterrows():
            rows.append("| " + " | ".join([str(idx)] + [str(v) for v in row]) + " |")
    else:
        header = list(map(str, df.columns))
        rows.append("| " + " | ".join(header) + " |")
        rows.append("|" + "|".join(["---"] * len(header)) + "|")
        for _, row in df.iterrows():
            rows.append("| " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(rows)


def save(fig, name: str):
    out_pdf = FIG_DIR / f"{name}.pdf"
    out_png = FIG_DIR / f"{name}.png"
    fig.savefig(out_pdf)
    fig.savefig(out_png)
    plt.close(fig)
    print(f"  ✓ {name}.{{pdf,png}}")


# ===========================================================================
# Tables
# ===========================================================================

def make_main_results_table():
    df = pd.read_csv(RESULTS / "report" / "accuracy.csv")
    df["acc_pct"] = df["accuracy"] * 100
    df["ci_lo_pct"] = df["ci_lo"] * 100
    df["ci_hi_pct"] = df["ci_hi"] * 100
    df["formatted"] = df.apply(
        lambda r: f"{r.acc_pct:.1f} [{r.ci_lo_pct:.1f},{r.ci_hi_pct:.1f}]", axis=1
    )
    pivot = df.pivot(index="condition", columns="task", values="formatted")
    pivot = pivot.reindex(CONDITION_ORDER)[TASKS]
    pivot.index = [CONDITION_LABEL.get(c, c) for c in pivot.index]
    pivot.columns = [TASK_LABEL[t] for t in pivot.columns]
    pivot.index.name = "Condition"

    md = df_to_markdown(pivot)
    (TAB_DIR / "table_main_accuracy.md").write_text(
        "# Table 1 — Main accuracy (Wilson 95% CI, accuracy in %)\n\n" + md + "\n"
    )

    tex = pivot.to_latex(
        column_format="l" + "c" * len(pivot.columns),
        caption="Accuracy (\\%) with Wilson 95\\% confidence interval. "
        "LatentMAS is compared against single-agent and TextMAS baselines and against "
        "five mechanistic ablations: removing inter-agent transfer (No-transfer), "
        "blocking KV passing (KV-blocked), randomising the $W_a$ alignment spectrum "
        "(Random-$W_a$), replacing $W_a$ with identity (Identity-$W_a$), and the "
        "trained-$W_a$ LatentMAS (default).",
        label="tab:main_accuracy",
        escape=False,
    )
    (TAB_DIR / "table_main_accuracy.tex").write_text(tex)
    pivot.to_csv(TAB_DIR / "table_main_accuracy.csv")
    print("  ✓ tables/table_main_accuracy.{md,tex,csv}")


def make_compute_table():
    df = pd.read_csv(RESULTS / "report" / "compute.csv")
    df["wall_s"] = df["wall_clock_ms_mean"] / 1000.0
    df["formatted"] = df.apply(
        lambda r: (
            f"{int(round(r.generated_tokens_mean))} tok / {r.wall_s:.1f} s / "
            f"{int(round(r.forward_passes_mean))} fwd"
        ),
        axis=1,
    )
    pivot = df.pivot(index="condition", columns="task", values="formatted")
    pivot = pivot.reindex(CONDITION_ORDER)[TASKS]
    pivot.index = [CONDITION_LABEL.get(c, c) for c in pivot.index]
    pivot.columns = [TASK_LABEL[t] for t in pivot.columns]
    pivot.index.name = "Condition"

    md = df_to_markdown(pivot)
    (TAB_DIR / "table_compute.md").write_text(
        "# Table 2 — Compute (mean tokens generated / wall-clock seconds / forward passes per example)\n\n"
        + md
        + "\n\nForward-pass count is 0 for TextMAS because that condition is run through vLLM and the "
        "HF forward-pass counter is not exposed; tokens generated are also not recorded for vLLM "
        "and shown as 0.\n"
    )

    tex = pivot.to_latex(
        column_format="l" + "c" * len(pivot.columns),
        caption="Compute per example: tokens generated / wall-clock seconds / forward passes. "
        "TextMAS is run through vLLM (forward-pass and generated-token counters not exposed).",
        label="tab:compute",
        escape=False,
    )
    (TAB_DIR / "table_compute.tex").write_text(tex)
    pivot.to_csv(TAB_DIR / "table_compute.csv")
    print("  ✓ tables/table_compute.{md,tex,csv}")


def make_buckets_table():
    df = pd.read_csv(RESULTS / "report" / "buckets.csv")
    df = df.rename(
        columns={
            "B1": "B1 (LMAS✓ / SA✗)",
            "B2": "B2 (LMAS✗ / SA✓)",
            "B3": "B3 (both ✓)",
            "B4": "B4 (both ✗)",
        }
    )
    df["task"] = df["task"].map(TASK_LABEL)
    md = df_to_markdown(df, index=False)
    (TAB_DIR / "table_buckets.md").write_text(
        "# Table 3 — Bucket distribution: LatentMAS (LMAS) vs Single-Agent (SA) outcomes per task\n\n"
        + md
        + "\n\nB1 isolates examples where the multi-agent latent channel demonstrably helps "
        "over the matched single-agent baseline. The B1 fraction is small but non-trivial, "
        "and is the population the predictive probe (Exp P) targets.\n"
    )
    tex = df.to_latex(
        index=False,
        caption="Bucket distribution comparing LatentMAS (LMAS) vs.\\ matched Single-Agent (SA). "
        "B1 = LMAS correct \\& SA wrong (the channel-positive set); B2 = LMAS wrong \\& SA correct.",
        label="tab:buckets",
    )
    (TAB_DIR / "table_buckets.tex").write_text(tex)
    df.to_csv(TAB_DIR / "table_buckets.csv", index=False)
    print("  ✓ tables/table_buckets.{md,tex,csv}")


def make_ablation_table():
    """Mechanistic decomposition: trained W_a vs identity W_a vs no transfer."""
    rows = []
    M = load_json(RESULTS / "exp_m" / "exp_m.json")
    for task in TASKS:
        d = M[task]["accuracy"]
        rows.append(
            {
                "Task": TASK_LABEL[task],
                "LatentMAS (trained $W_a$)": f"{d['latent_mas']['accuracy']*100:.1f} (n={d['latent_mas']['n']})",
                "Identity $W_a$": f"{d['exp_m_identity_wa']['accuracy']*100:.1f} (n={d['exp_m_identity_wa']['n']})",
                "No transfer": f"{d['no_transfer']['accuracy']*100:.1f} (n={d['no_transfer']['n']})",
            }
        )
    df = pd.DataFrame(rows).set_index("Task")
    md = df_to_markdown(df)
    (TAB_DIR / "table_wa_ablation.md").write_text(
        "# Table 4 — $W_a$ decomposition (Exp M)\n\n"
        + md
        + "\n\n*Identity $W_a$* removes the trained linear bridge while keeping the KV channel intact. "
        "*No transfer* removes the channel entirely. The trained $W_a$ adds 0–1pp on GSM8K/ARC and "
        "+5.8pp on MBPP+ (McNemar p=0.010 vs no-transfer): the alignment matrix is load-bearing on "
        "code generation but contributes little on arithmetic and multiple choice.\n"
    )
    tex = df.to_latex(
        column_format="lccc",
        caption="$W_a$ decomposition. Trained $W_a$ vs.\\ identity $W_a$ isolates whether the alignment "
        "matrix carries reasoning content beyond simply preserving the KV channel.",
        label="tab:wa_ablation",
        escape=False,
    )
    (TAB_DIR / "table_wa_ablation.tex").write_text(tex)
    df.to_csv(TAB_DIR / "table_wa_ablation.csv")
    print("  ✓ tables/table_wa_ablation.{md,tex,csv}")


# ===========================================================================
# Figures
# ===========================================================================

def fig_main_accuracy():
    df = pd.read_csv(RESULTS / "report" / "accuracy.csv")
    fig, ax = plt.subplots(figsize=(11, 4.6))
    bar_w = 0.10
    conds = [c for c in CONDITION_ORDER if c in df["condition"].unique()]
    x = np.arange(len(conds))
    for i, task in enumerate(TASKS):
        sub = df[df["task"] == task].set_index("condition").reindex(conds)
        accs = sub["accuracy"].values * 100
        lo = (sub["accuracy"].values - sub["ci_lo"].values) * 100
        hi = (sub["ci_hi"].values - sub["accuracy"].values) * 100
        ax.bar(
            x + (i - 1) * bar_w * 1.05,
            accs,
            bar_w,
            yerr=[lo, hi],
            color=TASK_COLOR[task],
            label=TASK_LABEL[task],
            capsize=2.5,
            edgecolor="white",
            linewidth=0.6,
        )
    ax.set_xticks(x)
    ax.set_xticklabels([CONDITION_LABEL[c] for c in conds], rotation=22, ha="right")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(60, 100)
    ax.set_title("Main accuracy (Wilson 95% CI)")
    ax.legend(loc="lower left", ncols=3)
    save(fig, "fig_main_accuracy")


def fig_compute_efficiency():
    df = pd.read_csv(RESULTS / "report" / "compute.csv")
    df["wall_s"] = df["wall_clock_ms_mean"] / 1000.0

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.0), sharey=False)
    for ax, task in zip(axes, TASKS):
        sub = df[df["task"] == task].set_index("condition").reindex(CONDITION_ORDER).dropna(how="all")
        labels = [CONDITION_LABEL[c] for c in sub.index]
        ax.barh(labels, sub["wall_s"].values, color=TASK_COLOR[task], alpha=0.85)
        ax.invert_yaxis()
        ax.set_title(TASK_LABEL[task])
        ax.set_xlabel("Wall-clock (s/example)")
        for i, v in enumerate(sub["wall_s"].values):
            ax.text(v, i, f" {v:.1f}", va="center", fontsize=8)
    fig.suptitle("Per-example wall-clock time (lower is better)")
    fig.tight_layout()
    save(fig, "fig_wallclock")

    # Tokens
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.0))
    for ax, task in zip(axes, TASKS):
        sub = df[df["task"] == task].set_index("condition").reindex(CONDITION_ORDER).dropna(how="all")
        # zero-out vllm rows where tokens are 0 (not measured)
        labels = [CONDITION_LABEL[c] for c in sub.index]
        toks = sub["generated_tokens_mean"].values
        ax.barh(labels, toks, color=TASK_COLOR[task], alpha=0.85)
        ax.invert_yaxis()
        ax.set_title(TASK_LABEL[task])
        ax.set_xlabel("Tokens generated / example")
        for i, v in enumerate(toks):
            ax.text(v, i, f" {int(v)}" if v > 0 else "  n/a", va="center", fontsize=8)
    fig.suptitle("Generated tokens per example (TextMAS = vLLM, counter not exposed)")
    fig.tight_layout()
    save(fig, "fig_tokens")


def fig_exp_c_geometry():
    """C1, C2, C4 — task-domain probing accuracy."""
    C = load_json(RESULTS / "exp_c" / "exp_c.json")
    pos = C["C4_per_position"]
    # build A (3) x R (4) heatmap
    A, R = 3, 4
    H = np.zeros((A, R))
    for a in range(A):
        for r in range(R):
            H[a, r] = pos[f"agent_{a}_round_{r}"]["accuracy"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw={"width_ratios": [1.4, 1]})
    ax = axes[0]
    im = ax.imshow(H, vmin=0.95, vmax=1.0, cmap="viridis", aspect="auto")
    ax.set_xticks(range(R))
    ax.set_yticks(range(A))
    ax.set_xticklabels([f"Round {r}" for r in range(R)])
    ax.set_yticklabels([f"Agent {a}" for a in range(A)])
    ax.set_title("Exp C4 — Task probe accuracy at each (agent, round) site")
    for a in range(A):
        for r in range(R):
            ax.text(r, a, f"{H[a,r]*100:.1f}", ha="center", va="center",
                    color="white" if H[a, r] < 0.985 else "black", fontsize=9)
    plt.colorbar(im, ax=ax, label="Accuracy")

    ax = axes[1]
    cats = ["LatentMAS\n(C1)", "Single-Agent\n(C2)"]
    vals = [C["C1_latent_mas"]["accuracy"] * 100, C["C2_single_agent_latent_sampled"]["accuracy"] * 100]
    los = [(C["C1_latent_mas"]["accuracy"] - C["C1_latent_mas"]["ci_lo"]) * 100,
           (C["C2_single_agent_latent_sampled"]["accuracy"] - C["C2_single_agent_latent_sampled"]["ci_lo"]) * 100]
    his = [(C["C1_latent_mas"]["ci_hi"] - C["C1_latent_mas"]["accuracy"]) * 100,
           (C["C2_single_agent_latent_sampled"]["ci_hi"] - C["C2_single_agent_latent_sampled"]["accuracy"]) * 100]
    ax.bar(cats, vals, yerr=[los, his], color=["#2E86AB", "#E07A5F"], capsize=4)
    ax.set_ylabel("Task-classification accuracy (%)")
    ax.set_ylim(98, 100.2)
    ax.set_title("Exp C1/C2 — Task identity is linearly readable")
    for i, v in enumerate(vals):
        ax.text(i, v + 0.05, f"{v:.2f}", ha="center", fontsize=9)
    fig.tight_layout()
    save(fig, "fig_exp_c_task_geometry")


def fig_exp_d_trajectory():
    """D1 — correctness AUC heatmap per (agent, round)."""
    D = load_json(RESULTS / "exp_d" / "exp_d.json")
    H = np.array(D["D1_latent_mas"]["heatmap"])  # 3 x 4
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    ax = axes[0]
    im = ax.imshow(H, vmin=0.6, vmax=0.75, cmap="magma", aspect="auto")
    ax.set_xticks(range(H.shape[1])); ax.set_yticks(range(H.shape[0]))
    ax.set_xticklabels([f"Round {r}" for r in range(H.shape[1])])
    ax.set_yticklabels([f"Agent {a}" for a in range(H.shape[0])])
    ax.set_title("Exp D1 — Correctness AUC per (agent, round)")
    for i in range(H.shape[0]):
        for j in range(H.shape[1]):
            ax.text(j, i, f"{H[i,j]:.2f}", ha="center", va="center",
                    color="white" if H[i, j] < 0.68 else "black", fontsize=9)
    plt.colorbar(im, ax=ax, label="AUC")

    ax = axes[1]
    H2 = np.array(D["D2_single_agent"]["heatmap"])  # 1 x 12
    im2 = ax.imshow(H2, vmin=0.65, vmax=0.72, cmap="magma", aspect="auto")
    ax.set_yticks([0]); ax.set_yticklabels(["Single agent"])
    ax.set_xticks(range(H2.shape[1])); ax.set_xticklabels([f"R{r}" for r in range(H2.shape[1])])
    for j in range(H2.shape[1]):
        ax.text(j, 0, f"{H2[0,j]:.2f}", ha="center", va="center",
                color="white" if H2[0, j] < 0.69 else "black", fontsize=8)
    ax.set_title("Exp D2 — Correctness AUC, matched single-agent")
    plt.colorbar(im2, ax=ax, label="AUC")
    fig.tight_layout()
    save(fig, "fig_exp_d_trajectory")


def fig_exp_d_intrinsic_dim():
    D = load_json(RESULTS / "exp_d" / "exp_d.json")
    rows = D["D4_intrinsic_dim"]
    tasks = [r["task"] for r in rows]
    cor = [r["intrinsic_dim_correct"] for r in rows]
    inc = [r["intrinsic_dim_incorrect"] for r in rows]
    x = np.arange(len(tasks))
    fig, ax = plt.subplots(figsize=(7, 4))
    w = 0.35
    ax.bar(x - w/2, cor, w, label="Correct", color="#3D5A80")
    ax.bar(x + w/2, inc, w, label="Incorrect", color="#E07A5F")
    ax.set_xticks(x)
    ax.set_xticklabels([TASK_LABEL[t] for t in tasks])
    ax.set_ylabel("Intrinsic dimension (TwoNN)")
    ax.set_title("Exp D4 — Intrinsic dimension of correct vs. incorrect trajectories")
    ax.legend()
    for i, (c, n) in enumerate(zip(cor, inc)):
        ax.text(i - w/2, c + 0.2, f"{c:.1f}", ha="center", fontsize=8)
        ax.text(i + w/2, n + 0.2, f"{n:.1f}", ha="center", fontsize=8)
    save(fig, "fig_exp_d_intrinsic_dim")


def fig_exp_e_role():
    E = load_json(RESULTS / "exp_e" / "exp_e.json")
    # E1 within vs cross + E2 agent-id classifier
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
    ax.set_title("Exp E1 — Within- vs cross-agent similarity")
    ax.legend()

    ax = axes[1]
    accs = [E["latent_mas"][t]["E2_agent_id_classifier"]["accuracy"] * 100 for t in TASKS]
    los = [(E["latent_mas"][t]["E2_agent_id_classifier"]["accuracy"]
            - E["latent_mas"][t]["E2_agent_id_classifier"]["ci_lo"]) * 100 for t in TASKS]
    his = [(E["latent_mas"][t]["E2_agent_id_classifier"]["ci_hi"]
            - E["latent_mas"][t]["E2_agent_id_classifier"]["accuracy"]) * 100 for t in TASKS]
    ax.bar(x, accs, yerr=[los, his], color=[TASK_COLOR[t] for t in TASKS], capsize=4)
    ax.axhline(33.3, color="grey", ls="--", lw=1, label="chance")
    ax.set_xticks(x); ax.set_xticklabels([TASK_LABEL[t] for t in TASKS])
    ax.set_ylim(30, 102)
    ax.set_ylabel("Agent-ID classifier accuracy (%)")
    ax.set_title("Exp E2 — Latent thoughts encode agent identity")
    ax.legend()
    for i, v in enumerate(accs):
        ax.text(i, v + 1, f"{v:.1f}", ha="center", fontsize=9)
    fig.tight_layout()
    save(fig, "fig_exp_e_role")


def fig_exp_e_layer_emergence():
    E = load_json(RESULTS / "exp_e" / "exp_e.json")
    fig, ax = plt.subplots(figsize=(8, 4))
    for t in TASKS:
        curve = E["E4_layer_emergence"][t]
        ax.plot(curve, label=TASK_LABEL[t], color=TASK_COLOR[t])
    ax.set_xlabel("Transformer layer")
    ax.set_ylabel("Agent-ID probe accuracy")
    ax.set_ylim(0.94, 1.0)
    ax.set_title("Exp E4 — Where does role identity emerge?")
    ax.legend()
    save(fig, "fig_exp_e_layer_emergence")


def fig_exp_f_information():
    F = load_json(RESULTS / "exp_f" / "exp_f.json")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax = axes[0]
    ks = [1, 2, 3]
    width = 0.25
    x = np.arange(len(ks))
    for i, t in enumerate(TASKS):
        vals = [F["latent_mas"][t][f"k={k}"]["mi_all"] for k in ks]
        ax.bar(x + (i - 1) * width, vals, width, label=TASK_LABEL[t], color=TASK_COLOR[t])
    ax.set_xticks(x); ax.set_xticklabels([f"k={k}" for k in ks])
    ax.set_ylabel("Kraskov mutual information (nats)")
    ax.set_title("Exp F — MI(latent, label) decays across rounds")
    ax.legend()

    ax = axes[1]
    for t in TASKS:
        curve = F["F4_rate_distortion"][t]["curve"]
        ks = [c["k"] for c in curve]
        aucs = [c["auc"] for c in curve]
        los = [c["auc"] - c["ci_lo"] for c in curve]
        his = [c["ci_hi"] - c["auc"] for c in curve]
        ax.errorbar(ks, aucs, yerr=[los, his], label=TASK_LABEL[t], color=TASK_COLOR[t],
                    marker="o", capsize=2)
    ax.set_xscale("log")
    ax.set_xlabel("k (top-PC retained)")
    ax.set_ylabel("Correctness AUC")
    ax.set_title("Exp F4 — Rate–distortion: AUC vs latent dimensions kept")
    ax.legend()
    fig.tight_layout()
    save(fig, "fig_exp_f_information")


def fig_exp_g_groupthink():
    G = load_json(RESULTS / "exp_g" / "exp_g.json")
    fig, ax = plt.subplots(figsize=(8, 4))
    for t in TASKS:
        rec = G["latent_mas"][t]["G1"]
        rounds = [r["round"] for r in rec]
        means = [r["mean_F"] for r in rec]
        los = [r["mean_F"] - r["ci_lo"] for r in rec]
        his = [r["ci_hi"] - r["mean_F"] for r in rec]
        ax.errorbar(rounds, means, yerr=[los, his], label=TASK_LABEL[t], color=TASK_COLOR[t],
                    marker="o", capsize=3)
    ax.set_xlabel("Round")
    ax.set_ylabel("Mean inter-agent cosine (1 = collapse)")
    ax.set_title("Exp G1 — Inter-agent agreement decreases over rounds")
    ax.legend()
    save(fig, "fig_exp_g_groupthink")


def fig_exp_i_blame():
    I = load_json(RESULTS / "exp_i" / "exp_i.json")
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, t in zip(axes, TASKS):
        H = np.array(I["latent_mas"][t]["I2_blame_distribution"])
        im = ax.imshow(H, cmap="Reds", aspect="auto")
        ax.set_title(f"{TASK_LABEL[t]} (n_fail={I['latent_mas'][t]['n_failures']})")
        ax.set_xticks(range(H.shape[1])); ax.set_xticklabels([f"R{r}" for r in range(H.shape[1])])
        ax.set_yticks(range(H.shape[0])); ax.set_yticklabels([f"Agent {a}" for a in range(H.shape[0])])
        for i in range(H.shape[0]):
            for j in range(H.shape[1]):
                if H[i, j] > 0:
                    ax.text(j, i, f"{H[i,j]:.2f}", ha="center", va="center",
                            color="white" if H[i, j] > 0.3 else "black", fontsize=9)
        plt.colorbar(im, ax=ax, label="Blame fraction")
    fig.suptitle("Exp I2 — Where errors are introduced (blame distribution over (agent, round))")
    fig.tight_layout()
    save(fig, "fig_exp_i_blame")


def fig_exp_j_uncertainty():
    J = load_json(RESULTS / "exp_j" / "exp_j.json")["J2"]
    rows = []
    for t in TASKS:
        for src, key in [("Latent", "auc_latent"), ("Text-regex", "auc_text_regex"),
                         ("Single-agent", "auc_single_agent")]:
            v = J[t][key]
            rows.append({"task": TASK_LABEL[t], "source": src, "auc": v["auc"],
                         "lo": v["auc"] - v["ci_lo"], "hi": v["ci_hi"] - v["auc"]})
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(10, 4))
    sources = ["Latent", "Text-regex", "Single-agent"]
    src_color = {"Latent": "#2E86AB", "Text-regex": "#3D5A80", "Single-agent": "#E07A5F"}
    width = 0.25
    x = np.arange(len(TASKS))
    for i, src in enumerate(sources):
        sub = df[df["source"] == src].set_index("task").reindex([TASK_LABEL[t] for t in TASKS])
        ax.bar(x + (i - 1) * width, sub["auc"].values, width,
               yerr=[sub["lo"].values, sub["hi"].values], capsize=3,
               color=src_color[src], label=src)
    ax.axhline(0.5, color="grey", ls="--", lw=1)
    ax.set_xticks(x); ax.set_xticklabels([TASK_LABEL[t] for t in TASKS])
    ax.set_ylabel("Correctness AUC")
    ax.set_ylim(0.4, 0.8)
    ax.set_title("Exp J2 — Where calibration lives: latent vs text vs single-agent")
    ax.legend()
    save(fig, "fig_exp_j_uncertainty")


def fig_exp_k_redundancy():
    K = load_json(RESULTS / "exp_k" / "exp_k.json")
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for ax, t in zip(axes, TASKS):
        H = np.array(K["latent_mas"][t]["K1_R2_matrix"])
        im = ax.imshow(H, vmin=0.6, vmax=1.0, cmap="Blues", aspect="auto")
        ax.set_title(f"{TASK_LABEL[t]} (mean R²={K['latent_mas'][t]['K1_mean_R2']:.2f})")
        ax.set_xticks(range(3)); ax.set_yticks(range(3))
        ax.set_xticklabels([f"A{i}" for i in range(3)])
        ax.set_yticklabels([f"A{i}" for i in range(3)])
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{H[i,j]:.2f}", ha="center", va="center",
                        color="white" if H[i, j] > 0.85 else "black", fontsize=9)
        plt.colorbar(im, ax=ax, label="R² (predict A_j from A_i)")
    fig.suptitle("Exp K1 — Cross-agent linear redundancy (R²)")
    fig.tight_layout()
    save(fig, "fig_exp_k_redundancy")


def fig_exp_l_subspace():
    L = load_json(RESULTS / "exp_l" / "exp_l.json")
    expvar = np.load(RESULTS / "exp_l" / "expvar_curve.npy")
    cum = np.cumsum(expvar)
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.plot(np.arange(1, len(cum) + 1), cum, color="#2E86AB", lw=2)
    elbow = L["elbow_k"]
    ax.axvline(elbow, color="#E07A5F", ls="--", lw=1.5,
               label=f"elbow k={elbow}  (cum.var={cum[elbow-1]:.2f})")
    ax.axvline(L["D"], color="grey", ls=":", lw=1, label=f"D={L['D']}")
    ax.set_xscale("log")
    ax.set_xlabel("Top-k principal components retained")
    ax.set_ylabel("Cumulative explained variance")
    ax.set_title(f"Exp L — Communication subspace ({L['n_discovery_vectors']:,} discovery vectors)")
    ax.legend()
    save(fig, "fig_exp_l_subspace")


def fig_exp_m_wa_ablation():
    M = load_json(RESULTS / "exp_m" / "exp_m.json")
    fig, ax = plt.subplots(figsize=(10, 4.2))
    width = 0.25
    x = np.arange(len(TASKS))
    series = [
        ("LatentMAS (trained $W_a$)", "latent_mas", "#2E86AB"),
        ("Identity $W_a$", "exp_m_identity_wa", "#3D5A80"),
        ("No transfer", "no_transfer", "#E07A5F"),
    ]
    for i, (label, key, color) in enumerate(series):
        accs = [M[t]["accuracy"][key]["accuracy"] * 100 for t in TASKS]
        los = [(M[t]["accuracy"][key]["accuracy"] - M[t]["accuracy"][key]["ci_lo"]) * 100 for t in TASKS]
        his = [(M[t]["accuracy"][key]["ci_hi"] - M[t]["accuracy"][key]["accuracy"]) * 100 for t in TASKS]
        ax.bar(x + (i - 1) * width, accs, width, yerr=[los, his], capsize=3,
               color=color, label=label)
    ax.set_xticks(x); ax.set_xticklabels([TASK_LABEL[t] for t in TASKS])
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(60, 100)
    ax.set_title("Exp M — $W_a$ decomposition")
    ax.legend(loc="lower left")
    # annotate diff vs trained
    for i, t in enumerate(TASKS):
        delta = M[t]["paired_vs_latent_mas"]["no_transfer"]["diff_pp"]
        p = M[t]["paired_vs_latent_mas"]["no_transfer"]["p_value"]
        ax.text(i, 62, f"Δ no-transfer\n{delta:+.1f}pp  p={p:.3f}", ha="center", fontsize=8,
                color="#E07A5F")
    save(fig, "fig_exp_m_wa_ablation")


def fig_exp_m_flocking():
    M = load_json(RESULTS / "exp_m" / "exp_m.json")
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
    for ax, t in zip(axes, TASKS):
        d = M[t]["M4_flocking"]
        rounds = list(range(4))
        for label, key, color in [
            ("LatentMAS", "latent_mas", "#2E86AB"),
            ("Identity $W_a$", "exp_m_identity_wa", "#3D5A80"),
            ("No transfer", "no_transfer", "#E07A5F"),
        ]:
            ys = [d[key][f"round_{r}"] for r in rounds]
            ax.plot(rounds, ys, marker="o", label=label, color=color)
        ax.set_title(TASK_LABEL[t])
        ax.set_xlabel("Round")
        ax.set_xticks(rounds)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Mean inter-agent cosine F")
    axes[-1].legend(loc="lower left")
    fig.suptitle("Exp M4 — 'Flocking' dynamics: convergence with vs without latent transfer")
    fig.tight_layout()
    save(fig, "fig_exp_m_flocking")


def fig_exp_n_sycophancy():
    N = load_json(RESULTS / "exp_n" / "exp_n.json")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ax = axes[0]
    x = np.arange(len(TASKS))
    width = 0.25
    for a in range(3):
        vals = [N["latent_mas"][t]["N1_dominance_per_agent"][a] for t in TASKS]
        ax.bar(x + (a - 1) * width, vals, width, label=f"Agent {a}",
               color=["#2E86AB", "#3D5A80", "#E07A5F"][a])
    ax.set_xticks(x); ax.set_xticklabels([TASK_LABEL[t] for t in TASKS])
    ax.set_ylabel("Dominance R²")
    ax.set_title("Exp N1 — Inter-agent dominance (R² of being predicted by others)")
    ax.legend()

    ax = axes[1]
    aucs = [N["latent_mas"][t]["N4_sycophancy_direction_auc"]["auc"] for t in TASKS]
    los = [N["latent_mas"][t]["N4_sycophancy_direction_auc"]["auc"]
           - N["latent_mas"][t]["N4_sycophancy_direction_auc"]["ci_lo"] for t in TASKS]
    his = [N["latent_mas"][t]["N4_sycophancy_direction_auc"]["ci_hi"]
           - N["latent_mas"][t]["N4_sycophancy_direction_auc"]["auc"] for t in TASKS]
    ax.bar(x, aucs, yerr=[los, his], capsize=3, color=[TASK_COLOR[t] for t in TASKS])
    ax.axhline(0.5, color="grey", ls="--", lw=1)
    ax.set_xticks(x); ax.set_xticklabels([TASK_LABEL[t] for t in TASKS])
    ax.set_ylim(0.45, 0.78)
    ax.set_ylabel("Sycophancy direction AUC")
    ax.set_title("Exp N4 — Sycophancy direction predicts failure")
    fig.tight_layout()
    save(fig, "fig_exp_n_sycophancy")


def fig_exp_o_layer_routing():
    O = load_json(RESULTS / "exp_o" / "exp_o.json")
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
    ax.set_title("Exp O1 — Per-layer probe AUC for the channel-positive set")
    ax.legend()

    ax = axes[1]
    for t in TASKS:
        adj = O[t]["O3_cka_adjacent"]
        layers = [a["layer_pair"][0] for a in adj]
        ckas = [a["cka"] for a in adj]
        ax.plot(layers, ckas, label=TASK_LABEL[t], color=TASK_COLOR[t], marker="o", ms=3)
    ax.set_xlabel("Layer (pair $\\ell, \\ell+1$)")
    ax.set_ylabel("Adjacent-layer CKA")
    ax.set_title("Exp O3 — Representation similarity across adjacent layers")
    ax.legend()
    fig.tight_layout()
    save(fig, "fig_exp_o_layer_routing")


def fig_exp_p_probe():
    P = load_json(RESULTS / "exp_p" / "exp_p.json")
    items = [("k-PCA probe", P["P2_main"]["auc"]),
             ("Full hidden", P["P3_baselines"]["full_hidden"]["auc"]),
             ("Random k-PCA", P["P3_baselines"]["random_kd"]["auc"]),
             ("Question length", P["P3_baselines"]["question_length"]["auc"]),
             ("Task one-hot", P["P3_baselines"]["task_onehot"]["auc"])]
    fig, ax = plt.subplots(figsize=(8, 4))
    names = [n for n, _ in items]
    aucs = [v["auc"] for _, v in items]
    los = [v["auc"] - v["ci_lo"] for _, v in items]
    his = [v["ci_hi"] - v["auc"] for _, v in items]
    palette = ["#2E86AB", "#3D5A80", "#9DA5BD", "#9DA5BD", "#E07A5F"]
    bars = ax.bar(names, aucs, yerr=[los, his], capsize=3, color=palette)
    ax.axhline(0.5, color="grey", ls="--", lw=1)
    ax.set_ylim(0.45, 0.85)
    ax.set_ylabel("Bucket-1 prediction AUC")
    ax.set_title(f"Exp P — Predictive probe (k={P['k']}, n_pos={P['P2_main']['auc']['n_pos']}, "
                 f"n_neg={P['P2_main']['auc']['n_neg']})")
    plt.xticks(rotation=15, ha="right")
    for bar, v in zip(bars, aucs):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.015, f"{v:.2f}",
                ha="center", fontsize=9)
    save(fig, "fig_exp_p_probe")


def fig_summary_panel():
    """One-shot 'paper-front' figure summarising the headline numbers."""
    df_acc = pd.read_csv(RESULTS / "report" / "accuracy.csv")
    df_cmp = pd.read_csv(RESULTS / "report" / "compute.csv")
    df_cmp["wall_s"] = df_cmp["wall_clock_ms_mean"] / 1000.0
    M = load_json(RESULTS / "exp_m" / "exp_m.json")
    L = load_json(RESULTS / "exp_l" / "exp_l.json")

    fig = plt.figure(figsize=(13, 8.5))
    gs = fig.add_gridspec(2, 3, hspace=0.42, wspace=0.32)

    # Panel A — accuracy bars (LatentMAS vs single-agent vs textmas)
    axA = fig.add_subplot(gs[0, 0])
    headline = ["latent_mas", "single_agent_latent_sampled", "text_mas"]
    width = 0.25
    x = np.arange(len(TASKS))
    for i, c in enumerate(headline):
        sub = df_acc[df_acc["condition"] == c].set_index("task").reindex(TASKS)
        accs = sub["accuracy"].values * 100
        lo = (sub["accuracy"].values - sub["ci_lo"].values) * 100
        hi = (sub["ci_hi"].values - sub["accuracy"].values) * 100
        axA.bar(x + (i - 1) * width, accs, width, yerr=[lo, hi],
                color=["#2E86AB", "#3D5A80", "#E07A5F"][i],
                label=CONDITION_LABEL[c], capsize=2.5)
    axA.set_xticks(x); axA.set_xticklabels([TASK_LABEL[t] for t in TASKS])
    axA.set_ylabel("Accuracy (%)"); axA.set_ylim(60, 100)
    axA.set_title("(A) Accuracy: LatentMAS vs baselines")
    axA.legend(fontsize=8, loc="lower left")

    # Panel B — wall clock
    axB = fig.add_subplot(gs[0, 1])
    for i, c in enumerate(headline):
        sub = df_cmp[df_cmp["condition"] == c].set_index("task").reindex(TASKS)
        axB.bar(x + (i - 1) * width, sub["wall_s"].values, width,
                color=["#2E86AB", "#3D5A80", "#E07A5F"][i],
                label=CONDITION_LABEL[c])
    axB.set_xticks(x); axB.set_xticklabels([TASK_LABEL[t] for t in TASKS])
    axB.set_ylabel("Wall-clock (s/example)")
    axB.set_title("(B) Compute: TextMAS is 3–5× slower")

    # Panel C — token usage
    axC = fig.add_subplot(gs[0, 2])
    for i, c in enumerate(headline):
        sub = df_cmp[df_cmp["condition"] == c].set_index("task").reindex(TASKS)
        toks = sub["generated_tokens_mean"].values
        axC.bar(x + (i - 1) * width, toks, width,
                color=["#2E86AB", "#3D5A80", "#E07A5F"][i],
                label=CONDITION_LABEL[c])
    axC.set_xticks(x); axC.set_xticklabels([TASK_LABEL[t] for t in TASKS])
    axC.set_ylabel("Tokens generated / example")
    axC.set_title("(C) Tokens (TextMAS via vLLM, n/a)")

    # Panel D — flocking
    axD = fig.add_subplot(gs[1, 0])
    for t in TASKS:
        d = M[t]["M4_flocking"]["latent_mas"]
        axD.plot(range(4), [d[f"round_{r}"] for r in range(4)],
                 marker="o", color=TASK_COLOR[t], label=TASK_LABEL[t])
    axD.set_xlabel("Round"); axD.set_xticks(range(4))
    axD.set_ylabel("Mean inter-agent cosine F")
    axD.set_title("(D) Flocking: agents converge over rounds")
    axD.legend(fontsize=8)

    # Panel E — subspace cumvar
    axE = fig.add_subplot(gs[1, 1])
    expvar = np.load(RESULTS / "exp_l" / "expvar_curve.npy")
    cum = np.cumsum(expvar)
    axE.plot(np.arange(1, len(cum) + 1), cum, color="#2E86AB", lw=2)
    axE.axvline(L["elbow_k"], color="#E07A5F", ls="--",
                label=f"elbow k={L['elbow_k']}")
    axE.set_xscale("log")
    axE.set_xlabel("Top-k PCs"); axE.set_ylabel("Cum. explained var")
    axE.set_title(f"(E) Subspace: {L['elbow_k']}/{L['D']} dims")
    axE.legend(fontsize=8)

    # Panel F — Per-layer probe peak (Exp O)
    axF = fig.add_subplot(gs[1, 2])
    O = load_json(RESULTS / "exp_o" / "exp_o.json")
    for t in TASKS:
        per = O[t]["O1_per_layer_auc"]
        axF.plot([p["layer"] for p in per], [p["auc"] for p in per],
                 color=TASK_COLOR[t], label=TASK_LABEL[t], marker="o", ms=2)
    axF.axhline(0.5, color="grey", ls="--", lw=1)
    axF.set_xlabel("Layer"); axF.set_ylabel("Channel-positive AUC")
    axF.set_title("(F) Per-layer probe peaks mid-stack")
    axF.legend(fontsize=8)

    fig.suptitle("LatentMAS: headline accuracy, efficiency, and mechanism", fontsize=13, y=1.01)
    save(fig, "fig_summary_panel")


# ===========================================================================
# Run
# ===========================================================================

if __name__ == "__main__":
    print("# Tables")
    make_main_results_table()
    make_compute_table()
    make_buckets_table()
    make_ablation_table()

    print("\n# Figures")
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
    fig_summary_panel()
    print("\nDone.")
