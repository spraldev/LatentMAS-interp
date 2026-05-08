"""Paper-quality architecture diagrams for LatentMAS, TextMAS, single-agent.

Built with matplotlib patches so they're vector-format (PDF) and reproducible.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from matplotlib.path import Path as MplPath

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "paper" / "diagrams"
OUT.mkdir(parents=True, exist_ok=True)

# Palette — paper-friendly muted palette
C_AGENT = "#2E86AB"          # planner / agents (blue)
C_AGENT_2 = "#5B8DBE"
C_AGENT_3 = "#88B0CE"
C_JUDGE = "#F4A261"          # judger (warm orange)
C_KV = "#264653"             # KV / latent (dark teal)
C_TEXT = "#E76F51"           # text channel (coral)
C_WA = "#9C46B5"             # W_a alignment (purple)
C_BG = "#F7F7F2"             # canvas background
C_OUTLINE = "#1A1A1A"
C_MUTED = "#7F8C8D"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.spines.left": False, "axes.spines.bottom": False,
})


def round_box(ax, x, y, w, h, label, fc, ec=C_OUTLINE, fontsize=10, fontcolor="white",
              fontweight="bold", radius=0.04, lw=1.0):
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.012,rounding_size={radius}",
        linewidth=lw, edgecolor=ec, facecolor=fc,
    )
    ax.add_patch(box)
    ax.text(x + w / 2, y + h / 2, label,
            ha="center", va="center",
            fontsize=fontsize, fontweight=fontweight, color=fontcolor)


def tag(ax, x, y, txt, color="#444", fontsize=8, ha="center"):
    ax.text(x, y, txt, ha=ha, va="center", fontsize=fontsize, color=color, style="italic")


def arrow(ax, x1, y1, x2, y2, color=C_KV, lw=1.5, style="->", curve=0.0, label=None,
          label_offset=(0, 0.04), label_color=None):
    ap = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle=style, mutation_scale=12,
        color=color, lw=lw,
        connectionstyle=f"arc3,rad={curve}",
    )
    ax.add_patch(ap)
    if label:
        mx = (x1 + x2) / 2 + label_offset[0]
        my = (y1 + y2) / 2 + label_offset[1]
        ax.text(mx, my, label, fontsize=8, ha="center",
                color=label_color or color, fontweight="bold")


def setup(ax, w=10, h=5.5, title=None):
    ax.set_xlim(0, w); ax.set_ylim(0, h)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_facecolor(C_BG)
    if title:
        ax.text(w / 2, h - 0.2, title, ha="center", va="top",
                fontsize=13, fontweight="bold", color="#222")


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", dpi=240)
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight", dpi=240)
    plt.close(fig)
    print(f"  ✓ diagrams/{name}.{{pdf,png}}")


# ---------------------------------------------------------------------------
# 1. LatentMAS architecture
# ---------------------------------------------------------------------------

def diagram_latentmas():
    fig, ax = plt.subplots(figsize=(14, 6.4))
    setup(ax, w=14, h=6.4, title="LatentMAS — agents communicate by passing KV / hidden-state through a shared working memory")

    # Question
    round_box(ax, 0.2, 3.0, 1.2, 0.9, "Question", "#34495E", radius=0.05)

    agents = [
        ("Planner", 1.9, C_AGENT),
        ("Critic", 5.5, C_AGENT_2),
        ("Refiner", 9.1, C_AGENT_3),
    ]

    # Agent boxes
    agent_w, agent_h = 2.8, 3.0
    for name, x, color in agents:
        round_box(ax, x, 1.6, agent_w, agent_h, "", color, radius=0.06, lw=1.2)
        ax.text(x + agent_w / 2, 4.2, name, ha="center", va="center",
                fontsize=12, fontweight="bold", color="white")
        ax.text(x + agent_w / 2, 3.75, "L latent rollout steps",
                ha="center", va="center", fontsize=8.5, color="white", style="italic")
        # latent step pellets row
        for i in range(6):
            lx = x + 0.18 + i * 0.42
            ax.add_patch(Rectangle((lx, 2.85), 0.34, 0.55,
                                   facecolor="white", edgecolor="#1A1A1A", lw=0.6))
            ax.text(lx + 0.17, 3.13, f"h{i+1}", ha="center", va="center",
                    fontsize=7.2, color="#333")
        # working memory bar
        ax.add_patch(Rectangle((x + 0.18, 1.95), agent_w - 0.36, 0.65,
                               facecolor="#1F3A4D", edgecolor="white", lw=0.8))
        ax.text(x + agent_w / 2, 2.27, "shared KV cache",
                ha="center", va="center", fontsize=8.5, color="white",
                fontweight="bold")

    # Wa boxes between agents
    wa_xs = [4.85, 8.45]   # midway between agent boundaries
    for x in wa_xs:
        round_box(ax, x, 2.95, 0.55, 0.55, "$W_a$", C_WA, fontsize=11, radius=0.08)
        ax.text(x + 0.275, 2.65, "align", color=C_WA, fontsize=7.5,
                ha="center", style="italic")

    # Question -> Planner
    arrow(ax, 1.4, 3.4, 1.9, 3.4, color=C_KV, lw=1.8)
    # Inter-agent arrows: out of agent edge → W_a → into next agent
    arrow(ax, 4.7, 3.22, 4.85, 3.22, color=C_KV, lw=1.8)
    arrow(ax, 5.4, 3.22, 5.5, 3.22, color=C_KV, lw=1.8)
    arrow(ax, 8.3, 3.22, 8.45, 3.22, color=C_KV, lw=1.8)
    arrow(ax, 9.0, 3.22, 9.1, 3.22, color=C_KV, lw=1.8)

    # KV channel labels
    ax.text(5.125, 3.55, "KV", color=C_KV, fontsize=8.5, ha="center", fontweight="bold")
    ax.text(8.725, 3.55, "KV", color=C_KV, fontsize=8.5, ha="center", fontweight="bold")

    # Judger
    round_box(ax, 12.05, 2.6, 1.65, 1.5, "Judger\n(decode)", C_JUDGE,
              fontsize=10, radius=0.06)
    arrow(ax, 11.9, 3.22, 12.05, 3.22, color=C_KV, lw=1.8)
    # Answer
    round_box(ax, 12.25, 0.95, 1.1, 0.75, "Answer", "#34495E", radius=0.05)
    arrow(ax, 12.85, 2.6, 12.85, 1.7, color=C_TEXT, lw=1.6,
          label="text", label_offset=(0.18, 0.0), label_color=C_TEXT)

    # Legend
    legend_y = 0.30
    ax.text(0.3, legend_y + 0.30, "Channels:", fontsize=10, fontweight="bold", color="#222")
    ax.add_patch(Rectangle((1.6, legend_y), 0.45, 0.18, facecolor=C_KV))
    ax.text(2.15, legend_y + 0.09, "KV / latent (no decoding)", fontsize=9, va="center")
    ax.add_patch(Rectangle((6.0, legend_y), 0.45, 0.18, facecolor=C_WA))
    ax.text(6.55, legend_y + 0.09, "$W_a$  alignment (latent → embedding)", fontsize=9, va="center")
    ax.add_patch(Rectangle((10.6, legend_y), 0.45, 0.18, facecolor=C_TEXT))
    ax.text(11.15, legend_y + 0.09, "Text (only final answer)", fontsize=9, va="center")

    save(fig, "arch_latentmas")


# ---------------------------------------------------------------------------
# 2. TextMAS architecture
# ---------------------------------------------------------------------------

def diagram_textmas():
    fig, ax = plt.subplots(figsize=(13, 5.6))
    setup(ax, w=13, h=5.6, title="TextMAS — agents communicate by writing and re-reading natural-language traces")

    round_box(ax, 0.2, 2.4, 1.2, 0.8, "Question", "#34495E", radius=0.05)

    agents = [
        ("Planner", 2.0, C_AGENT),
        ("Critic", 5.2, C_AGENT_2),
        ("Refiner", 8.4, C_AGENT_3),
    ]

    for name, x, color in agents:
        round_box(ax, x, 1.4, 2.6, 2.6, "", color, radius=0.06, lw=1.2)
        ax.text(x + 1.3, 3.7, name, ha="center", va="center",
                fontsize=11, fontweight="bold", color="white")
        # text pellets (decoded tokens)
        for i, ty in enumerate([3.05, 2.65, 2.25]):
            ax.add_patch(Rectangle((x + 0.25, ty), 2.1, 0.30,
                                   facecolor="white", edgecolor="#1A1A1A", lw=0.6))
            sample = ["Tokens t_1 t_2 ... t_N", "decode → write trace", "re-encode for next agent"][i]
            ax.text(x + 1.3, ty + 0.15, sample, ha="center", va="center",
                    fontsize=8, color="#333")
        ax.text(x + 1.3, 1.75, "Each turn: full decode → text",
                ha="center", va="center", fontsize=8.5, color="white",
                fontweight="bold", style="italic")

    # Question -> Planner
    arrow(ax, 1.4, 2.8, 2.0, 2.8, color=C_TEXT)

    # Inter-agent: text channel
    arrow(ax, 4.6, 2.8, 5.2, 2.8, color=C_TEXT, lw=2.0,
          label="text", label_offset=(0, 0.18), label_color=C_TEXT)
    arrow(ax, 7.8, 2.8, 8.4, 2.8, color=C_TEXT, lw=2.0,
          label="text", label_offset=(0, 0.18), label_color=C_TEXT)

    # Judger
    round_box(ax, 11.05, 2.0, 1.7, 1.3, "Judger\n(decode)", C_JUDGE,
              fontsize=10, radius=0.06)
    arrow(ax, 11.0, 2.8, 11.05, 2.65, color=C_TEXT)

    round_box(ax, 11.6, 0.6, 1.1, 0.7, "Answer", "#34495E", radius=0.05)
    arrow(ax, 11.9, 2.0, 12.15, 1.3, color=C_TEXT, lw=1.5)

    # Legend
    ax.text(0.3, 0.85, "Channel legend:", fontsize=9, fontweight="bold", color="#222")
    ax.add_patch(Rectangle((0.3, 0.55), 0.4, 0.16, facecolor=C_TEXT))
    ax.text(0.78, 0.63, "Text (every turn — long traces, slow decoding)", fontsize=8.5, va="center")

    # contrast call-out
    ax.text(6.5, 0.40,
            "Contrast: every inter-agent edge requires full token decoding before re-tokenisation.",
            ha="center", fontsize=8.5, color="#555", style="italic")

    save(fig, "arch_textmas")


# ---------------------------------------------------------------------------
# 3. Single-agent baseline
# ---------------------------------------------------------------------------

def diagram_single_agent():
    fig, ax = plt.subplots(figsize=(13, 5.0))
    setup(ax, w=13, h=5.0, title="Single-Agent (matched-budget baseline) — one agent, longer latent rollout, no inter-agent transfer")

    round_box(ax, 0.4, 2.0, 1.4, 0.9, "Question", "#34495E", radius=0.05)

    # one big agent block
    round_box(ax, 2.4, 0.9, 8.5, 3.2, "", C_AGENT, radius=0.05, lw=1.2)
    ax.text(6.65, 3.85, "Single agent (Qwen3-4B)",
            ha="center", va="center", fontsize=11, fontweight="bold", color="white")

    # 12 latent step pellets
    for i in range(12):
        x0 = 2.7 + i * 0.65
        ax.add_patch(Rectangle((x0, 2.05), 0.55, 0.55,
                               facecolor="white", edgecolor="#1A1A1A", lw=0.6))
        ax.text(x0 + 0.275, 2.32, f"h{i+1}", ha="center", va="center",
                fontsize=7, color="#333")

    ax.text(6.65, 1.55, "12 sequential latent steps  (matches LatentMAS total: 3 agents × 4 rounds)",
            ha="center", va="center", fontsize=9, color="white", style="italic")

    arrow(ax, 1.8, 2.45, 2.4, 2.45, color=C_KV)

    round_box(ax, 11.2, 1.85, 1.4, 1.15, "Decode\nanswer", C_JUDGE, fontsize=10, radius=0.06)
    arrow(ax, 10.9, 2.45, 11.2, 2.45, color=C_KV)
    round_box(ax, 11.5, 0.45, 0.9, 0.65, "Answer", "#34495E", radius=0.05)
    arrow(ax, 11.9, 1.85, 11.95, 1.1, color=C_TEXT, lw=1.5)

    # Note
    ax.text(6.65, 0.35,
            "Same total latent compute and same final-decode budget as LatentMAS, with the only difference being agent multiplicity.",
            ha="center", fontsize=8.5, color="#555", style="italic")

    save(fig, "arch_single_agent")


# ---------------------------------------------------------------------------
# 4. W_a / KV mechanistic decomposition
# ---------------------------------------------------------------------------

def diagram_wa_decomposition():
    fig, ax = plt.subplots(figsize=(13, 5.4))
    setup(ax, w=13, h=5.4, title="What carries the signal? Decomposing the inter-agent channel")

    # Three sub-panels: trained W_a, identity W_a, no transfer
    panels = [
        ("LatentMAS\n(trained $W_a$)", 0.5, C_AGENT,
         True, True, "Both KV and $W_a$ active"),
        ("Identity $W_a$\n(Exp M)", 4.7, C_AGENT_2,
         True, False, "KV passes; $W_a$ is identity (no learned alignment)"),
        ("No transfer\n(matched single-agent)", 8.9, C_AGENT_3,
         False, False, "Each agent re-starts; channel removed entirely"),
    ]

    for name, x0, color, kv_on, wa_on, sub in panels:
        # 3 stacked agents
        for i, y in enumerate([3.0, 2.0, 1.0]):
            round_box(ax, x0, y, 1.0, 0.7, f"A{i+1}", color, fontsize=10, radius=0.06)
        # KV connectors
        if kv_on:
            arrow(ax, x0 + 0.5, 3.0, x0 + 0.5, 2.7, color=C_KV)
            arrow(ax, x0 + 0.5, 2.0, x0 + 0.5, 1.7, color=C_KV)
        else:
            # dashed crossed
            ax.plot([x0 + 0.5, x0 + 0.5], [3.0, 2.7], color=C_MUTED, ls=":", lw=1)
            ax.plot([x0 + 0.5, x0 + 0.5], [2.0, 1.7], color=C_MUTED, ls=":", lw=1)
            ax.text(x0 + 0.7, 2.85, "✕", color="#C0392B", fontweight="bold", fontsize=12)
            ax.text(x0 + 0.7, 1.85, "✕", color="#C0392B", fontweight="bold", fontsize=12)
        # Wa indicator on the connectors
        if kv_on:
            wa_color = C_WA if wa_on else "#BDC3C7"
            wa_label = "$W_a$" if wa_on else "$I$"
            round_box(ax, x0 + 1.05, 2.7, 0.4, 0.4, wa_label, wa_color,
                      fontsize=9, radius=0.06)
            round_box(ax, x0 + 1.05, 1.7, 0.4, 0.4, wa_label, wa_color,
                      fontsize=9, radius=0.06)
            ax.plot([x0 + 0.5, x0 + 1.05], [2.85, 2.9], color=C_KV, lw=1)
            ax.plot([x0 + 0.5, x0 + 1.05], [1.85, 1.9], color=C_KV, lw=1)

        # Sub label
        ax.text(x0 + 0.5, 3.95, name,
                ha="center", va="center", fontsize=10, fontweight="bold", color="#222")
        ax.text(x0 + 0.5, 0.6, sub, ha="center", va="center",
                fontsize=8, color="#555", style="italic")

    # Result strip
    ax.add_patch(Rectangle((0.4, 4.55), 12.2, 0.55, facecolor="white", edgecolor="#888", lw=0.7))
    ax.text(0.55, 4.83, "Empirical decomposition (Exp M, GSM8K / ARC / MBPP+):",
            fontsize=9, fontweight="bold", color="#222", va="center")
    ax.text(7.6, 4.83,
            "Trained $W_a$  ≈  Identity $W_a$  on GSM8K & ARC;  +5.8pp on MBPP+ (McNemar p=0.010)",
            fontsize=9, color=C_WA, fontweight="bold", va="center")

    save(fig, "arch_wa_decomposition")


# ---------------------------------------------------------------------------
# 5. Side-by-side trio for the paper opener
# ---------------------------------------------------------------------------

def diagram_overview():
    fig, axes = plt.subplots(3, 1, figsize=(12, 10.5))

    # use simpler in-axes drawings
    def panel_lmas(ax):
        setup(ax, w=12, h=3.4, title=None)
        ax.text(6, 3.1, "(a) LatentMAS — KV channel, no decode between agents",
                ha="center", va="center", fontsize=11, fontweight="bold", color="#222")
        round_box(ax, 0.2, 1.4, 1.0, 0.8, "x", "#34495E", radius=0.05)
        for i, x in enumerate([1.6, 4.6, 7.6]):
            color = [C_AGENT, C_AGENT_2, C_AGENT_3][i]
            round_box(ax, x, 0.9, 2.6, 1.6, ["Planner", "Critic", "Refiner"][i],
                      color, fontsize=10, radius=0.06)
            for k in range(5):
                ax.add_patch(Rectangle((x + 0.2 + k * 0.45, 1.1), 0.35, 0.35,
                                       facecolor="white", edgecolor="#1A1A1A", lw=0.5))
        # arrows w/ Wa
        for x in [4.2, 7.2]:
            round_box(ax, x, 1.55, 0.4, 0.4, "$W_a$", C_WA, fontsize=8, radius=0.07)
        for (a, b) in [(1.2, 1.6), (4.2, 4.6), (4.2, 4.6), (7.2, 7.6)]:
            arrow(ax, a, 1.7, b, 1.7, color=C_KV, lw=1.6)
        round_box(ax, 10.5, 1.4, 1.3, 0.9, "Judger", C_JUDGE, fontsize=10, radius=0.06)
        arrow(ax, 10.2, 1.85, 10.5, 1.85, color=C_KV)
        ax.text(11.15, 0.95, "answer", color=C_TEXT, fontsize=8.5, ha="center")
        ax.text(0.6, 0.55, "Final text only at the very end →",
                color="#444", fontsize=8.5, style="italic")

    def panel_textmas(ax):
        setup(ax, w=12, h=3.4, title=None)
        ax.text(6, 3.1, "(b) TextMAS — text channel between every agent (re-decoded each turn)",
                ha="center", va="center", fontsize=11, fontweight="bold", color="#222")
        round_box(ax, 0.2, 1.4, 1.0, 0.8, "x", "#34495E", radius=0.05)
        for i, x in enumerate([1.6, 4.6, 7.6]):
            color = [C_AGENT, C_AGENT_2, C_AGENT_3][i]
            round_box(ax, x, 0.9, 2.6, 1.6, ["Planner", "Critic", "Refiner"][i],
                      color, fontsize=10, radius=0.06)
            ax.add_patch(Rectangle((x + 0.2, 1.05), 2.2, 0.5, facecolor="white",
                                   edgecolor="#1A1A1A", lw=0.5))
            ax.text(x + 1.3, 1.30, "decoded text trace", ha="center", va="center",
                    fontsize=8, color="#333")
        for (a, b) in [(1.2, 1.6), (4.2, 4.6), (7.2, 7.6)]:
            arrow(ax, a, 1.7, b, 1.7, color=C_TEXT, lw=2.0,
                  label="text", label_offset=(0, 0.18), label_color=C_TEXT)
        round_box(ax, 10.5, 1.4, 1.3, 0.9, "Judger", C_JUDGE, fontsize=10, radius=0.06)
        arrow(ax, 10.2, 1.85, 10.5, 1.85, color=C_TEXT, lw=2.0)

    def panel_single(ax):
        setup(ax, w=12, h=3.4, title=None)
        ax.text(6, 3.1, "(c) Single-Agent baseline — one agent, 12 sequential latent steps",
                ha="center", va="center", fontsize=11, fontweight="bold", color="#222")
        round_box(ax, 0.2, 1.4, 1.0, 0.8, "x", "#34495E", radius=0.05)
        round_box(ax, 1.6, 0.9, 8.6, 1.6, "single agent", C_AGENT, fontsize=10, radius=0.06)
        for k in range(12):
            ax.add_patch(Rectangle((1.8 + k * 0.7, 1.1), 0.6, 0.6,
                                   facecolor="white", edgecolor="#1A1A1A", lw=0.5))
            ax.text(2.1 + k * 0.7, 1.4, f"h{k+1}", ha="center", va="center",
                    fontsize=6.8, color="#333")
        round_box(ax, 10.5, 1.4, 1.3, 0.9, "Judger", C_JUDGE, fontsize=10, radius=0.06)
        arrow(ax, 1.2, 1.7, 1.6, 1.7, color=C_KV, lw=1.6)
        arrow(ax, 10.2, 1.85, 10.5, 1.85, color=C_KV)

    panel_lmas(axes[0])
    panel_textmas(axes[1])
    panel_single(axes[2])
    fig.tight_layout()
    save(fig, "arch_overview_trio")


if __name__ == "__main__":
    print("# Architecture diagrams")
    diagram_latentmas()
    diagram_textmas()
    diagram_single_agent()
    diagram_wa_decomposition()
    diagram_overview()
    print("Done.")
