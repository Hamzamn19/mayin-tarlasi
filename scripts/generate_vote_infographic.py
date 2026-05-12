from __future__ import annotations

import os

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Circle


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "plots")
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "vote_flow_infographic.png")


def add_box(ax, xy, width, height, title, body, facecolor, edgecolor, title_color="white"):
    box = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.02,rounding_size=0.05",
        linewidth=2.5,
        facecolor=facecolor,
        edgecolor=edgecolor,
        zorder=2,
    )
    ax.add_patch(box)
    x, y = xy
    ax.text(x + width / 2, y + height * 0.72, title, ha="center", va="center", fontsize=18, fontweight="bold", color=title_color)
    ax.text(x + width / 2, y + height * 0.36, body, ha="center", va="center", fontsize=11.5, color="#e6edf7", linespacing=1.35)


def add_arrow(ax, start, end, color, text=None, text_pos=0.5, rad=0.0):
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="Simple,head_width=14,head_length=14,tail_width=4",
        connectionstyle=f"arc3,rad={rad}",
        linewidth=0,
        facecolor=color,
        edgecolor=color,
        alpha=0.95,
        zorder=1,
    )
    ax.add_patch(arrow)
    if text:
        tx = start[0] + (end[0] - start[0]) * text_pos
        ty = start[1] + (end[1] - start[1]) * text_pos + (0.03 if rad >= 0 else -0.03)
        ax.text(tx, ty, text, ha="center", va="center", fontsize=10.5, fontweight="bold", color="#0b1020",
                bbox=dict(boxstyle="round,pad=0.28", facecolor="white", edgecolor="none", alpha=0.95))


def add_pulse(ax, center, radius, color, alpha=0.15):
    for scale, a in [(1.0, alpha), (1.3, alpha * 0.65), (1.65, alpha * 0.3)]:
        ax.add_patch(Circle(center, radius * scale, facecolor=color, edgecolor="none", alpha=a, zorder=0))


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    fig = plt.figure(figsize=(18, 10), dpi=300)
    ax = plt.gca()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Background layers
    fig.patch.set_facecolor("#07111f")
    ax.set_facecolor("#07111f")
    ax.add_patch(FancyBboxPatch((0.02, 0.03), 0.96, 0.94, boxstyle="round,pad=0.015,rounding_size=0.03",
                                facecolor="#0b1526", edgecolor="#1e3556", linewidth=2.5, zorder=0))
    for x in [0.06, 0.18, 0.30, 0.42, 0.58, 0.72, 0.86]:
        ax.plot([x, x + 0.06], [0.08, 0.92], color="white", alpha=0.03, linewidth=14, zorder=0)

    ax.text(0.5, 0.93, "DECISION FLOW: YOLO26  →  RF  →  VOTE  →  FINAL LABEL",
            ha="center", va="center", fontsize=24, fontweight="bold", color="white")
    ax.text(0.5, 0.885, "A visual explanation of how the hybrid detector turns candidate regions into a final mine decision",
            ha="center", va="center", fontsize=12.5, color="#bfd0ea")

    # Stage labels
    add_pulse(ax, (0.16, 0.55), 0.085, "#4fc3f7")
    add_pulse(ax, (0.50, 0.55), 0.085, "#7ee787")
    add_pulse(ax, (0.84, 0.55), 0.085, "#ffb86b")

    add_box(
        ax,
        (0.07, 0.40),
        0.18,
        0.30,
        "1. YOLO26",
        "Finds candidate mine regions\nOutputs bounding boxes + confidence\nKeeps recall high at the first gate",
        facecolor="#12355f",
        edgecolor="#4fc3f7",
    )
    add_box(
        ax,
        (0.36, 0.40),
        0.28,
        0.30,
        "2. Random Forest",
        "Reads thermal + geometric features\nVerifies whether the crop looks like a real mine\nReduces false alarms from raw detections",
        facecolor="#123f2e",
        edgecolor="#7ee787",
    )
    add_box(
        ax,
        (0.74, 0.40),
        0.19,
        0.30,
        "3. Vote",
        "Combines YOLO confidence\nwith RF probability\nApplies the final decision rule",
        facecolor="#4a2b12",
        edgecolor="#ffb86b",
    )

    # Arrows and callouts
    add_arrow(ax, (0.25, 0.55), (0.36, 0.55), "#4fc3f7", text="candidate crops")
    add_arrow(ax, (0.64, 0.55), (0.74, 0.55), "#7ee787", text="probability check")
    add_arrow(ax, (0.93, 0.55), (0.98, 0.55), "#ffb86b", text="final label", text_pos=0.3)

    # Bottom logic strip
    ax.add_patch(FancyBboxPatch((0.08, 0.14), 0.84, 0.16, boxstyle="round,pad=0.02,rounding_size=0.03",
                                facecolor="#0f1d33", edgecolor="#27486f", linewidth=2))
    ax.text(0.5, 0.255, "Voting rule in deployment", ha="center", va="center", fontsize=14, fontweight="bold", color="white")
    ax.text(0.5, 0.205, "final_score = (YOLO_confidence + RF_probability) / 2",
            ha="center", va="center", fontsize=13.5, color="#d6e6ff", fontfamily="monospace")
    ax.text(0.5, 0.165, "if final_score >= 0.50  →  mine detected  |  else  →  reject candidate",
            ha="center", va="center", fontsize=12.5, color="#9ad0ff")

    # Final badge
    ax.add_patch(FancyBboxPatch((0.37, 0.74), 0.26, 0.08, boxstyle="round,pad=0.02,rounding_size=0.04",
                                facecolor="#162742", edgecolor="#79b8ff", linewidth=1.8))
    ax.text(0.50, 0.779, "High recall first, precise decision second", ha="center", va="center", fontsize=11.5, color="#e9f3ff")

    plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()