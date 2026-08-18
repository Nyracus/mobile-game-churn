"""Generate the research workflow image used by the written report."""

from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "reports" / "figures" / "research_workflow.png"


def main() -> None:
    steps = [
        ("Raw logs", "2.19M attempts"),
        ("Clean + join", "TSV parsing\nlevel metadata"),
        ("Aggregate", "41 player\nfeatures"),
        ("Tune", "4 models\n5-fold CV"),
        ("Select", "OOF threshold\nAUROC"),
        ("Evaluate", "Dev holdout\nSHAP + errors"),
    ]
    fig, ax = plt.subplots(figsize=(12, 2.5))
    ax.set_xlim(0, len(steps))
    ax.set_ylim(0, 1)
    ax.axis("off")
    for index, (title, detail) in enumerate(steps):
        x = index + 0.5
        if index:
            ax.annotate("", xy=(x - 0.42, 0.5), xytext=(x - 0.58, 0.5), arrowprops={"arrowstyle": "->", "color": "#3D8DFF", "lw": 2})
        rect = plt.Rectangle((x - 0.38, 0.20), 0.76, 0.60, facecolor="#E8EEF5", edgecolor="#2E74B5", linewidth=1.2)
        ax.add_patch(rect)
        ax.text(x, 0.59, title, ha="center", va="center", fontsize=11, fontweight="bold", color="#1F4D78")
        ax.text(x, 0.38, detail, ha="center", va="center", fontsize=8.5, color="#303841")
    fig.tight_layout(pad=0.2)
    fig.savefig(OUTPUT, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    main()
