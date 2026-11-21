"""Save a comparison of training curves and held-out evaluation scores."""

import argparse
import json

import matplotlib.pyplot as plt
import numpy as np

import config


def plot(smoke):
    root = config.RUN_ROOT / config.scope(smoke)
    comparison = json.loads((root / "model_comparison.json").read_text(encoding="utf-8"))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    for name in config.MODELS:
        state = json.loads((root / name / "state.json").read_text(encoding="utf-8"))
        train_top1 = [row["accuracy"] * 100 for row in state["history"]]
        axes[0].plot(range(1, len(train_top1) + 1), train_top1, label=name.upper(), marker="o")
    axes[0].set(title="Augmented training samples", xlabel="Epoch", ylabel="Top-1 accuracy (%)")
    axes[0].legend(title="Model")
    rows = comparison["results"]
    x = np.arange(len(rows))
    for offset, metric, label in [(-0.18, "accuracy", "Top-1"), (0.18, "top5_accuracy", "Top-5")]:
        bars = axes[1].bar(x + offset, [row[metric] * 100 for row in rows], 0.34, label=label)
        axes[1].bar_label(bars, fmt="%.1f", padding=2)
    axes[1].set(title="Untouched test samples", ylabel="Recognition accuracy (%)", ylim=(0, 110))
    axes[1].set_xticks(x, [row["model"].upper() for row in rows])
    axes[1].legend()
    fig.suptitle("32-class development smoke run" if smoke else "Full 13,065-class experiment")
    fig.savefig(root / "model_comparison.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    plot(parser.parse_args().smoke)
