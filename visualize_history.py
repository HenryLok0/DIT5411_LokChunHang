"""
Plot training history and model comparison.

Usage:
    python visualize_history.py --model a
    python visualize_history.py --compare
"""

import argparse
import json
import os

import matplotlib.pyplot as plt

import config


def plot_history(model_name: str):
    """Plot accuracy and loss curves for a single model."""
    history_path = os.path.join(config.RESULTS_DIR, f"history_{model_name}.json")
    if not os.path.exists(history_path):
        print(f"History file not found: {history_path}")
        return

    with open(history_path, "r", encoding="utf-8") as f:
        history = json.load(f)

    epochs = range(1, len(history.get("loss", [])) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Loss
    ax = axes[0]
    ax.plot(epochs, history["loss"], label="train")
    ax.plot(epochs, history["val_loss"], label="validation")
    ax.set_title(f"Model {model_name.upper()} - Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()

    # Accuracy
    ax = axes[1]
    ax.plot(epochs, history["accuracy"], label="train top-1")
    ax.plot(epochs, history["val_accuracy"], label="validation top-1")
    if "val_top5_accuracy" in history:
        ax.plot(epochs, history["val_top5_accuracy"], label="validation top-5")
    ax.set_title(f"Model {model_name.upper()} - Accuracy")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.legend()

    out_path = os.path.join(config.RESULTS_DIR, f"history_{model_name}.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved plot to {out_path}")
    plt.close()


def plot_comparison():
    """Plot a bar chart comparing test accuracies of all models."""
    comparison_path = os.path.join(config.RESULTS_DIR, "model_comparison.json")
    if not os.path.exists(comparison_path):
        print("Run evaluate.py --compare first to generate model_comparison.json")
        return

    with open(comparison_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    models = [row["model"].upper() for row in data]
    top1 = [row["accuracy"] for row in data]
    top5 = [row["top5_accuracy"] for row in data]

    x = range(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar([i - width / 2 for i in x], top1, width, label="Top-1 Accuracy")
    ax.bar([i + width / 2 for i in x], top5, width, label="Top-5 Accuracy")

    ax.set_ylabel("Accuracy")
    ax.set_title("Model Comparison on Test Set")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()
    ax.set_ylim(0, 1.05)

    for i, (v1, v5) in enumerate(zip(top1, top5)):
        ax.text(i - width / 2, v1 + 0.02, f"{v1:.3f}", ha="center", va="bottom")
        ax.text(i + width / 2, v5 + 0.02, f"{v5:.3f}", ha="center", va="bottom")

    out_path = os.path.join(config.RESULTS_DIR, "model_comparison.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved comparison plot to {out_path}")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize training history")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--model", type=str, default="a", choices=["a", "b", "c"])
    group.add_argument("--compare", action="store_true")
    args = parser.parse_args()

    if args.compare:
        plot_comparison()
    else:
        plot_history(args.model)
