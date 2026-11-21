"""
Evaluate a trained model on the test set and compare multiple models.

Usage:
    python evaluate.py --model a
    python evaluate.py --compare  # evaluates a, b and c checkpoints
"""

import argparse
import json
import os

import numpy as np
import tensorflow as tf
from tensorflow import keras

import config
from models import compile_model, get_model
from utils import configure_for_performance, decode_image, discover_classes


def create_test_dataset(data_dir: str):
    """Build a tf.data.Dataset for evaluation."""
    classes = discover_classes(data_dir)
    label_map = {name: idx for idx, name in enumerate(classes)}

    file_paths = []
    labels = []
    for char in classes:
        char_dir = os.path.join(data_dir, char)
        for img_name in os.listdir(char_dir):
            if img_name.lower().endswith((".png", ".jpg", ".jpeg")):
                file_paths.append(os.path.join(char_dir, img_name))
                labels.append(label_map[char])

    ds = tf.data.Dataset.from_tensor_slices((np.array(file_paths), np.array(labels, dtype=np.int32)))
    ds = ds.map(decode_image, num_parallel_calls=tf.data.AUTOTUNE)
    return ds


def evaluate_model(model_name: str, batch_size: int = config.BATCH_SIZE):
    """Load the best checkpoint for a model and evaluate it on the test set."""
    checkpoint_path = os.path.join(
        config.MODELS_DIR, f"best_model_{model_name}.keras"
    )
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found: {checkpoint_path}")
        return None

    # Number of classes is read from the saved model architecture
    model = keras.models.load_model(checkpoint_path)
    model.summary()

    test_ds = create_test_dataset(config.TEST_DIR)
    test_ds = configure_for_performance(test_ds, batch_size=batch_size, shuffle=False)

    results = model.evaluate(test_ds, return_dict=True)
    print(f"Model {model_name} test results: {results}")

    out_path = os.path.join(config.RESULTS_DIR, f"test_results_{model_name}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    return results


def compare_models(batch_size: int = config.BATCH_SIZE):
    """Evaluate all three model checkpoints and print a comparison table."""
    summary = []
    for name in ["a", "b", "c"]:
        res = evaluate_model(name, batch_size=batch_size)
        if res:
            summary.append(
                {
                    "model": name,
                    "loss": res.get("loss"),
                    "accuracy": res.get("accuracy"),
                    "top5_accuracy": res.get("top5_accuracy"),
                }
            )

    print("\n=== Model Comparison ===")
    print(f"{'Model':<8} {'Loss':<12} {'Top-1 Acc':<12} {'Top-5 Acc':<12}")
    for row in summary:
        print(
            f"{row['model']:<8} {row['loss']:<12.4f} "
            f"{row['accuracy']:<12.4f} {row['top5_accuracy']:<12.4f}"
        )

    summary_path = os.path.join(config.RESULTS_DIR, "model_comparison.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained models")
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--model",
        type=str,
        default="a",
        choices=["a", "b", "c"],
        help="Evaluate a single model",
    )
    group.add_argument(
        "--compare",
        action="store_true",
        help="Evaluate and compare all three models",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=config.BATCH_SIZE,
        help="Batch size for evaluation",
    )
    args = parser.parse_args()

    if args.compare:
        compare_models(batch_size=args.batch_size)
    else:
        evaluate_model(args.model, batch_size=args.batch_size)
