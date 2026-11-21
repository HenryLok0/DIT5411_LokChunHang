"""Measure top-1/top-5 on the untouched holdout and write a Markdown report."""

import argparse
import json
import math
from pathlib import Path

from tensorflow import keras

import config
from models import compile_model
from utils import InputData, prepared, read_json, save_json


def read_completed_run(run_dir, name, fingerprint):
    folder = run_dir / name
    state = read_json(folder / "state.json")
    protocol = state["protocol"]
    if protocol["dataset"] != fingerprint:
        raise ValueError(f"Model {name} was trained on a different dataset cache.")
    if len(state["history"]) != protocol["epochs"]:
        raise ValueError(f"Model {name} has not reached its planned epoch budget.")
    return state, folder / state["checkpoint"]


def measure(smoke, batch_size, names):
    if batch_size < 1:
        raise ValueError("Batch size must be positive.")
    cache, meta, fingerprint = prepared(smoke)
    run_dir = config.RUN_ROOT / config.scope(smoke)
    runs = [(name, *read_completed_run(run_dir, name, fingerprint)) for name in names]
    protocol = runs[0][1]["protocol"]
    if any(state["protocol"] != protocol for _, state, _ in runs):
        raise ValueError("The models must share the same data, epoch budget and training settings.")
    data = InputData(cache, meta, "test")
    test = data.dataset(batch_size)
    results = []
    for name, _, checkpoint in runs:
        model = keras.models.load_model(checkpoint, compile=False)
        if model.output_shape[-1] != len(meta["labels"]):
            raise ValueError(f"Model {name} has the wrong number of output classes.")
        # Evaluation only needs the weights; do not restore the optimizer state.
        compile_model(model, protocol["learning_rate"])
        scores = {k: float(v) for k, v in model.evaluate(test, return_dict=True, verbose=2).items()}
        if not all(math.isfinite(value) for value in scores.values()):
            raise ValueError(f"Non-finite evaluation metric for model {name}.")
        results.append({"model": name, "parameters": model.count_params(), **scores})
        del model
        keras.backend.clear_session()
    return {
        "scope": config.scope(smoke), "archive_sha256": meta["archive_sha256"],
        "stats": meta["stats"], "training": protocol, "evaluation_batch_size": batch_size,
        "results": results,
    }


def rank(payload):
    results = payload["results"]
    highest = max(row["accuracy"] for row in results)
    payload["highest_top1_models"] = [row["model"] for row in results if row["accuracy"] == highest]
    return payload


def compare(smoke, batch_size, report_path=None):
    payload = rank(measure(smoke, batch_size, config.MODELS))
    run_dir = config.RUN_ROOT / config.scope(smoke)
    save_json(run_dir / "model_comparison.json", payload)
    report = report_path or run_dir / "results.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(format_report(payload), encoding="utf-8")
    print(json.dumps(payload["results"], indent=2))
    print(f"Comparison report: {report}")


def format_report(payload):
    stats = payload["stats"]
    training = payload["training"]
    if payload["scope"] == "smoke":
        title = "# Real-data CPU smoke test"
        note = "Development subset only. These are not full-dataset assignment scores."
    else:
        title = "# Full-dataset test results"
        note = "All three classifiers were evaluated on the same unaugmented holdout."
    lines = [
        title, "", note, "",
        f"- Output classes: {stats['classes']:,}; test coverage: {stats['classes_with_test_samples']:,} classes",
        f"- Original training images: {stats['train_originals']:,}; original test images: {stats['test_originals']:,}",
        f"- Augmented training draws per epoch: {stats['augmented_training_images_per_epoch']:,}",
        f"- Training: {training['epochs']} epochs, batch {training['batch_size']}, Adam "
        f"learning rate {training['learning_rate']:g}, {training['precision']}, seed {training['seed']}",
        f"- Dataset ZIP SHA-256: `{payload['archive_sha256']}`",
        "",
        "| Model | Parameters | Test loss | Top-1 | Top-5 |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in payload["results"]:
        lines.append(
            f"| {row['model'].upper()} | {row['parameters']:,} | {row['loss']:.4f} | "
            f"{row['accuracy']:.2%} | {row['top5_accuracy']:.2%} |"
        )
    best = ", ".join(name.upper() for name in payload["highest_top1_models"])
    return "\n".join(lines + [
        "", f"Highest observed test top-1: **{best}**.",
        "Top-1 counts exact first-choice predictions; top-5 counts true labels among the five highest scores.",
        "These are image-weighted scores, and classes without held-out samples are not measured.", "",
    ])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--model", choices=config.MODELS, help="Evaluate just one checkpoint")
    parser.add_argument("--metrics", type=Path, help="Write single-model metrics JSON")
    args = parser.parse_args()
    if args.model:
        if not args.metrics or args.report:
            parser.error("Single-model evaluation requires --metrics and does not take --report.")
        args.metrics.parent.mkdir(parents=True, exist_ok=True)
        save_json(args.metrics, measure(args.smoke, args.batch_size, (args.model,)))
    elif args.metrics:
        parser.error("--metrics requires --model.")
    else:
        compare(args.smoke, args.batch_size, args.report)
