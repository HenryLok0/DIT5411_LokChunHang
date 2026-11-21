"""Recognize one grayscale handwriting image using a completed checkpoint."""

import argparse
import json
from pathlib import Path

import numpy as np
from tensorflow import keras

import config
from evaluate import read_completed_run
from prepare_data import preprocess
from utils import prepared


def predict(path, name, smoke=False):
    _, meta, fingerprint = prepared(smoke)
    _, checkpoint = read_completed_run(config.RUN_ROOT / config.scope(smoke), name, fingerprint)
    model = keras.models.load_model(checkpoint, compile=False)
    image = preprocess(path.read_bytes()).astype(np.float32)[None, :, :, None] / 255.
    probabilities = model(image, training=False).numpy()[0]
    labels = meta["labels"]
    if len(probabilities) != len(labels):
        raise ValueError("Checkpoint output does not match the label map.")
    return [
        {"character": labels[idx]["character"], "source_label": labels[idx]["source_label"],
         "score": float(probabilities[idx])}
        for idx in np.argsort(-probabilities)[:5]
    ]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path, help="Image containing a single character")
    parser.add_argument("--model", choices=config.MODELS, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    print(json.dumps(predict(args.image, args.model, args.smoke), ensure_ascii=False, indent=2))
