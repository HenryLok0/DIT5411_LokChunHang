"""Shared settings for the full experiment and the isolated smoke run."""

import os
from pathlib import Path

ROOT = Path(os.environ.get("DIT5411_PROJECT_ROOT", Path(__file__).resolve().parent))
CACHE_ROOT = Path(os.environ.get("DIT5411_CACHE_ROOT", ROOT / "data" / "prepared_v2"))
RUN_ROOT = ROOT / "runs"
NUM_CLASSES = 13_065
TRAIN_PER_CLASS = 40
AUGMENT_FACTOR = 5
IMG_SIZE = 48
SEED = 42
BATCH_SIZE = 128
EPOCHS = 5
LEARNING_RATE = 1e-3
MODELS = ("a", "b", "c")
SMOKE_CLASSES = 32


def scope(smoke):
    return "smoke" if smoke else "full"
