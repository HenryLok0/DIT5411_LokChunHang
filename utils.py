"""Shared data pipeline. The test array is never opened by train.py."""

import hashlib
import json
import os

import numpy as np
import tensorflow as tf

import config


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path, payload):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def prepared(smoke):
    cache = config.CACHE_ROOT / config.scope(smoke)
    path = cache / "metadata.json"
    meta = read_json(path)
    expected = config.SMOKE_CLASSES if smoke else config.NUM_CLASSES
    if (meta["format"], meta["scope"], meta["image_size"], len(meta["labels"])) != (
        1, config.scope(smoke), config.IMG_SIZE, expected
    ):
        raise ValueError(f"Prepared metadata in {path} does not match the requested experiment.")
    return cache, meta, hashlib.sha256(path.read_bytes()).hexdigest()


def training_indices(counts):
    """Assign exactly 200 augmented draws per class; cycle originals for short classes."""
    per_class = config.TRAIN_PER_CLASS * config.AUGMENT_FACTOR
    idx_chunks = []
    start = 0
    for count in counts:
        if not 0 < count <= config.TRAIN_PER_CLASS:
            raise ValueError(f"Bad original training sample count: {count}")
        idx_chunks.append(np.arange(per_class, dtype=np.int32) % count + start)
        start += count
    return np.concatenate(idx_chunks)


class InputData:
    def __init__(self, cache, meta, split):
        counts = np.array([row[f"{split}_count"] for row in meta["labels"]], dtype=np.int32)
        images = np.load(cache / f"{split}.npy", mmap_mode="r", allow_pickle=False)
        if images.shape != (int(counts.sum()), config.IMG_SIZE, config.IMG_SIZE, 1):
            raise ValueError(f"Image cache shape does not match the metadata for {split}.")
        if not len(images) or images.dtype != np.uint8:
            raise ValueError(f"Invalid image cache for {split}.")
        self.split = split
        self.indices = training_indices(counts) if split == "train" else np.arange(len(images), dtype=np.int32)
        with tf.device("/CPU:0"):
            self.images = tf.convert_to_tensor(images)
            self.labels = tf.convert_to_tensor(np.repeat(np.arange(len(counts), dtype=np.int32), counts))

    def dataset(self, batch_size, epoch=0):
        indices = self.indices.copy()
        if self.split == "train":
            np.random.default_rng(config.SEED + epoch).shuffle(indices)

        def load(batch):
            with tf.device("/CPU:0"):
                img = tf.cast(tf.gather(self.images, batch), tf.float32) / 255.
                return img, tf.gather(self.labels, batch)

        return (tf.data.Dataset.from_tensor_slices(indices).batch(batch_size)
                .map(load, num_parallel_calls=2).prefetch(2))
