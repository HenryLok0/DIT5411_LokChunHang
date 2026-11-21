"""
Utility helpers used across the project.
TensorFlow is imported lazily so that scripts which do not need it
(prepare_data.py, augment.py) can still run before TensorFlow is installed.
"""

import os
import random

import numpy as np

import config


def _import_tf():
    """Lazy import of tensorflow."""
    import tensorflow as tf

    return tf


def set_seeds(seed: int = config.SEED) -> None:
    """Fix random seeds for reproducible experiments."""
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    np.random.seed(seed)
    tf = _import_tf()
    tf.random.set_seed(seed)


def discover_classes(data_dir: str) -> list:
    """
    Return a sorted list of class folder names under data_dir.
    Each subfolder should be named after one Chinese character.
    """
    if not os.path.isdir(data_dir):
        return []
    classes = [
        name
        for name in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, name))
    ]
    # Sort using Python's default unicode ordering so labels are deterministic.
    classes.sort()
    return classes


def build_label_map(classes: list) -> dict:
    """Map class name (Chinese character) to integer index."""
    return {name: idx for idx, name in enumerate(classes)}


def decode_image(file_path: str, label: int, img_size: int = config.IMG_SIZE):
    """
    Load a PNG/JPEG image, convert to grayscale, resize and normalize.
    Used inside a tf.data pipeline.
    """
    tf = _import_tf()
    img = tf.io.read_file(file_path)
    img = tf.io.decode_png(img, channels=config.IMG_CHANNELS)
    img = tf.image.convert_image_dtype(img, tf.float32)  # scales to [0, 1]
    img = tf.image.resize(img, [img_size, img_size])
    return img, label


def random_augment(image):
    """
    Optional extra augmentation applied inside a tf.data pipeline.
    The model itself already contains RandomRotation / RandomZoom layers, so
    this function is kept only for advanced custom pipelines.
    """
    tf = _import_tf()
    # Small random brightness / contrast jitter
    image = tf.image.random_brightness(image, max_delta=0.05)
    image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
    image = tf.clip_by_value(image, 0.0, 1.0)
    return image


def augment_dataset(ds, augment_fn=random_augment):
    """Apply extra data augmentation to every sample in a tf.data.Dataset."""
    tf = _import_tf()
    return ds.map(lambda x, y: (augment_fn(x), y), num_parallel_calls=tf.data.AUTOTUNE)


def configure_for_performance(
    ds,
    batch_size: int = config.BATCH_SIZE,
    shuffle: bool = True,
    cache: bool = True,
):
    """
    Add caching, shuffling, batching and prefetching to a tf.data pipeline.
    """
    tf = _import_tf()
    if cache:
        ds = ds.cache()
    if shuffle:
        ds = ds.shuffle(buffer_size=2000, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds
