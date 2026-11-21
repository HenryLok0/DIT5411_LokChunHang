"""
Train a CNN model on the prepared dataset.

Usage examples:
    python train.py --model a --epochs 5
    python train.py --model b --batch-size 512 --augment-dir data/augmented
    python train.py --model a --epochs 5 --resume

If --resume is set, the script will load the latest checkpoint and continue training
from the last saved epoch. This is useful when Colab disconnects or when you want
to split training across multiple days.

If --augment-dir is provided, training images are loaded from that folder.
Otherwise augmentation is applied on-the-fly using the model's preprocessing layers.
"""

import argparse
import json
import os

import numpy as np
import tensorflow as tf
from tensorflow import keras

import config
from models import compile_model, get_model

# Enable mixed-precision training for faster GPU computation
if config.MIXED_PRECISION:
    from tensorflow.keras import mixed_precision
    mixed_precision.set_global_policy("mixed_float16")
    print("Mixed precision enabled (float16).")
from utils import (
    configure_for_performance,
    decode_image,
    discover_classes,
    set_seeds,
)


def create_dataset(data_dir: str):
    """
    Build a tf.data.Dataset from a directory of class folders.
    Each folder name becomes the integer label.
    """
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

    file_paths = np.array(file_paths)
    labels = np.array(labels, dtype=np.int32)

    ds = tf.data.Dataset.from_tensor_slices((file_paths, labels))
    ds = ds.map(decode_image, num_parallel_calls=tf.data.AUTOTUNE)

    # Augmentation is handled by the model's preprocessing layers (RandomRotation,
    # RandomZoom, etc.) so it is only applied during training, not evaluation.
    return ds, len(classes)


def _load_history(model_name: str) -> dict:
    """Load previous training history if it exists."""
    history_path = os.path.join(config.RESULTS_DIR, f"history_{model_name}.json")
    if not os.path.exists(history_path):
        return {}
    with open(history_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _merge_histories(old: dict, new: dict) -> dict:
    """Append new history values to old ones for continuity."""
    merged = {}
    for key in set(old) | set(new):
        merged[key] = old.get(key, []) + new.get(key, [])
    return merged


def train(
    model_name: str,
    epochs: int = config.EPOCHS,
    batch_size: int = config.BATCH_SIZE,
    learning_rate: float = config.LEARNING_RATE,
    augment_dir: str = None,
    resume: bool = False,
):
    """Train one model and save the best checkpoint plus training history."""
    set_seeds(config.SEED)

    # Decide where training images come from
    train_dir = augment_dir if augment_dir else config.TRAIN_DIR
    if not os.path.isdir(train_dir):
        raise FileNotFoundError(f"Training directory not found: {train_dir}")

    # Build datasets
    train_ds, num_classes = create_dataset(train_dir)
    test_ds, _ = create_dataset(config.TEST_DIR)

    print(f"Training on {num_classes} classes")
    # Avoid materializing the whole dataset with list(); that would exhaust it.

    # Paths
    checkpoint_path = os.path.join(
        config.MODELS_DIR, f"best_model_{model_name}.keras"
    )
    final_path = os.path.join(config.MODELS_DIR, f"final_model_{model_name}.keras")
    history_path = os.path.join(config.RESULTS_DIR, f"history_{model_name}.json")

    # Resume logic
    old_history = _load_history(model_name)
    initial_epoch = 0
    if resume and os.path.exists(checkpoint_path):
        print(f"Resuming from checkpoint: {checkpoint_path}")
        model = keras.models.load_model(checkpoint_path)
        initial_epoch = len(old_history.get("loss", []))
        print(f"Continuing from epoch {initial_epoch} up to epoch {epochs}")
    else:
        if resume:
            print(f"No checkpoint found at {checkpoint_path}; starting from scratch.")
        # Recompile with the actual number of classes seen in the data
        config.NUM_CLASSES = num_classes
        model = get_model(model_name)
        model = compile_model(model, learning_rate=learning_rate)

    model.summary()

    # Callbacks
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            checkpoint_path,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=2, verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=5, restore_best_weights=True, verbose=1
        ),
    ]

    # Prepare performance pipelines
    train_ds = configure_for_performance(train_ds, batch_size=batch_size, shuffle=True)
    test_ds = configure_for_performance(test_ds, batch_size=batch_size, shuffle=False)

    # Train
    history = model.fit(
        train_ds,
        epochs=epochs,
        initial_epoch=initial_epoch,
        validation_data=test_ds,
        callbacks=callbacks,
    )

    # Save final model and merged history
    model.save(final_path)

    merged_history = _merge_histories(old_history, history.history)
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(merged_history, f, indent=2)

    print(f"Training complete. Best model saved to {checkpoint_path}")
    print(f"Final model saved to {final_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Chinese handwriting CNN")
    parser.add_argument(
        "--model",
        type=str,
        default="a",
        choices=["a", "b", "c"],
        help="Model architecture to train (a/b/c)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=config.EPOCHS,
        help="Total number of training epochs (including already completed ones)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=config.BATCH_SIZE,
        help="Batch size",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=config.LEARNING_RATE,
        help="Initial learning rate for Adam (ignored when resuming)",
    )
    parser.add_argument(
        "--augment-dir",
        type=str,
        default=None,
        help="Use pre-generated augmented images instead of on-the-fly augmentation",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from the best checkpoint if it exists",
    )
    args = parser.parse_args()

    train(
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        augment_dir=args.augment_dir,
        resume=args.resume,
    )
