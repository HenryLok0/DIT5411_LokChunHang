"""
Train a CNN model on the prepared dataset.

Usage examples:
    python train.py --model a --epochs 5
    python train.py --model b --batch-size 512 --augment-dir data/augmented

If --augment-dir is provided, training images are loaded from that folder.
Otherwise augmentation is applied on-the-fly using tf.image operations.
"""

import argparse
import json
import os

import numpy as np
import tensorflow as tf
from tensorflow import keras

import config
from models import compile_model, get_model
from utils import (
    configure_for_performance,
    decode_image,
    discover_classes,
    set_seeds,
)


def create_dataset(data_dir: str, augment: bool = True):
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


def train(
    model_name: str,
    epochs: int = config.EPOCHS,
    batch_size: int = config.BATCH_SIZE,
    learning_rate: float = config.LEARNING_RATE,
    augment_dir: str = None,
):
    """Train one model and save the best checkpoint plus training history."""
    set_seeds(config.SEED)

    # Decide where training images come from
    train_dir = augment_dir if augment_dir else config.TRAIN_DIR
    if not os.path.isdir(train_dir):
        raise FileNotFoundError(f"Training directory not found: {train_dir}")

    # Build datasets
    train_ds, num_classes = create_dataset(train_dir, augment=(augment_dir is None))
    test_ds, _ = create_dataset(config.TEST_DIR, augment=False)

    print(f"Training on {num_classes} classes")
    print(f"Train batches: {len(list(train_ds))}, Test batches: {len(list(test_ds))}")

    # Recompile with the actual number of classes seen in the data
    config.NUM_CLASSES = num_classes

    model = get_model(model_name)
    model = compile_model(model, learning_rate=learning_rate)
    model.summary()

    # Callbacks
    checkpoint_path = os.path.join(
        config.MODELS_DIR, f"best_model_{model_name}.keras"
    )
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
        validation_data=test_ds,
        callbacks=callbacks,
    )

    # Save final model and history
    final_path = os.path.join(config.MODELS_DIR, f"final_model_{model_name}.keras")
    model.save(final_path)

    history_path = os.path.join(config.RESULTS_DIR, f"history_{model_name}.json")
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history.history, f, indent=2)

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
        help="Number of training epochs",
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
        help="Initial learning rate for Adam",
    )
    parser.add_argument(
        "--augment-dir",
        type=str,
        default=None,
        help="Use pre-generated augmented images instead of on-the-fly augmentation",
    )
    args = parser.parse_args()

    train(
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        augment_dir=args.augment_dir,
    )
