"""
CNN model architectures for Traditional Chinese handwritten character recognition.

Each model accepts grayscale images of size IMG_SIZE x IMG_SIZE and outputs
NUM_CLASSES logits (one per character). The assignment asks for multiple
architectures so you can compare them.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import config


def _input_block():
    """Common input layer."""
    return keras.Input(shape=(config.IMG_SIZE, config.IMG_SIZE, config.IMG_CHANNELS))


def _augmentation_block(x):
    """
    On-the-fly data augmentation applied only during training.
    Includes small rotations, scaling (zoom), translation and contrast jitter.
    fill_value=1.0 assumes a white background after normalization.
    """
    x = layers.RandomRotation(factor=0.1, fill_mode="constant", fill_value=1.0)(x)
    x = layers.RandomZoom(
        height_factor=(-0.15, 0.0),
        width_factor=(-0.15, 0.0),
        fill_mode="constant",
        fill_value=1.0,
    )(x)
    x = layers.RandomTranslation(height_factor=0.05, width_factor=0.05, fill_mode="constant", fill_value=1.0)(x)
    x = layers.RandomContrast(factor=0.1)(x)
    return x


def _conv_block(x, filters: int, kernel_size: int = 3, pool: bool = True):
    """A small convolutional block: Conv -> BatchNorm -> LeakyReLU -> (MaxPool)."""
    x = layers.Conv2D(filters, kernel_size, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(negative_slope=0.1)(x)
    if pool:
        x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    return x


def build_model_a(
    dense_units: int = 128,
    dropout_rate: float = config.DROPOUT_RATE,
) -> keras.Model:
    """
    Deep CNN (Model A2 style).
    Filters: 32 -> 64 -> 128, Dense: 128.
    This tends to give the best accuracy among the three models.
    """
    inputs = _input_block()
    x = _augmentation_block(inputs)
    x = _conv_block(x, 32)
    x = _conv_block(x, 64)
    x = _conv_block(x, 128, pool=False)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(dense_units, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(negative_slope=0.1)(x)
    x = layers.Dropout(dropout_rate)(x)
    # Final layer uses float32 for numerical stability with mixed precision
    outputs = layers.Dense(
        config.NUM_CLASSES, activation="softmax", dtype="float32", name="predictions"
    )(x)

    model = keras.Model(inputs, outputs, name="model_a")
    return model


def build_model_b(
    dense_units: int = 64,
    dropout_rate: float = config.DROPOUT_RATE,
) -> keras.Model:
    """
    Medium CNN.
    Filters: 16 -> 32, Dense: 64.
    A balance between speed and capacity.
    """
    inputs = _input_block()
    x = _augmentation_block(inputs)
    x = _conv_block(x, 16)
    x = _conv_block(x, 32, pool=False)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(dense_units, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(negative_slope=0.1)(x)
    x = layers.Dropout(dropout_rate)(x)
    # Final layer uses float32 for numerical stability with mixed precision
    outputs = layers.Dense(
        config.NUM_CLASSES, activation="softmax", dtype="float32", name="predictions"
    )(x)

    model = keras.Model(inputs, outputs, name="model_b")
    return model


def build_model_c(
    dense_units: int = 16,
    dropout_rate: float = config.DROPOUT_RATE,
) -> keras.Model:
    """
    Small CNN.
    Filters: 8, Dense: 16.
    Trains quickly but has lower capacity.
    """
    inputs = _input_block()
    x = _augmentation_block(inputs)
    x = _conv_block(x, 8, pool=False)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(dense_units, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(negative_slope=0.1)(x)
    x = layers.Dropout(dropout_rate)(x)
    # Final layer uses float32 for numerical stability with mixed precision
    outputs = layers.Dense(
        config.NUM_CLASSES, activation="softmax", dtype="float32", name="predictions"
    )(x)

    model = keras.Model(inputs, outputs, name="model_c")
    return model


MODEL_BUILDERS = {
    "a": build_model_a,
    "b": build_model_b,
    "c": build_model_c,
}


def get_model(name: str, **kwargs) -> keras.Model:
    """Return a compiled or uncompiled model by name."""
    name = name.lower()
    if name not in MODEL_BUILDERS:
        raise ValueError(f"Unknown model '{name}'. Choose from {list(MODEL_BUILDERS)}")
    return MODEL_BUILDERS[name](**kwargs)


def compile_model(model: keras.Model, learning_rate: float = config.LEARNING_RATE):
    """Compile the model with Adam and sparse categorical crossentropy."""
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy", keras.metrics.SparseTopKCategoricalAccuracy(k=5, name="top5_accuracy")],
    )
    return model
