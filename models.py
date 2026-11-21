"""Three CNN classifiers with the same data and optimizer."""

from tensorflow import keras
from tensorflow.keras import layers as L

from augment import augment_batch
import config


ARCHITECTURES = {
    "a": ((32, 64, 128), 256, 0.30),
    "b": ((24, 48), 128, 0.25),
    "c": ((16,), 64, 0.20),
}


def get_model(name, num_classes=config.NUM_CLASSES):
    filters, dense, dropout = ARCHITECTURES[name]
    image = keras.Input((config.IMG_SIZE, config.IMG_SIZE, 1), name="grayscale")
    x = augment_batch(image)
    x = L.AveragePooling2D(2)(x)
    for i, width in enumerate(filters):
        x = L.Conv2D(width, 3, padding="same", use_bias=False)(x)
        # Faster moving statistics reduce the train/inference gap in short runs.
        x = L.BatchNormalization(momentum=0.9)(x)
        x = L.LeakyReLU(negative_slope=0.1)(x)
        if i < 2:
            x = L.MaxPooling2D(2)(x)
    if name == "c":
        x = L.AveragePooling2D(2)(x)
    # Keep the 6×6 spatial layout: stroke positions matter for distinguishing glyphs.
    x = L.Flatten()(x)
    x = L.Dense(dense, activation="relu")(x)
    x = L.Dropout(dropout)(x)
    y = L.Dense(num_classes, activation="softmax", dtype="float32", name="character")(x)
    return keras.Model(image, y, name=f"model_{name}")


def compile_model(model, learning_rate):
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=[
            keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
            keras.metrics.SparseTopKCategoricalAccuracy(k=5, name="top5_accuracy"),
        ],
        jit_compile=False,
    )
    return model
