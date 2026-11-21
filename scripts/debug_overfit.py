"""
Debug: try to overfit 20 classes. If this fails, the data pipeline is broken.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import config
from utils import discover_classes, decode_image

# CUDA libs
site = __import__("site").getsitepackages()[0]
libs = [
    "cublas", "cuda_cupti", "cuda_nvrtc", "cuda_runtime", "cudnn",
    "cufft", "curand", "cusolver", "cusparse", "nccl", "nvjitlink",
]
os.environ["LD_LIBRARY_PATH"] = ":".join(
    [f"{site}/nvidia/{n}/lib" for n in libs] + [os.environ.get("LD_LIBRARY_PATH", "")]
)

classes = discover_classes(config.TRAIN_DIR)[:20]
print("Using classes:", classes)

paths, labels = [], []
for idx, char in enumerate(classes):
    char_dir = os.path.join(config.TRAIN_DIR, char)
    for name in sorted(os.listdir(char_dir))[:40]:
        if name.lower().endswith((".png", ".jpg", ".jpeg")):
            paths.append(os.path.join(char_dir, name))
            labels.append(idx)

print("samples:", len(paths), "unique labels:", len(set(labels)))

# Inspect first image
img0, y0 = decode_image(paths[0], labels[0])
arr = img0.numpy()
print("image0", paths[0], "label", int(y0), "shape", arr.shape, "min/max/mean", arr.min(), arr.max(), arr.mean())

ds = tf.data.Dataset.from_tensor_slices((np.array(paths), np.array(labels, dtype=np.int32)))
ds = ds.map(decode_image, num_parallel_calls=tf.data.AUTOTUNE)
ds = ds.shuffle(1000).batch(64).prefetch(tf.data.AUTOTUNE)

inputs = keras.Input(shape=(config.IMG_SIZE, config.IMG_SIZE, 1))
x = layers.Conv2D(32, 3, activation="relu", padding="same")(inputs)
x = layers.MaxPooling2D()(x)
x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)
x = layers.MaxPooling2D()(x)
x = layers.Flatten()(x)
x = layers.Dense(128, activation="relu")(x)
outputs = layers.Dense(20, activation="softmax")(x)
model = keras.Model(inputs, outputs)
model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
hist = model.fit(ds, epochs=8, verbose=1)
print("final accuracy:", hist.history["accuracy"][-1])
print("final loss:", hist.history["loss"][-1])
