"""Quick sanity check for train/test images and labels."""

import os
import numpy as np
import tensorflow as tf
import config
from utils import discover_classes, decode_image

train_classes = discover_classes(config.TRAIN_DIR)
print("train classes:", len(train_classes))
print("first 5:", train_classes[:5])

# Load a few images
paths = []
labels = []
for i, char in enumerate(train_classes[:3]):
    char_dir = os.path.join(config.TRAIN_DIR, char)
    for name in os.listdir(char_dir)[:2]:
        paths.append(os.path.join(char_dir, name))
        labels.append(i)

for p, y in zip(paths, labels):
    img, lab = decode_image(p, y)
    arr = img.numpy()
    print(p, "label", int(lab), "shape", arr.shape, "min", arr.min(), "max", arr.max(), "mean", arr.mean())

# Tiny overfitting test: 1 class, 8 images, many steps
print("\nOverfit smoke test on 1 class...")
cls = train_classes[0]
char_dir = os.path.join(config.TRAIN_DIR, cls)
files = [os.path.join(char_dir, f) for f in os.listdir(char_dir)[:8]]
ys = np.zeros(len(files), dtype=np.int32)
ds = tf.data.Dataset.from_tensor_slices((np.array(files), ys))
ds = ds.map(decode_image).batch(8).repeat()

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(config.IMG_SIZE, config.IMG_SIZE, 1)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(1, activation="softmax"),
])
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
hist = model.fit(ds, steps_per_epoch=20, epochs=3, verbose=1)
print("smoke history:", hist.history)
