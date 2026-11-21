"""Training-only affine image augmentation for normalized grayscale images."""

from tensorflow.keras import layers


def augment_batch(x):
    # Float32 avoids interpolation precision issues under mixed precision.
    x = layers.RandomRotation(10 / 360, fill_mode="constant", fill_value=1.,
                              dtype="float32", name="rotation")(x)
    # Zoom factors are percentage changes in the sampling window.
    x = layers.RandomZoom((-0.12, 0.12), fill_mode="constant", fill_value=1.,
                          dtype="float32", name="zoom")(x)
    # Keras independently flips the sign; explicit nonnegative tuples also round-trip.
    x = layers.RandomShear(x_factor=(0., 0.08), y_factor=(0., 0.), fill_mode="constant",
                           fill_value=1., dtype="float32", name="shear")(x)
    return x
