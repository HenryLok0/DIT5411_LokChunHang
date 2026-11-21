"""
Data augmentation script.

The assignment asks for at least 200 training samples per character. With only
40 original samples, we generate AUGMENT_FACTOR (default 5) augmented copies
of each original image, giving 40 * 5 = 200 effective samples per class.

Two modes are supported:
  1) generate-augmented: save augmented PNGs to disk (data/augmented/).
  2) on-the-fly: augmentation is applied inside the tf.data pipeline in train.py.

Mode 2 is recommended for the full 13,065-class dataset because it avoids
writing millions of files to disk.
"""

import argparse
import os
import random

import cv2
import numpy as np

import config
from utils import discover_classes


def _affine_matrix(img: np.ndarray, angle: float, shear: float, scale: float) -> np.ndarray:
    """
    Build an affine transformation matrix for rotation, shear and scaling.
    Positive angle rotates counter-clockwise; shear is in degrees.
    """
    h, w = img.shape[:2]
    center = (w / 2, h / 2)

    # Rotation + scale as a 2x3 matrix
    rot_mat = cv2.getRotationMatrix2D(center, angle, scale)

    # Convert to 3x3 for combination with shear
    rot_3x3 = np.vstack([rot_mat, [0.0, 0.0, 1.0]])

    # Shear matrix (3x3)
    shear_rad = np.deg2rad(shear)
    shear_3x3 = np.array(
        [
            [1.0, np.tan(shear_rad), 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )

    # Combine: first rotate/scale, then shear
    combined = shear_3x3 @ rot_3x3
    return combined[:2, :]


def augment_image(
    img: np.ndarray,
    max_angle: float = 10.0,
    max_shear: float = 8.0,
    scale_range: tuple = (0.85, 1.15),
    max_shift: int = 2,
) -> np.ndarray:
    """
    Apply one random augmentation to a grayscale image.
    Keeps the output image the same size as the input.
    """
    h, w = img.shape[:2]

    angle = random.uniform(-max_angle, max_angle)
    shear = random.uniform(-max_shear, max_shear)
    scale = random.uniform(*scale_range)
    tx = random.randint(-max_shift, max_shift)
    ty = random.randint(-max_shift, max_shift)

    affine = _affine_matrix(img, angle, shear, scale)
    affine[0, 2] += tx
    affine[1, 2] += ty

    aug = cv2.warpAffine(img, affine, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=255)

    # Apply a tiny bit of random brightness / noise
    if random.random() < 0.5:
        noise = np.random.normal(0, 3, aug.shape).astype(np.int16)
        aug = np.clip(aug.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    return aug


def generate_augmented_images(
    src_dir: str,
    dest_dir: str,
    factor: int = config.AUGMENT_FACTOR,
):
    """Generate `factor` augmented copies of every training image and save them."""
    os.makedirs(dest_dir, exist_ok=True)
    classes = discover_classes(src_dir)
    print(f"Generating augmented images for {len(classes)} classes ...")

    total_generated = 0
    for char in classes:
        char_src = os.path.join(src_dir, char)
        char_dest = os.path.join(dest_dir, char)
        os.makedirs(char_dest, exist_ok=True)

        images = [
            f
            for f in os.listdir(char_src)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]

        for img_name in images:
            img_path = os.path.join(char_src, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            base_name = os.path.splitext(img_name)[0]
            for i in range(factor):
                aug = augment_image(img)
                out_name = f"{base_name}_aug{i:03d}.png"
                cv2.imwrite(os.path.join(char_dest, out_name), aug)
                total_generated += 1

    print(f"Generated {total_generated} augmented images in {dest_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate augmented training images")
    parser.add_argument(
        "--src",
        type=str,
        default=config.TRAIN_DIR,
        help="Source training directory",
    )
    parser.add_argument(
        "--dest",
        type=str,
        default=config.AUGMENTED_DIR,
        help="Destination directory for augmented images",
    )
    parser.add_argument(
        "--factor",
        type=int,
        default=config.AUGMENT_FACTOR,
        help="How many augmented copies to create per original image",
    )
    args = parser.parse_args()

    generate_augmented_images(args.src, args.dest, args.factor)
