"""
Prepare the dataset for training and testing.

Steps:
1. Locate the raw image directory (full 300x300 dataset or 50x50 common words).
2. For each character folder, copy the first N images to train/ and the rest to test/.
3. Rename folders and files to ASCII-safe numeric IDs. This avoids encoding
   issues with TensorFlow / OpenCV on Windows when paths contain Chinese characters.
4. Save a label mapping (metadata.json) to translate numeric IDs back to characters.

Expected raw folder structure after unzip:
    data/raw/cleaned_data/<character>/<character>_<id>.png

If you use the 50x50 common-words dataset the folder is named
`cleaned_data(50_50)` and the script will use that automatically.
"""

import argparse
import json
import os
import shutil

import config
from utils import discover_classes


def find_raw_data_dir():
    """Pick the first existing raw dataset folder."""
    for candidate in [config.RAW_DATA_DIR, config.RAW_DATA_DIR_50x50]:
        if os.path.isdir(candidate):
            print(f"Using raw dataset: {candidate}")
            return candidate
    raise FileNotFoundError(
        f"No raw dataset folder found at {config.RAW_DATA_DIR} or "
        f"{config.RAW_DATA_DIR_50x50}. Please download and unzip the dataset first."
    )


def _copy_with_numeric_names(
    src_dir: str, dest_dir: str, image_names: list, start_index: int = 0
) -> int:
    """Copy selected images into dest_dir with numeric filenames."""
    os.makedirs(dest_dir, exist_ok=True)
    count = 0
    for idx, img_name in enumerate(image_names, start=start_index):
        src_path = os.path.join(src_dir, img_name)
        ext = os.path.splitext(img_name)[1].lower()
        dest_path = os.path.join(dest_dir, f"{idx:04d}{ext}")
        shutil.copy2(src_path, dest_path)
        count += 1
    return count


def split_character_folder(
    src_dir: str,
    train_dir: str,
    test_dir: str,
    n_train: int,
):
    """
    Copy the first n_train images from src_dir to train_dir and the rest to test_dir.
    Renames files to numeric IDs. Returns (train_count, test_count).
    """
    images = [
        f
        for f in os.listdir(src_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]
    # Sort to make the split deterministic across runs
    images.sort()

    train_images = images[:n_train]
    test_images = images[n_train:]

    tr = _copy_with_numeric_names(src_dir, train_dir, train_images, start_index=0)
    te = _copy_with_numeric_names(src_dir, test_dir, test_images, start_index=n_train)

    return tr, te


def prepare(raw_dir: str, n_train: int = config.TRAIN_SAMPLES_PER_CLASS):
    """Split every character folder into train/test folders with numeric IDs."""
    classes = discover_classes(raw_dir)
    print(f"Found {len(classes)} character classes in {raw_dir}")

    # Limit to the requested number of classes if there are more
    if len(classes) > config.NUM_CLASSES:
        classes = classes[: config.NUM_CLASSES]

    # Numeric label map: index -> Chinese character
    label_map = {idx: char for idx, char in enumerate(classes)}

    total_train = 0
    total_test = 0
    skipped = []

    for idx, char in enumerate(classes):
        src = os.path.join(raw_dir, char)
        train_dest = os.path.join(config.TRAIN_DIR, f"{idx:05d}")
        test_dest = os.path.join(config.TEST_DIR, f"{idx:05d}")

        n_img = len(
            [f for f in os.listdir(src) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
        )
        if n_img < n_train:
            skipped.append((char, n_img))
            continue

        tr, te = split_character_folder(src, train_dest, test_dest, n_train)
        total_train += tr
        total_test += te

    # Save metadata
    metadata = {
        "num_classes": len(classes),
        "train_samples": total_train,
        "test_samples": total_test,
        "train_per_class": n_train,
        "label_map": label_map,
        "skipped_classes": skipped,
    }
    meta_path = os.path.join(config.RESULTS_DIR, "metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"Prepared {len(classes)} classes")
    print(f"Train samples: {total_train}, Test samples: {total_test}")
    if skipped:
        print(f"Skipped {len(skipped)} classes with fewer than {n_train} samples")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare train/test split for Chinese handwriting dataset"
    )
    parser.add_argument(
        "--raw-dir",
        type=str,
        default=None,
        help="Override the raw dataset directory from config.py",
    )
    parser.add_argument(
        "--n-train",
        type=int,
        default=config.TRAIN_SAMPLES_PER_CLASS,
        help="Number of training samples per class",
    )
    args = parser.parse_args()

    raw_dir = args.raw_dir if args.raw_dir else find_raw_data_dir()
    prepare(raw_dir, n_train=args.n_train)
