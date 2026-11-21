"""Audit the official ZIP and prepare compact, disjoint train and test arrays."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import shutil
import tempfile
from zipfile import ZipFile

import cv2
import numpy as np

import config


def index_archive(archive):
    """Use the numeric folder as the label and the filename for the glyph/sample ID."""
    folders = defaultdict(dict)
    characters = {}
    for info in archive.infolist():
        if info.is_dir():
            continue
        match = re.fullmatch(r"cleaned_data/([0-9]+)/(.+)_([0-9]+)\.png", info.filename)
        if not match:
            raise ValueError(f"Unexpected member: {info.filename!r}")
        folder, char, sample = match.groups()
        label, sample = int(folder), int(sample)
        if characters.setdefault(label, char) != char or sample in folders[label]:
            raise ValueError(f"Inconsistent glyph or duplicate sample in {info.filename!r}")
        folders[label][sample] = info
    if sorted(folders) != list(range(1, config.NUM_CLASSES + 1)):
        raise ValueError(f"Expected source labels 1..{config.NUM_CLASSES}; found {len(folders)} folders.")
    return [
        (label, characters[label], [info for _, info in sorted(folders[label].items())])
        for label in sorted(folders)
    ]


def split_counts(index):
    counts = [len(images) for _, _, images in index]
    return {
        "classes": len(counts),
        "original_images": sum(counts),
        "minimum_originals_per_class": min(counts),
        "maximum_originals_per_class": max(counts),
        "classes_below_40": sum(n < config.TRAIN_PER_CLASS for n in counts),
        "classes_with_test_samples": sum(n > config.TRAIN_PER_CLASS for n in counts),
        "train_originals": sum(min(n, config.TRAIN_PER_CLASS) for n in counts),
        "test_originals": sum(max(0, n - config.TRAIN_PER_CLASS) for n in counts),
        "augmented_training_images_per_epoch": (
            len(counts) * config.AUGMENT_FACTOR * config.TRAIN_PER_CLASS
        ),
        "originals_per_class_histogram": dict(sorted(Counter(counts).items())),
    }


def pick_smoke_classes(index):
    """Sample across the label space, including one short class to exercise the fallback."""
    eligible = [row for row in index if len(row[2]) > config.TRAIN_PER_CLASS]
    short = [row for row in index if len(row[2]) < config.TRAIN_PER_CLASS]
    rng = np.random.default_rng(config.SEED)
    choice = rng.choice(len(eligible), config.SMOKE_CLASSES - 1, replace=False)
    return sorted([eligible[i] for i in choice] + [min(short, key=lambda row: len(row[2]))])


def preprocess(png):
    image = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("Failed to decode an image in the dataset.")
    # Normalize image polarity using the border, so the affine fill is always white.
    if np.median(np.concatenate((image[0], image[-1], image[:, 0], image[:, -1]))) < 128:
        image = 255 - image
    return cv2.resize(image, (config.IMG_SIZE, config.IMG_SIZE), interpolation=cv2.INTER_AREA)


def prepare(archive_path, smoke=False, audit_only=False):
    with archive_path.open("rb") as stream:
        source_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
    # Python's Big5 codec rejects some names in this archive; CP950 accepts them.
    with ZipFile(archive_path, metadata_encoding="cp950") as archive:
        full_index = index_archive(archive)
        if audit_only:
            return {"archive_sha256": source_sha256, **split_counts(full_index)}
        index = pick_smoke_classes(full_index) if smoke else full_index
        stats = split_counts(index)
        dest = config.CACHE_ROOT / config.scope(smoke)
        if dest.exists():
            raise FileExistsError(f"{dest} already exists; use that cache or select a new cache root.")
        dest.parent.mkdir(parents=True, exist_ok=True)
        temp = Path(tempfile.mkdtemp(dir=dest.parent, prefix=f"{dest.name}-partial-"))
        try:
            train = np.lib.format.open_memmap(
                temp / "train.npy", mode="w+", dtype=np.uint8,
                shape=(stats["train_originals"], config.IMG_SIZE, config.IMG_SIZE, 1),
            )
            test = np.lib.format.open_memmap(
                temp / "test.npy", mode="w+", dtype=np.uint8,
                shape=(stats["test_originals"], config.IMG_SIZE, config.IMG_SIZE, 1),
            )
            classes = []
            tr = te = 0
            for i, (label, char, items) in enumerate(index):
                n_train = min(len(items), config.TRAIN_PER_CLASS)
                classes.append({
                    "source_label": label, "character": char,
                    "train_count": n_train, "test_count": len(items) - n_train,
                })
                for j, item in enumerate(items):
                    try:
                        image = preprocess(archive.read(item))
                    except ValueError as exc:
                        raise ValueError(f"Cannot preprocess {item.filename!r}") from exc
                    if j < n_train:
                        train[tr, :, :, 0] = image
                        tr += 1
                    else:
                        test[te, :, :, 0] = image
                        te += 1
                if (i + 1) % 1000 == 0:
                    print(f"Decoded {i + 1}/{len(index)} classes", flush=True)
            train.flush()
            test.flush()
            del train, test
            meta = {
                "format": 1, "scope": config.scope(smoke), "archive_sha256": source_sha256,
                "image_size": config.IMG_SIZE, "stats": stats, "labels": classes,
            }
            (temp / "metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                                                encoding="utf-8")
            temp.rename(dest)
        finally:
            if temp.exists():
                shutil.rmtree(temp)
    return {"prepared": str(dest), **stats}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path, help="Combined full dataset ZIP")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--smoke", action="store_true", help="Prepare a 32-class development run")
    group.add_argument("--audit-only", action="store_true", help="Read ZIP index without decoding")
    args = parser.parse_args()
    print(json.dumps(prepare(args.archive, args.smoke, args.audit_only), indent=2))
