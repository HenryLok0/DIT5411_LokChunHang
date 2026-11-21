"""Run with python -m unittest discover -s tests."""

from io import BytesIO
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from zipfile import ZipFile

import numpy as np
from tensorflow import keras

from evaluate import read_completed_run
from models import get_model
from prepare_data import index_archive, split_counts
from train import train
from utils import read_json, save_json, training_indices


class DataTests(unittest.TestCase):
    def test_archive_numeric_order_and_split(self):
        with ZipFile(BytesIO(), "w") as archive:
            for i in list(range(1, 42)) + [50]:
                archive.writestr(f"cleaned_data/1/一_{i}.png", b"unused")
            for i in [1, 2, 10]:
                archive.writestr(f"cleaned_data/2/乙_{i}.png", b"unused")
            with patch("config.NUM_CLASSES", 2):
                index = index_archive(archive)
        self.assertEqual(index[0][2][39].filename, "cleaned_data/1/一_40.png")
        self.assertEqual(index[0][2][40].filename, "cleaned_data/1/一_41.png")
        self.assertEqual(index[1][1], "乙")
        stats = split_counts(index)
        self.assertEqual(stats["train_originals"], 43)
        self.assertEqual(stats["test_originals"], 2)
        self.assertEqual(stats["classes_with_test_samples"], 1)
        self.assertEqual(stats["augmented_training_images_per_epoch"], 400)

    def test_augmentation_draw_counts(self):
        indices = training_indices([40, 16])
        self.assertEqual(len(indices), 400)
        np.testing.assert_array_equal(np.bincount(indices[:200]), np.repeat(5, 40))
        reuse = np.bincount(indices[200:] - 40)
        self.assertEqual(sum(reuse), 200)
        self.assertLessEqual(max(reuse) - min(reuse), 1)


class ModelTests(unittest.TestCase):
    def test_full_output_shapes(self):
        for name in ("a", "b", "c"):
            self.assertEqual(get_model(name).output_shape, (None, 13065))
            keras.backend.clear_session()

    def test_augmentation_disabled_at_inference(self):
        model = get_model("c")
        image = np.ones((4, 48, 48, 1), dtype=np.float32)
        image[:, 5:30, 6:10] = 0.
        with tempfile.TemporaryDirectory() as dest:
            path = Path(dest) / "test.keras"
            model.save(path)
            for restored in [model, keras.models.load_model(path)]:
                aug = keras.Model(restored.input, restored.get_layer("shear").output)
                np.testing.assert_array_equal(aug(image, training=False).numpy(), image)
                self.assertFalse(np.array_equal(aug(image, training=True).numpy(), image))

    def test_reject_incomplete_checkpoint(self):
        with tempfile.TemporaryDirectory() as dest:
            run = Path(dest)
            (run / "a").mkdir()
            save_json(run / "a" / "state.json", {
                "protocol": {"epochs": 2, "dataset": "current"},
                "checkpoint": "epoch_1.keras", "history": [{"loss": 1.0}],
            })
            with self.assertRaisesRegex(ValueError, "epoch budget"):
                read_completed_run(run, "a", "current")
            with self.assertRaisesRegex(ValueError, "different dataset"):
                read_completed_run(run, "a", "different")

    def test_resume_after_interrupted_save_without_test_input(self):
        with tempfile.TemporaryDirectory() as dest:
            root = Path(dest)
            cache = root / "cache" / "smoke"
            cache.mkdir(parents=True)
            rng = np.random.default_rng(42)
            np.save(cache / "train.npy", rng.integers(0, 256, (5, 48, 48, 1), dtype=np.uint8))
            save_json(cache / "metadata.json", {
                "format": 1, "scope": "smoke", "image_size": 48,
                "labels": [{"train_count": 1, "test_count": 1} for _ in range(5)],
            })
            args = SimpleNamespace(
                model="c", epochs=2, batch_size=5, learning_rate=0.001,
                mixed_precision=False, smoke=True, resume=True,
            )

            def interrupt_second_commit(path, state):
                if len(state["history"]) == 2:
                    raise OSError("simulated disconnect before committing the second epoch")
                save_json(path, state)

            with (patch("config.CACHE_ROOT", root / "cache"),
                  patch("config.RUN_ROOT", root / "runs"),
                  patch("config.SMOKE_CLASSES", 5),
                  patch("config.TRAIN_PER_CLASS", 1),
                  patch("config.AUGMENT_FACTOR", 1)):
                folder = root / "runs" / "smoke" / "c"
                with patch("train.save_json", side_effect=interrupt_second_commit):
                    with self.assertRaisesRegex(OSError, "simulated disconnect"):
                        train(args)
                self.assertEqual(read_json(folder / "state.json")["checkpoint"], "epoch_1.keras")
                train(args)
                state = read_json(folder / "state.json")
                self.assertEqual(len(state["history"]), 2)
                self.assertFalse((folder / "epoch_1.keras").exists())
                restored = keras.models.load_model(folder / state["checkpoint"])
                self.assertEqual(int(restored.optimizer.iterations.numpy()), 2)
