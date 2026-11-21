"""Train a fixed number of epochs, without using the held-out test set."""

import argparse
import math
import os

import tensorflow as tf
from tensorflow import keras

import config
from models import compile_model, get_model
from utils import InputData, prepared, read_json, save_json


class Progress(keras.callbacks.Callback):
    def on_train_batch_end(self, batch, logs=None):
        if (batch + 1) % 1000 == 0:
            print(f"Training batch {batch + 1:,}/{self.params['steps']:,}", flush=True)


def train(args):
    if args.epochs < 1 or args.batch_size < 1 or args.learning_rate <= 0:
        raise ValueError("Epochs, batch size and learning rate must all be positive.")
    if args.mixed_precision and not tf.config.list_physical_devices("GPU"):
        raise RuntimeError("--mixed-precision requires a supported GPU; omit it on CPU.")
    keras.utils.set_random_seed(config.SEED)
    policy = "mixed_float16" if args.mixed_precision else "float32"
    keras.mixed_precision.set_global_policy(policy)
    cache, meta, fingerprint = prepared(args.smoke)
    run_dir = config.RUN_ROOT / config.scope(args.smoke) / args.model
    run_dir.mkdir(parents=True, exist_ok=True)
    state_path = run_dir / "state.json"
    protocol = {
        "version": 1, "dataset": fingerprint, "batch_size": args.batch_size,
        "learning_rate": args.learning_rate, "precision": policy, "seed": config.SEED,
        "epochs": args.epochs,
    }
    if state_path.exists():
        if not args.resume:
            raise FileExistsError(f"Run already exists at {run_dir}. Use --resume with the same options.")
        state = read_json(state_path)
        if state["protocol"] != protocol:
            raise ValueError("Cannot resume with different data, epoch target or hyperparameters.")
        if len(state["history"]) == args.epochs:
            print(f"Model {args.model.upper()} already completed {args.epochs} epochs.")
            return
        model = keras.models.load_model(run_dir / state["checkpoint"])
    else:
        model = compile_model(get_model(args.model, len(meta["labels"])), args.learning_rate)
        state = {"protocol": protocol, "checkpoint": None, "history": []}

    data = InputData(cache, meta, "train")
    print(f"Training on {len(meta['labels']):,} labels: {len(data.indices):,} draws per epoch.")
    callbacks = [Progress()] if os.environ.get("GITHUB_ACTIONS") == "true" else []
    for epoch in range(len(state["history"]), args.epochs):
        print(f"Training epoch {epoch + 1}/{args.epochs}", flush=True)
        history = model.fit(data.dataset(args.batch_size, epoch), initial_epoch=epoch,
                            epochs=epoch + 1, shuffle=False, verbose=2, callbacks=callbacks).history
        values = {k: float(v[-1]) for k, v in history.items()}
        if not all(math.isfinite(value) for value in values.values()):
            raise ValueError("Non-finite training metric; the last valid checkpoint was left intact.")
        previous = state["checkpoint"]
        checkpoint = f"epoch_{epoch + 1}.keras"
        # State advances only after the new model and optimizer have been saved.
        tmp = run_dir / "partial.keras"
        model.save(tmp)
        os.replace(tmp, run_dir / checkpoint)
        state["checkpoint"] = checkpoint
        state["history"].append(values)
        save_json(state_path, state)
        if previous:
            (run_dir / previous).unlink(missing_ok=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=config.MODELS, required=True)
    parser.add_argument("--epochs", type=int, default=config.EPOCHS)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=config.LEARNING_RATE)
    parser.add_argument("--mixed-precision", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    train(parser.parse_args())
