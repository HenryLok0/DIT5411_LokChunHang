"""Verify and publish the three full-dataset metrics files produced by CI."""

import argparse
import math
from pathlib import Path

import config
from evaluate import format_report, rank
from utils import read_json, save_json


PENDING = "**Full-dataset accuracy and the winning architecture are pending the full experiment.**"


def collect(metrics_dir, report_dir=Path("reports"), readme=Path("README.md")):
    packets = [read_json(metrics_dir / f"{name}.json") for name in config.MODELS]
    protocol_keys = ("scope", "archive_sha256", "stats", "training", "evaluation_batch_size")
    common = {key: packets[0][key] for key in protocol_keys}
    audit = read_json(report_dir / "dataset-audit.json")
    expected_stats = {key: value for key, value in audit.items() if key != "archive_sha256"}
    if (common["scope"] != "full" or common["archive_sha256"] != audit["archive_sha256"]
            or common["stats"] != expected_stats or common["stats"]["classes"] != config.NUM_CLASSES):
        raise ValueError("Metrics do not describe the audited full dataset.")
    rows = []
    for name, packet in zip(config.MODELS, packets):
        if any(packet[key] != common[key] for key in protocol_keys):
            raise ValueError(f"Training or evaluation protocol differs for model {name}.")
        row, = packet["results"]
        if (row["model"] != name or row["parameters"] <= 0 or not math.isfinite(row["loss"])
                or not 0 <= row["accuracy"] <= row["top5_accuracy"] <= 1):
            raise ValueError(f"Invalid metrics for model {name}.")
        rows.append(row)
    payload = rank({**common, "results": rows})
    report = format_report(payload)
    (report_dir / "full-results.md").write_text(report, encoding="utf-8")
    save_json(report_dir / "full-results.json", payload)
    overview = readme.read_text(encoding="utf-8")
    if overview.count(PENDING) != 1:
        raise ValueError("Expected one pending-results marker in the README.")
    leaders = ", ".join(name.upper() for name in payload["highest_top1_models"])
    score = max(row["accuracy"] for row in rows)
    lines = [
        f"Highest observed full-dataset top-1: **Model {leaders} ({score:.2%})**.",
        "See the [full test results](reports/full-results.md) for the scores of all three models.",
    ]
    start = overview.index(PENDING)
    end = overview.index("\n\n", start)
    readme.write_text(overview[:start] + " ".join(lines) + overview[end:], encoding="utf-8")
    print(report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metrics_dir", type=Path)
    args = parser.parse_args()
    collect(args.metrics_dir)
