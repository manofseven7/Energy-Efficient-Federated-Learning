"""Leakage-safe preprocessing for the official Intel Lab sensor dataset."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def preprocess(source: Path, destination: Path) -> dict:
    # Columns: epoch, mote id, temperature, humidity, light, voltage.
    with gzip.open(source, "rt") as handle:
        # The official file contains a small number of truncated records;
        # genfromtxt skips only those malformed rows instead of aborting the
        # reproducible preprocessing run.
        values = np.genfromtxt(
            handle, usecols=(2, 3, 4, 5, 6, 7), dtype=np.float64,
            invalid_raise=False,
        )
    finite = np.isfinite(values).all(axis=1)
    physical = (
        (values[:, 1] >= 1) & (values[:, 1] <= 54)
        & (values[:, 2] >= -10) & (values[:, 2] <= 60)
        & (values[:, 3] >= 0) & (values[:, 3] <= 100)
        & (values[:, 4] >= 0) & (values[:, 5] >= 0)
    )
    values = values[finite & physical]
    order = np.lexsort((values[:, 0], values[:, 1]))
    values = values[order]
    current = values[:-1]
    following = values[1:]
    same_sensor = current[:, 1] == following[:, 1]
    positive_gap = (following[:, 0] - current[:, 0]) > 0
    # Exclude long outages: this is next-observation forecasting, not imputation.
    bounded_gap = (following[:, 0] - current[:, 0]) <= 120
    keep = same_sensor & positive_gap & bounded_gap
    current = current[keep]; following = following[keep]
    gap = following[:, 0] - current[:, 0]
    # Current four measurements, elapsed seconds, and normalized sensor id.
    features = np.c_[current[:, 2:6], gap, current[:, 1] / 54.0]
    targets = following[:, 2:4]  # next temperature and next humidity
    timestamps = current[:, 0]
    cutoff = np.quantile(timestamps, 0.8)
    train = timestamps <= cutoff
    test = timestamps > cutoff
    x_mean = features[train].mean(axis=0); x_std = features[train].std(axis=0)
    x_std[x_std < 1e-8] = 1.0
    y_mean = targets[train].mean(axis=0); y_std = targets[train].std(axis=0)
    y_std[y_std < 1e-8] = 1.0
    x = ((features - x_mean) / x_std).astype(np.float32)
    y = ((targets - y_mean) / y_std).astype(np.float32)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination,
        x_train=x[train], x_test=x[test],
        y1_train=y[train, 0], y1_test=y[test, 0],
        y2_train=y[train, 1], y2_test=y[test, 1],
        client_train=current[train, 1].astype(np.int16),
        client_test=current[test, 1].astype(np.int16),
        x_mean=x_mean, x_std=x_std, y_mean=y_mean, y_std=y_std,
        split_epoch=np.asarray([cutoff]), task_type=np.asarray(["regression", "regression"]),
    )
    manifest = {
        "source": str(source), "source_sha256": sha256(source),
        "raw_rows": int(len(values)), "paired_rows": int(len(features)),
        "train_rows": int(train.sum()), "test_rows": int(test.sum()),
        "split": "global chronological 80/20 by epoch",
        "targets": ["next_temperature", "next_humidity"],
        "outlier_policy": "finite; mote 1..54; temperature -10..60 C; humidity 0..100%; nonnegative light/voltage",
        "maximum_gap_seconds": 120,
    }
    manifest_path = destination.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(preprocess(args.source, args.output), indent=2))


if __name__ == "__main__":
    main()
