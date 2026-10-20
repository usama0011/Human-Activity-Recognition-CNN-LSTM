"""Read UCI HAR inertial windows while respecting participant boundaries."""
from pathlib import Path

import numpy as np

SIGNALS = tuple(
    f"{source}_{axis}"
    for source in ("body_acc", "body_gyro", "total_acc")
    for axis in ("x", "y", "z")
)
ACTIVITIES = (
    "WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS",
    "SITTING", "STANDING", "LAYING",
)


def find_dataset(root):
    root = Path(root)
    for candidate in (root, root / "UCI HAR Dataset"):
        if (candidate / "train" / "Inertial Signals").is_dir():
            return candidate
    raise FileNotFoundError(
        f"Could not find UCI HAR data in {root}. "
        "Download and extract it first; see README.md."
    )


def load_split(root, split):
    """Return sensor windows [examples, 128, 9], class IDs 0..5, subject IDs."""
    if split not in ("train", "test"):
        raise ValueError("split must be 'train' or 'test'")
    root = find_dataset(root)
    sensor_dir = root / split / "Inertial Signals"
    channels = [
        np.loadtxt(sensor_dir / f"{name}_{split}.txt", dtype=np.float32, ndmin=2)
        for name in SIGNALS
    ]
    if any(ch.shape != channels[0].shape for ch in channels):
        raise ValueError("Inertial-signal files have inconsistent shapes")
    x = np.stack(channels, axis=-1)
    y = np.loadtxt(root / split / f"y_{split}.txt", dtype=np.int64, ndmin=1) - 1
    subjects = np.loadtxt(root / split / f"subject_{split}.txt", dtype=np.int64, ndmin=1)
    if len(x) != len(y) or len(y) != len(subjects):
        raise ValueError("Samples, labels and participant IDs differ in length")
    if x.shape[1:] != (128, 9) or np.any((y < 0) | (y >= len(ACTIVITIES))):
        raise ValueError("Expected 128 time steps, 9 channels and six activity labels")
    if not np.isfinite(x).all():
        raise ValueError("Sensor data contains non-finite values")
    return x, y, subjects


def train_validation_indices(subject_ids, fraction=0.2, seed=42):
    """Hold out participants, not overlapping sensor windows."""
    people = np.unique(subject_ids)
    if len(people) < 2:
        raise ValueError("At least two participants are required for grouped splitting")
    n_val = max(1, min(len(people) - 1, round(len(people) * fraction)))
    rng = np.random.default_rng(seed)
    held_out = rng.permutation(people)[:n_val]
    valid = np.isin(subject_ids, held_out)
    return np.flatnonzero(~valid), np.flatnonzero(valid)


class ChannelStandardizer:
    """Standardize each signal using training participants only."""
    def fit(self, x):
        self.mean = x.mean(axis=(0, 1), keepdims=True)
        self.std = x.std(axis=(0, 1), keepdims=True)
        self.std = np.maximum(self.std, 1e-6)
        return self

    def transform(self, x):
        return ((x - self.mean) / self.std).astype(np.float32)
