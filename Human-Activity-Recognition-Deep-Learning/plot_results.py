"""Create a confusion matrix image from genuine train.py results."""
import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from har.dataset import ACTIVITIES


def main():
    parser = argparse.ArgumentParser(description="Plot a saved model confusion matrix")
    parser.add_argument("--model", choices=("cnn", "lstm"), required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    args = parser.parse_args()
    folder = args.output / args.model
    with (folder / "confusion_matrix.csv").open(newline="") as f:
        rows = list(csv.reader(f))
    counts = np.array([[int(v) for v in row[1:]] for row in rows[1:]], dtype=int)
    if counts.shape != (6, 6):
        raise ValueError("Confusion matrix must be 6 x 6")
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(counts, cmap="Blues")
    fig.colorbar(im, ax=ax)
    ax.set_xticks(range(6), ACTIVITIES, rotation=45, ha="right")
    ax.set_yticks(range(6), ACTIVITIES)
    ax.set_xlabel("Predicted activity")
    ax.set_ylabel("True activity")
    ax.set_title(f"{args.model.upper()} test-set confusion matrix")
    for i in range(6):
        for j in range(6):
            ax.text(j, i, str(counts[i, j]), ha="center", va="center", fontsize=9)
    fig.tight_layout()
    dest = folder / "confusion_matrix.png"
    fig.savefig(dest, dpi=160)
    plt.close(fig)
    print(f"Saved {dest}")


if __name__ == "__main__":
    main()
