"""Show one held-out test prediction using a previously trained checkpoint."""
import argparse
from pathlib import Path

import numpy as np
import torch

from har.dataset import ACTIVITIES, load_split
from har.models import build_model


def main():
    parser = argparse.ArgumentParser(description="Inspect one UCI HAR test window")
    parser.add_argument("--model", required=True, choices=("cnn", "lstm"))
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    args = parser.parse_args()

    x, y, _ = load_split(args.data, "test")
    if not 0 <= args.index < len(y):
        parser.error(f"index must be between 0 and {len(y)-1}")
    folder = args.output / args.model
    try:
        stats = np.load(folder / "normalization.npz")
        mean, std = stats["mean"], stats["std"]
        model = build_model(args.model)
        model.load_state_dict(torch.load(folder / "best_model.pt", map_location="cpu", weights_only=True))
    except FileNotFoundError as exc:
        parser.error(f"Trained model not found: {exc}. Run train.py first.")

    model.eval()
    window = (x[args.index:args.index+1] - mean) / std
    with torch.no_grad():
        scores = torch.softmax(model(torch.from_numpy(window.astype(np.float32))), dim=1)[0]
    predicted = int(scores.argmax())
    print(f"Test window #{args.index}")
    print(f"Ground truth: {ACTIVITIES[int(y[args.index])]}")
    print(f"Prediction:   {ACTIVITIES[predicted]}")
    print("Model scores: " + ", ".join(f"{ACTIVITIES[i]}={scores[i].item():.2f}" for i in range(6)))
    print("These scores are model probabilities, not a clinical confidence estimate.")


if __name__ == "__main__":
    main()
