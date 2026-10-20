"""Compare 1D-CNN and LSTM on the official UCI HAR participant split."""
import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch

from har.dataset import ACTIVITIES, ChannelStandardizer, load_split, train_validation_indices
from har.models import build_model
from har.training import evaluate, loader, train_epoch


def main():
    parser = argparse.ArgumentParser(description="Train CNN and LSTM on UCI HAR inertial signals")
    parser.add_argument("--data", type=Path, default=Path("data"), help="Extracted UCI HAR folder or its parent")
    parser.add_argument("--model", choices=("cnn", "lstm", "both"), default="both")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("epochs and batch-size must be positive")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    x_training, y_training, train_subjects = load_split(args.data, "train")
    x_test, y_test, test_subjects = load_split(args.data, "test")
    assert set(train_subjects).isdisjoint(set(test_subjects)), "Official test subjects overlap training!"
    train_ids, val_ids = train_validation_indices(train_subjects, seed=args.seed)
    assert set(train_subjects[train_ids]).isdisjoint(set(train_subjects[val_ids]))

    scaler = ChannelStandardizer().fit(x_training[train_ids])
    x_train = scaler.transform(x_training[train_ids])
    x_val = scaler.transform(x_training[val_ids])
    x_test = scaler.transform(x_test)
    train_batches = loader(x_train, y_training[train_ids], args.batch_size, shuffle=True)
    val_batches = loader(x_val, y_training[val_ids], args.batch_size)
    test_batches = loader(x_test, y_test, args.batch_size)
    print(f"Windows: train={len(train_ids)}, val={len(val_ids)}, test={len(y_test)}")
    print(f"Subjects: train={len(set(train_subjects[train_ids]))}, val={len(set(train_subjects[val_ids]))}, test={len(set(test_subjects))}")

    models = ("cnn", "lstm") if args.model == "both" else (args.model,)
    for name in models:
        folder = args.output / name
        folder.mkdir(parents=True, exist_ok=True)
        np.savez(folder / "normalization.npz", mean=scaler.mean, std=scaler.std)
        model = build_model(name).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        best_f1 = -1.0
        history = []
        no_improvement = 0
        for epoch in range(1, args.epochs + 1):
            train_loss = train_epoch(model, train_batches, optimizer, device)
            val_results = evaluate(model, val_batches, device)
            val_f1 = val_results["macro_f1"]
            history.append({"epoch": epoch, "train_loss": train_loss,
                            "val_accuracy": val_results["accuracy"], "val_macro_f1": val_f1})
            print(f"{name.upper()} epoch {epoch:02d}: loss={train_loss:.4f} val_f1={val_f1:.4f}")
            if val_f1 > best_f1:
                best_f1 = val_f1
                torch.save(model.state_dict(), folder / "best_model.pt")
                no_improvement = 0
            else:
                no_improvement += 1
            if no_improvement >= 5:
                print(f"Early stopping {name} after {epoch} epochs")
                break
        with (folder / "history.csv").open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=history[0].keys())
            writer.writeheader()
            writer.writerows(history)

        # The test set is evaluated only once, using the best validation checkpoint.
        model.load_state_dict(torch.load(folder / "best_model.pt", map_location=device, weights_only=True))
        final = evaluate(model, test_batches, device)
        summary = {
            "model": name, "dataset": "UCI HAR Smartphones (dataset 240)",
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "split": "Official subject-disjoint test; validation subjects from official train",
            "epochs_trained": len(history), "seed": args.seed,
            "train_windows": len(train_ids), "val_windows": len(val_ids), "test_windows": len(y_test),
            "train_subjects": len(set(train_subjects[train_ids])),
            "val_subjects": len(set(train_subjects[val_ids])),
            "test_subjects": len(set(test_subjects)),
            "test_accuracy": final["accuracy"], "test_macro_f1": final["macro_f1"],
            "per_class": final["per_class"],
        }
        with (folder / "metrics.json").open("w") as f:
            json.dump(summary, f, indent=2)
        with (folder / "confusion_matrix.csv").open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["actual / predicted", *ACTIVITIES])
            for label, row in zip(ACTIVITIES, final["confusion_matrix"]):
                writer.writerow([label, *row])
        print(f"{name.upper()} TEST: accuracy={final['accuracy']:.4f}, macro-F1={final['macro_f1']:.4f}")
        print(f"Saved evaluation files to {folder / 'metrics.json'}")


if __name__ == "__main__":
    main()
