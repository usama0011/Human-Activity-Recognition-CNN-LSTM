"""Reusable training and evaluation operations."""
import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .dataset import ACTIVITIES


def loader(x, y, batch_size=128, shuffle=False):
    dataset = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)


def train_epoch(model, batches, optimizer, device):
    model.train()
    criterion = nn.CrossEntropyLoss()
    loss_sum, seen = 0.0, 0
    for x, y in batches:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item() * len(y)
        seen += len(y)
    return loss_sum / max(seen, 1)


@torch.no_grad()
def predict(model, batches, device):
    model.eval()
    predictions, ground_truth = [], []
    for x, y in batches:
        logits = model(x.to(device))
        predictions.extend(logits.argmax(1).cpu().numpy().tolist())
        ground_truth.extend(y.numpy().tolist())
    return np.asarray(ground_truth), np.asarray(predictions)


def evaluate(model, batches, device):
    y_true, y_pred = predict(model, batches, device)
    labels = list(range(len(ACTIVITIES)))
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "per_class": classification_report(
            y_true, y_pred, labels=labels, target_names=ACTIVITIES,
            output_dict=True, zero_division=0,
        ),
    }
