"""Small, readable CNN and LSTM baselines for 9-channel sensor windows."""
import torch
from torch import nn


class ActivityCNN(nn.Module):
    def __init__(self, channels=9, classes=6):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(channels, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Linear(64, classes)

    def forward(self, windows):
        # Conv1d expects [batch, channels, time].
        features = self.features(windows.transpose(1, 2)).squeeze(-1)
        return self.classifier(features)


class ActivityLSTM(nn.Module):
    def __init__(self, channels=9, classes=6, hidden_size=64):
        super().__init__()
        self.lstm = nn.LSTM(channels, hidden_size, batch_first=True)
        self.classifier = nn.Linear(hidden_size, classes)

    def forward(self, windows):
        # Last recurrent hidden state summarizes the 128-step sequence.
        _, (hidden, _) = self.lstm(windows)
        return self.classifier(hidden[-1])


def build_model(name):
    if name == "cnn":
        return ActivityCNN()
    if name == "lstm":
        return ActivityLSTM()
    raise ValueError(f"Unknown model: {name}")
