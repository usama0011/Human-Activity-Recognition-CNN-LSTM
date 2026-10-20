import numpy as np
import pytest
import torch

from har.dataset import (ACTIVITIES, ChannelStandardizer, SIGNALS,
                         load_split, train_validation_indices)
from har.models import build_model
from har.training import evaluate, loader, train_epoch


def test_sensor_count():
    assert len(SIGNALS) == 9
    assert len(ACTIVITIES) == 6


@pytest.mark.parametrize("name", ["cnn", "lstm"])
def test_model_shapes(name):
    y = build_model(name)(torch.randn(3, 128, 9))
    assert y.shape == (3, 6)
    assert torch.isfinite(y).all()


@pytest.mark.parametrize("name", ["cnn", "lstm"])
def test_backward_pass(name):
    model = build_model(name)
    loss = torch.nn.functional.cross_entropy(model(torch.randn(2, 128, 9)), torch.tensor([1, 4]))
    loss.backward()
    assert any(p.grad is not None for p in model.parameters())


def test_participants_do_not_cross_split():
    people = np.repeat(np.arange(1, 22), 3)
    train, valid = train_validation_indices(people, seed=5)
    assert len(train) + len(valid) == len(people)
    assert not (set(people[train]) & set(people[valid]))


def test_split_is_repeatable():
    subjects = np.repeat(np.arange(1, 11), 8)
    a, b = train_validation_indices(subjects)
    c, d = train_validation_indices(subjects)
    assert np.array_equal(a, c) and np.array_equal(b, d)


def test_split_needs_two_people():
    with pytest.raises(ValueError):
        train_validation_indices(np.ones(8, dtype=int))


def test_standardization_uses_fit_data_only():
    train = np.full((4, 128, 9), 2.0, dtype=np.float32)
    test = np.full((1, 128, 9), 100.0, dtype=np.float32)
    scaler = ChannelStandardizer().fit(train)
    assert np.allclose(scaler.transform(train), 0.0)
    assert np.all(scaler.transform(test) > 1.0)


def test_bad_model_name():
    with pytest.raises(ValueError):
        build_model("transformer")


def test_split_validation():
    with pytest.raises(ValueError):
        load_split("does_not_exist", "other")


def test_file_loading(tmp_path):
    root = tmp_path / "UCI HAR Dataset"
    sensor = root / "train" / "Inertial Signals"
    sensor.mkdir(parents=True)
    for index, name in enumerate(SIGNALS):
        np.savetxt(sensor / f"{name}_train.txt", np.full((3, 128), index + 1))
    np.savetxt(root / "train" / "y_train.txt", [1, 6, 2], fmt="%d")
    np.savetxt(root / "train" / "subject_train.txt", [1, 2, 1], fmt="%d")
    x, y, subjects = load_split(tmp_path, "train")
    assert x.shape == (3, 128, 9)
    assert y.tolist() == [0, 5, 1]
    assert subjects.tolist() == [1, 2, 1]
    assert x[0, 0, 8] == 9.0


def test_training_epoch_and_metrics():
    torch.manual_seed(2)
    torch.set_num_threads(2)
    x = np.random.default_rng(1).normal(size=(12, 128, 9)).astype(np.float32)
    y = np.array([0, 1, 2, 3, 4, 5] * 2, dtype=np.int64)
    batches = loader(x, y, batch_size=4)
    model = build_model("cnn")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss = train_epoch(model, batches, optimizer, torch.device("cpu"))
    metrics = evaluate(model, batches, torch.device("cpu"))
    assert np.isfinite(loss)
    assert 0 <= metrics["accuracy"] <= 1
    assert 0 <= metrics["macro_f1"] <= 1
    assert len(metrics["confusion_matrix"]) == 6


def test_checkpoint_roundtrip(tmp_path):
    model = build_model("cnn")
    x = torch.randn(2, 128, 9)
    model.eval()
    with torch.no_grad():
        reference = model(x)
    path = tmp_path / "model.pt"
    torch.save(model.state_dict(), path)
    loaded = build_model("cnn")
    loaded.load_state_dict(torch.load(path, weights_only=True))
    loaded.eval()
    with torch.no_grad():
        assert torch.allclose(reference, loaded(x))
