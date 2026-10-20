# Human Activity Recognition with CNN and LSTM

A Python project that classifies six everyday activities from smartphone motion-sensor recordings. It uses **raw inertial-signal windows** from the UCI HAR dataset, not the precomputed 561-feature table.

This repository is a learning/research implementation of two baseline neural networks. It does **not** detect falls, provide medical diagnoses or demonstrate clinical validation.

## What the project predicts

1. Walking
2. Walking upstairs
3. Walking downstairs
4. Sitting
5. Standing
6. Laying down

Each input sample contains **128 time steps × 9 sensor channels** (accelerometer and gyroscope-derived signals). The original dataset was collected from **30 volunteers** at **50 Hz**. The supplied official split separates participants into train and test groups.

## Models

- **1D-CNN:** two temporal convolution layers, temporal pooling and a six-class classifier.
- **LSTM:** one recurrent LSTM layer and a classifier applied to the last hidden state.

Both models receive the same 128×9 input and are evaluated using **test accuracy, macro F1-score and confusion matrices**. The official participant-disjoint test split is preserved. Validation also holds out participants from the official training portion, and standardization uses only training participants.

## Install and run

Requires Python 3.10+.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

### 1. Get the dataset

**Option A:** Download [UCI HAR Dataset](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones) (click Download) and run:

```bash
python prepare_data.py --zip "path/to/UCI HAR Dataset.zip"
```

**Option B:** Try the included downloader, which requires internet access:

```bash
python prepare_data.py
```

Once extracted, the expected folder is `data/UCI HAR Dataset/train/Inertial Signals/`.

### 2. Train and compare both models

```bash
python train.py --model both --epochs 15
```

For a shorter trial run:

```bash
python train.py --model cnn --epochs 2
```

To change the batch size: `--batch-size 64`. To use another output folder: `--output results`.

### 3. Inspect results

After **training on the real UCI dataset**, the script saves per-model results under `outputs/cnn/` and `outputs/lstm/`:

- `best_model.pt` — checkpoint chosen using validation macro-F1
- `normalization.npz` — channel means and standard deviations from training subjects
- `history.csv` — loss and validation metric per epoch
- `metrics.json` — test accuracy, macro F1 and per-class precision/recall
- `confusion_matrix.csv` — six-class confusion matrix

To predict one held-out example after training:

```bash
python predict_sample.py --model cnn --index 0
```

To create a confusion-matrix figure from a saved experiment:

```bash
python plot_results.py --model cnn
python plot_results.py --model lstm
```

These result files are excluded from Git to avoid including model weights or unverified experiment results by accident. If reporting metrics, include your actual run configuration and the generated result files (except large checkpoints as appropriate).

### 4. Check the implementation

```bash
python -m pytest -q
```

Tests check model shapes, backpropagation, participant splitting, train-only normalization, raw-signal file loading, small-batch training and saving/loading weights. **Passing these tests does not establish real-data classification performance.**

## Project layout

```text
har/dataset.py       UCI HAR data reader, subject split and normalization
har/models.py        CNN and LSTM architectures
har/training.py      Training step and evaluation metrics
prepare_data.py      Dataset ZIP downloader/extractor
train.py             Experiment entry point
tests/               Unit and smoke tests
docs/experiment.md   Experiment notes and report template
predict_sample.py    Inspect a held-out activity prediction
plot_results.py      Plot an actual experiment confusion matrix
```

## Limitations

- The reference dataset contains smartphone motion signals, **not EMG or other biomedical signals**.
- Human activities come from a fixed, laboratory-recorded set of six labels; performance on other people, phones or placements could differ.
- Accuracy and F1 should not be claimed until a real-dataset run is complete. At packaging time, 14 software tests passed; synthetic end-to-end smoke tests completed; real-data performance remains unmeasured.
- This is an implementation of standard pretrained-free neural-network architectures, **not a novel published method**.

## Dataset credit

Reyes-Ortiz, J., Anguita, D., Ghio, A., Oneto, L., & Parra, X. (2013). *Human Activity Recognition Using Smartphones*. UCI Machine Learning Repository. https://doi.org/10.24432/C54S4K. Dataset page: https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones

The UCI dataset webpage currently lists the dataset under CC BY 4.0. Check the official source for any distribution or reuse conditions before publishing dataset files. This repository distributes **code only**, not the dataset.
