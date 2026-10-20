# Human Activity Recognition: Experiment notes

**Project status:** Source code written; software smoke tests required; real-data model evaluation pending.

## Motivation

Compare two basic deep-learning architectures for recognizing six everyday activities from sequential smartphone inertial signals. The question is whether an LSTM or 1D-CNN generalizes better to held-out individuals on the official UCI HAR dataset, under a shared training procedure.

## Dataset and methodology

- Official UCI HAR Smartphones dataset (2013), raw inertial-signal windows; input (128, 9).
- Six class labels, subject-disjoint official test split.
- Internal validation: hold out 20% of training participants using a fixed seed.
- Normalization: channel means/std from training participants only.
- Models: two-layer 1D-CNN baseline, single-layer LSTM baseline.
- Loss: cross-entropy. Optimizer: Adam. Model selection: validation macro F1.
- Report both test accuracy and macro F1, including confusion matrices.

## Steps to reproduce

1. Download UCI HAR data from its official archive.
2. `python prepare_data.py --zip "<downloaded UCI ZIP>"`
3. `python -m pytest -q`
4. `python train.py --model both --epochs 15 --seed 42`
5. Read `outputs/cnn/metrics.json` and `outputs/lstm/metrics.json`.

## Results template — fill with genuine measured results

| Model | Accuracy (test) | Macro-F1 (test) | Epochs run |
|---|---|---|---|
| 1D-CNN | Pending real experiment | Pending real experiment | — |
| LSTM | Pending real experiment | Pending real experiment | — |

## Limitations

The dataset uses a smartphone at the waist and a small population. Overlapping windows are correlated within people, so partitioning by participant is important. These results are not a clinical evaluation and do not measure fall detection or medical use.

## Next work

- Repeat experiments across several development seeds without choosing parameters on the held-out official test subjects.
- Compare class-level errors for sitting vs standing and walking vs stair walking.
- Measure parameter counts and inference latency on CPU.
- Document how sensor placement or noise may affect predictions, if tested.

## Citation

Reyes-Ortiz, J., et al. (2013). *Human Activity Recognition Using Smartphones*. UCI ML Repository. https://doi.org/10.24432/C54S4K.
