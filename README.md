# Deep Learning and EEG Dynamics for Human Attention Prediction

Classifying three mental attention states (**focused, unfocused, drowsy**) from consumer-grade EEG, with a strict
recording-independent evaluation protocol and a comparison between hand-crafted spectral/temporal features and
deep networks trained on raw EEG windows.

> **Status:** notebooks 01 to 04 have been run on the full dataset. Notebook 05 (deep learning) is written but its
> results are not yet included below. Sections marked *pending* are filled in after running it.

---

## Overview

This project builds a complete, reproducible EEG pipeline on the public *EEG Data for Mental Attention State Detection*
dataset: dataset inspection, quality control and preprocessing, feature analysis, classical machine-learning baselines,
and deep learning on raw EEG windows.

Three design principles run through every notebook:

1. **No window-level random splits.** Windows from one recording overlap in time and share recording-specific
   characteristics. All evaluation holds out whole recordings.
2. **Say what is not known.** Labels are derived from the experimental protocol, and the mapping from recordings to
   participants is unresolved. Both limitations are stated wherever they affect an interpretation.
3. **Look at the data before modelling.** The column layout, active channels, label source and class balance were
   verified from the files rather than assumed.

## Research Question

> Which changes in EEG spectral and temporal dynamics accompany the transition from focused to unfocused to drowsy
> states, and can a model trained on **raw** EEG windows predict the state of a **previously unseen recording**
> better than classifiers built on hand-crafted features?

## Objectives

1. Understand and document the dataset structure (MATLAB files, 25 columns, labels, sampling rate).
2. Build a quality-controlled preprocessing pipeline: channel selection, filtering, windowing, artifact rejection,
   scaling.
3. Characterise how band power, band ratios and time-domain features differ between attention states.
4. Establish baselines with classical models under recording-grouped cross-validation, and quantify how much random
   splitting would overstate performance.
5. Train and compare a 1D CNN, a CNN-LSTM and a Transformer on raw windows under the same folds.
6. Test how much performance depends on signal amplitude (a possible non-neural confound).
7. Report limitations honestly: protocol-derived labels, unresolved subject mapping, small number of recordings.

## Dataset

| Item | Value |
|---|---|
| Source | Kaggle: [`inancigdem/eeg-data-for-mental-attention-state-detection`](https://www.kaggle.com/datasets/inancigdem/eeg-data-for-mental-attention-state-detection) |
| Recordings | 34 MATLAB files (`eeg_record1.mat` to `eeg_record34.mat`) |
| Device | EMOTIV EPOC headset, sampling rate 128 Hz |
| Layout | `o.data` is `n_samples x 25`; columns 4 to 17 are the 14 EEG channels |
| EEG channels | AF3, F7, F3, FC5, T7, P7, O1, O2, P8, T8, FC6, F4, F8, AF4 |
| Other columns | packet counter, interpolation flag, contact quality, gyroscope X/Y, timestamps, function id/value, marker, sync signal |
| Recording length | about 28 to 63 minutes |
| Reported participants | 5 (not identifiable from the files, see limitations) |

**Labels are not stored in the files.** The `tag` field is empty and `marker` is a sparse pulse channel. Labels are
reconstructed from the experimental protocol: roughly the first 10 minutes are *focused*, the next 10 minutes
*unfocused*, and the remainder *drowsy*. They should be described as *protocol-derived*, not as physiologically
verified ground truth.

**Practical notes**

- The Kaggle download contains a nested duplicate (`EEG Data/EEG Data/`). Using a recursive glob returns 68 files
  (34 x 2). The notebooks read only the deepest folder to get the 34 unique recordings.
- The raw dataset is **not** included in this repository. It is obtained with `kagglehub` (see [Installation](#installation)).

**Citation of the dataset**

> C. I. Acı, M. Kaya, Y. Mishchenko, "Distinguishing mental attention states of humans via an EEG-based passive BCI
> using machine learning methods", *Expert Systems with Applications*, vol. 134, pp. 153-166, 2019.

## Methodology

### Preprocessing (notebook 02)

| Step | Details |
|---|---|
| Quality control | duration, packet-counter jumps, interpolated fraction, per-channel standard deviation across all 34 files |
| Channel selection | automatic flat-versus-active detection from the channel standard deviations, with a manual override; 14 channels retained on the full dataset |
| Filtering | DC removal, then zero-phase Butterworth band-pass 1 to 45 Hz (order 4), applied to the whole continuous recording |
| Labels | assigned per sample from elapsed time (0 to 10 min focused, 10 to 20 min unfocused, remainder drowsy) |
| Windowing | 2 s windows (256 samples), 1 s step (50 % overlap) |
| Window rejection | edge of recording, window straddling two label segments, packet-counter gap, flat channel, artifact (peak-to-peak far above the channel's own median) |
| Scaling | robust z-score (median and MAD) computed **per recording** from that recording only |
| Storage | continuous signal (`.npz`) plus a window index table; windows are not duplicated on disk |

After cleaning: **47,354 windows** from 34 recordings.

### Features (notebook 03)

- **Spectral:** Welch PSD per window and channel; absolute and relative power in delta (1 to 4 Hz), theta (4 to 8 Hz),
  alpha (8 to 13 Hz), beta (13 to 30 Hz), gamma (30 to 45 Hz).
- **Ratios:** theta/beta, alpha/beta, (theta + alpha)/beta.
- **Temporal:** mean, standard deviation, RMS, peak-to-peak, line length, zero-crossing rate, Hjorth activity,
  mobility and complexity.
- **Analysis:** channel-averaged features, schematic electrode topography, Kruskal-Wallis tests (exploratory),
  correlation matrix, PCA coloured by state and by recording.

### Models (notebooks 04 and 05)

| Family | Models | Input |
|---|---|---|
| Baselines | majority class, Logistic Regression, RBF SVM, Random Forest (XGBoost optional) | hand-crafted features (absolute power dropped, since it depends on per-recording scaling) |
| Deep learning | 1D CNN, CNN-LSTM, Transformer (about 50k to 90k parameters each) | raw windows, 14 channels x 256 samples |

### Evaluation protocol

- `StratifiedGroupKFold` (5 folds, seed 42) grouped by **recording**; the deep-learning notebook uses exactly the same
  folds, verified by an assertion.
- Headline metric: **balanced accuracy** (61 % of windows are drowsy, so plain accuracy is misleading). Accuracy and
  macro-F1 are also reported.
- Deep models use early stopping on a validation set made of held-out *training recordings*; test recordings are
  never used for model selection.
- Leave-one-recording-out is used on the best baseline for a per-recording view.
- The evaluation is **recording-independent, not subject-independent** (see limitations).

## Experiments

| # | Experiment | Notebook | Status |
|---|---|---|---|
| E1 | Dataset structure, channel layout, raw signal inspection | 01 | done |
| E2 | Quality control, channel selection, filtering, windowing, artifact rejection | 02 | done |
| E3 | Spectral and temporal differences between states, statistical ranking, PCA | 03 | done |
| E4 | Leakage check: random window split versus recording-grouped split | 04 | done |
| E5 | Classical baselines with recording-grouped cross-validation, leave-one-recording-out | 04 | done |
| E6 | 1D CNN, CNN-LSTM, Transformer on raw windows, same folds as E5 | 05 | written, results pending |
| E7 | Amplitude ablation: raw-scaled input versus per-window z-score | 05 | written, results pending |
| E8 | Consolidated figures and interpretability (channel occlusion, saliency by band) | 06 | planned |

## Results

### Data after cleaning

| | Value |
|---|---|
| Kept windows / recordings | 47,354 / 34 |
| Class distribution | focused 7,765 (16 %), unfocused 10,706 (23 %), drowsy 28,883 (61 %) |
| Windows per recording | very uneven, from 1 to 3,623 (several recordings lose most of their windows to cleaning) |
| Majority-class accuracy | 0.624 |

### What changes in the EEG (notebook 03)

- Channel-averaged **relative alpha power increases** from focused to unfocused to drowsy (medians roughly 0.14, 0.15,
  0.20), while **relative delta power decreases**. Theta is nearly flat across states.
- By Kruskal-Wallis ranking, the strongest separators are relative alpha, the alpha/beta ratio and Hjorth complexity.
  The theta/beta ratio is among the weakest on this dataset. With tens of thousands of non-independent windows these
  tests are exploratory and the H values are inflated, so they are used only to rank features.
- In PCA the three states overlap heavily while recordings form visible clusters, so a strong recording effect is
  present.

### Baselines (notebook 04, recording-grouped 5-fold CV)

| Model | Balanced accuracy | Accuracy | Macro-F1 |
|---|---|---|---|
| Majority class | 0.333 | 0.624 | 0.256 |
| Logistic Regression | **0.605 +/- 0.050** | 0.648 | 0.582 |
| RBF SVM | 0.586 +/- 0.036 | 0.685 | 0.578 |
| Random Forest | 0.478 +/- 0.045 | 0.685 | 0.475 |

- **Effect of leakage:** the same Logistic Regression scores 0.703 balanced accuracy with a random window split and
  0.577 with the recording-grouped split, a gap of 0.126. Random window splitting clearly overstates performance.
- **Confusion pattern (Logistic Regression):** per-class recall focused 0.62, unfocused 0.48, drowsy 0.69. Unfocused is
  the hardest class and is confused with both neighbours.
- **Per-recording variability:** leave-one-recording-out balanced accuracy ranges from about 0.26 to 0.81. This table
  includes recordings with very few windows and should be read with that in mind.
- The most influential Logistic Regression features are amplitude-based (RMS, standard deviation, line length). This
  can reflect physiology but can also reflect slow non-neural drift, since labels depend only on elapsed time. This
  motivates the amplitude ablation in notebook 05.

### Deep learning (notebook 05)

*Pending: fill in after running `05_deep_learning.ipynb`. The notebook writes its results to
`data/processed/deep_learning_results/`.*

| Model | Input | Balanced accuracy | Accuracy | Macro-F1 |
|---|---|---|---|---|
| 1D CNN | robust-scaled | pending | pending | pending |
| CNN-LSTM | robust-scaled | pending | pending | pending |
| Transformer | robust-scaled | pending | pending | pending |
| 1D CNN (ablation) | per-window z-score | pending | pending | pending |

With fold-to-fold standard deviations of about 0.05 in the baselines, differences of a few points between models should
not be treated as real. The notebook prints this comparison explicitly.

### Limitations

- **Labels are protocol-derived**, not verified per participant, and they are perfectly confounded with elapsed time
  in the session.
- **The mapping of the 34 recordings to the (reportedly) 5 participants is unresolved.** Splitting by recording
  prevents leakage within a recording but not between recordings of the same person. Results should be called
  *recording-independent*, not *subject-independent*.
- Artifact handling is a simple amplitude rule; no ICA or ocular-artifact regression was applied.
- Several recordings contribute very few windows after cleaning.
- Baseline hyper-parameters are defaults and deep models are trained with one seed and no search.

## Repository Structure

```
eeg-attention-deep-learning/
├── README.md
├── notebooks/
│   ├── 01_dataset_exploration.ipynb      # structure, channels, raw signal
│   ├── 02_eeg_preprocessing.ipynb        # QC, filtering, windowing, rejection, scaling
│   ├── 03_feature_analysis.ipynb         # spectral/temporal features, statistics, PCA
│   ├── 04_baseline_models.ipynb          # classical ML, recording-grouped evaluation
│   ├── 05_deep_learning.ipynb            # 1D CNN, CNN-LSTM, Transformer on raw windows
│   └── 06_results_visualization.ipynb    # planned: report figures and interpretability
└── data/
    ├── README.md                         # how to obtain the dataset
    ├── raw/                              # not tracked by Git
    ├── interim/                          # QC tables, subject-map template
    └── processed/
        ├── continuous/                   # filtered, scaled signals (.npz)
        ├── windows.csv                   # window index with keep/reject reasons
        ├── preprocessing_config.json     # exact preprocessing parameters
        ├── eeg_features.csv              # one row per kept window
        ├── baseline_results/             # notebook 04 outputs
        └── deep_learning_results/        # notebook 05 outputs
```

Planned: a `src/` package (for example `src/data/loader.py`) once the notebooks have stabilised. Raw data and large
processed files should stay out of Git (add `data/raw/` and `data/processed/` to `.gitignore`).

## Installation

Python 3.10 or newer is recommended.

```bash
git clone <your-repository-url>
cd eeg-attention-deep-learning

python -m venv .venv
# Windows: .venv\Scripts\activate      Linux/macOS: source .venv/bin/activate

pip install numpy scipy pandas matplotlib scikit-learn jupyter kagglehub tqdm
pip install torch          # see https://pytorch.org/get-started/locally/ for a CUDA build
pip install xgboost        # optional, used by notebook 04 if present
```

**Download the dataset** (requires a Kaggle account configured for `kagglehub`):

```python
import kagglehub
path = kagglehub.dataset_download("inancigdem/eeg-data-for-mental-attention-state-detection")
print(path)
```

The dataset is cached by `kagglehub` and not copied into the repository. Notebook 01 contains a `DATASET_PATH`
variable that must point to this location. Notebooks 02 onward read the path from the `EEG_DATASET_PATH` environment
variable if it is set, otherwise from a default in their configuration cell.

## Usage

Run the notebooks **in order**, from the `notebooks/` directory, since each one reads files written by the previous one.

| Notebook | Reads | Writes |
|---|---|---|
| 01 | raw `.mat` files | `data/interim/recording_summary.csv` |
| 02 | raw `.mat` files | `data/processed/continuous/`, `windows.csv`, `preprocessing_config.json`, QC tables |
| 03 | outputs of 02 | `data/processed/eeg_features.csv` |
| 04 | `eeg_features.csv` | `data/processed/baseline_results/` |
| 05 | outputs of 02 and 04 | `data/processed/deep_learning_results/` |

Notes:

- Notebook 05 should be run once with `QUICK_RUN = True` (one model, one fold, two epochs) to check the setup and
  estimate the time per epoch before a full run. A GPU is helpful but not required.
- All tunable settings (filter band, window length, rejection thresholds, training hyper-parameters) are at the top of
  each notebook.
- Notebook 02 writes `data/interim/subject_map_TEMPLATE.csv`. Once a recording-to-participant mapping is established,
  save it as `subject_map.csv` and use it as the grouping variable.

## Future Work

- **Resolve the subject mapping** (34 recordings to participants) and repeat the evaluation with leave-one-subject-out
  or group k-fold, which would allow the claim *subject-independent*.
- **Investigate heavy window loss** in a few recordings by checking rejection reasons per recording.
- **Control for the time confound:** compare against amplitude-free input, and design checks that separate attention
  effects from slow drift over the session.
- **Repeat deep-learning runs with several seeds** and report mean and spread; tune hyper-parameters with a
  nested, recording-grouped procedure.
- **Stronger artifact handling** (ICA, ocular-artifact regression) and an alternative stricter normalisation, where
  scaling statistics come only from training recordings.
- **Interpretability and visualisation:** channel-occlusion importance, input-gradient saliency by frequency band,
  anatomically interpolated scalp maps with MNE.
- **Further evaluation:** calibration of predicted probabilities, ensembling features with raw-signal models,
  temporal smoothing of window predictions.

## References

1. C. I. Acı, M. Kaya, Y. Mishchenko, "Distinguishing mental attention states of humans via an EEG-based passive BCI
   using machine learning methods", *Expert Systems with Applications*, 134, 153-166, 2019.
2. EEG Data for Mental Attention State Detection, Kaggle dataset:
   https://www.kaggle.com/datasets/inancigdem/eeg-data-for-mental-attention-state-detection
3. "Optimized preprocessing and Tiny ML for Attention State Classification", arXiv:2303.11371
   (related work that uses this dataset).