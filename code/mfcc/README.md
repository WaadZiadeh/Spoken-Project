# Waad's experiment: MFCC + SVM

Independent implementation; the existing LFCC notebook is untouched. The runner
requires your **existing** ASVspoof 2021 DF balanced part00 CSV splits and audio.
Neither is included in this checkout. It never generates, shuffles, or rewrites splits.

## Google Colab

Copy this repository's `code/mfcc/` folder into your existing Drive project so that
it contains `code/mfcc/run_mfcc_experiment.py` alongside `dataset/train.csv`,
`dataset/validation.csv`, and `dataset/test.csv`. Keep your existing audio in place.
Run these cells:

```python
from google.colab import drive
drive.mount('/content/drive')
PROJECT_PATH = '/content/drive/MyDrive/Spoken_Project'
```

```python
%pip install -r {PROJECT_PATH}/code/mfcc/requirements.txt
```

```python
import subprocess
import sys
subprocess.run([
    sys.executable,
    f'{PROJECT_PATH}/code/mfcc/run_mfcc_experiment.py',
    '--project-path', PROJECT_PATH,
], check=True)
```

If Colab requests a runtime restart after installation, restart and run the mount
and path cells again before running the experiment. A CPU runtime is sufficient;
SVC does not use the Colab GPU. Extraction and the linear SVM candidates may take time.

## Local execution

From the repository root, with dependencies installed in your Python environment:

```bash
python -m pip install -r code/mfcc/requirements.txt
python code/mfcc/run_mfcc_experiment.py --project-path /path/to/Spoken_Project
```

Relative CSV `audio_path` values resolve against `--project-path`; absolute paths
are used directly. When moving a Drive project to a local computer, explicitly
map its old project prefix without editing the CSVs:

```bash
python code/mfcc/run_mfcc_experiment.py \
  --project-path /local/Spoken_Project \
  --audio-path-prefix /content/drive/MyDrive/Spoken_Project
```

This maps only paths under that prefix, preserving the relative suffix. It never
searches by basename or substitutes other recordings. Audio can live outside the
repository using absolute CSV paths.

## Method and safeguards

- Required CSV columns: `file_id`, `label`, `audio_path`. Labels must be exactly
  `bonafide` (0) or `spoof` (1). Spoof is positive for all binary metrics/scores.
- Strict counts: train 7,749 (3,874 bonafide / 3,875 spoof), validation 1,660
  (830 / 830), test 1,661 (831 / 830). Duplicate IDs and resolved paths are rejected
  both within and across splits. CSV row order is preserved.
- Librosa loads mono float32 at 16 kHz. Peak amplitude normalization matches the
  existing LFCC notebook (silence stays zero); original audio is never modified.
- 20 MFCCs, FFT 512, hop 256, 40 Slaney-normalized Mel bands, Hann window,
  centered frames with constant padding, power 2, orthonormal type-II DCT.
  See `config.py` for all settings. Vector order: 20 means, then 20 population
  standard deviations (`ddof=0`); no truncation or fixed-duration padding.
- Every extraction error reports its split, file ID, path and reason. All splits
  receive requested/success/failed/shape summaries. Any failure stops training
  and prevents saving a partial cache. Fix the reported files/paths and rerun.
- Cache validation checks ordered split contents, labels, IDs, shapes, finite
  values, extraction configuration, and NumPy/librosa versions. Audio is assumed
  immutable: content changes at the same path require `--force-extract`.
  Valid caches work without rereading audio. Invalid caches are rebuilt.
- StandardScaler fits only training features. All SVC candidates fit only training
  labels/features. The grid has 4 linear and 16 RBF candidates: C=0.1,1,10,100;
  RBF gamma=scale,0.01,0.1,1. Highest validation F1 wins; exact ties retain the
  first candidate in that order. No test data enters tuning and there is no
  train+validation refit. Random state is 42 wherever applicable.
- ROC-AUC/ROC use continuous SVC decision scores, positive toward class 1.
  EER linearly interpolates the empirical ROC where FPR equals 1−TPR; this may
  fall between attainable thresholds. Classification uses SVC's default threshold.
- Test confusion matrix rows are true labels, columns predictions, ordered
  `[Bonafide, Spoof]`: `[[True Bonafide, False Spoof], [False Bonafide, True Spoof]]`.

## Generated outputs

Under `<PROJECT_PATH>/results/mfcc_svm/`:

| File | Contents |
| --- | --- |
| `mfcc_features.npz` | X/y/file IDs for train, val, test, plus cache metadata |
| `validation_results.csv` | All 20 candidates and validation metrics |
| `test_metrics.json` | Test metrics, matrix, report, selected configuration, class counts and package versions |
| `test_metrics.csv` | One row of final test metrics |
| `confusion_matrix.png` | Labeled test confusion matrix |
| `roc_curve.png` | Test ROC, AUC and random reference |
| `svm_model.joblib` | Selected SVC, fitted on training data only |
| `scaler.joblib` | Training-fitted scaler; apply before saved model inference |
| `experiment_summary.txt` | Report-ready configuration and actual results |
| `extraction_failures.json` | Extraction errors, or empty list after successful extraction |

A run reuses valid features but retrains the candidates. Completed runs replace
MFCC outputs; copy a previous output folder before running changed configurations
if you need to retain it. After a failed run, old artifacts may still exist: only
use outputs from a successfully completed run. Outputs are ignored by Git; small
reports/plots can be deliberately added for the report after checking their provenance.
Never add audio, archives, or the generated feature/model binaries.

## Verification

```bash
python -m pip install pytest
python -m pytest code/mfcc/tests -q
```

Tests use temporary synthetic audio/features, not ASVspoof performance results.
Actual accuracy, F1, AUC and EER require running on the supplied Drive dataset.
For fair LFCC comparison, confirm that its reported run used these same CSVs and
spoof-positive score orientation; this implementation does not change LFCC code.
