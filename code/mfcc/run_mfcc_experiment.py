"""Colab/local entry point. Never creates or shuffles dataset splits."""
import argparse
import importlib.metadata
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from config import LABELS, MFCCConfig, RANDOM_STATE
from extract_mfcc import cache_metadata, load_or_extract, read_splits
from train_svm import train_and_select
from evaluate import evaluate_test


def run(project_path, audio_path_prefix=None, force_extract=False):
    project_path = Path(project_path).expanduser().resolve()
    config = MFCCConfig()
    np.random.seed(RANDOM_STATE)
    print("MFCC configuration:", json.dumps(config.to_dict(), indent=2))
    print(f"Project path: {project_path}; random_state={RANDOM_STATE}; labels={LABELS}")
    splits = read_splits(project_path, audio_path_prefix)
    output_dir = project_path / "results" / "mfcc_svm"
    output_dir.mkdir(parents=True, exist_ok=True)
    data = load_or_extract(splits, config, output_dir, force=force_extract)
    model, scaler, table, best = train_and_select(
        data["X_train"], data["y_train"], data["X_val"], data["y_val"])
    table.to_csv(output_dir / "validation_results.csv", index=False)
    # The selected training-fitted model is frozen here; there is no train+val refit.
    metrics, cm, report = evaluate_test(model, scaler, data["X_test"], data["y_test"], output_dir)
    joblib.dump(model, output_dir / "svm_model.joblib")
    joblib.dump(scaler, output_dir / "scaler.joblib")
    distributions = {name: frame.label.value_counts().to_dict() for name, frame in splits.items()}
    record = {"dataset": "ASVspoof 2021 DF", "feature_type": "MFCC",
              "feature_dimension": 40, "configuration": config.to_dict(),
              "random_state": RANDOM_STATE, "label_mapping": LABELS,
              "class_distributions": distributions, "best_model": best,
              "test_metrics": metrics, "confusion_matrix": cm.tolist(),
              "confusion_matrix_order": "rows=true, columns=predicted; [Bonafide, Spoof]",
              "classification_report": report,
              "cache_metadata": cache_metadata(splits, config),
              "package_versions": {p: importlib.metadata.version(p) for p in
                                   ("numpy", "pandas", "librosa", "scikit-learn", "matplotlib", "joblib")}}
    (output_dir / "test_metrics.json").write_text(json.dumps(record, indent=2, allow_nan=False))
    pd.DataFrame([metrics]).to_csv(output_dir / "test_metrics.csv", index=False)
    lines = ["=" * 30, "MFCC + SVM FINAL RESULTS", "=" * 30,
             "Dataset: ASVspoof 2021 DF", "Feature type: MFCC", "MFCC coefficients: 20",
             "Feature dimension: 40", f"Sample rate: {config.sample_rate} Hz",
             f"Configuration: {json.dumps(config.to_dict(), sort_keys=True)}",
             f"Random state: {RANDOM_STATE}", "Labels: bonafide=0, spoof=1 (positive)"]
    for name, frame in splits.items():
        lines.append(f"{name} samples: {len(frame)}; class distribution: {distributions[name]}")
    lines += ["", "Best SVM:", f"Kernel: {best['kernel']}", f"C: {best['C']}",
              f"Gamma: {best['gamma'] if best['gamma'] is not None else 'not applicable'}",
              "", "VALIDATION RESULTS"]
    lines += [f"{key}: {value:.6f}" for key, value in best.items() if key.startswith("validation_")]
    lines += ["", "TEST RESULTS"] + [f"{key}: {value:.6f}" for key, value in metrics.items()]
    lines += ["EER method: linear interpolation of empirical ROC at FPR = 1 - TPR",
              record["confusion_matrix_order"], str(cm), "", report]
    summary = "\n".join(lines)
    (output_dir / "experiment_summary.txt").write_text(summary + "\n")
    print(summary)
    print(f"Outputs saved to {output_dir}")
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-path", type=Path, default=Path("/content/drive/MyDrive/Spoken_Project"))
    parser.add_argument("--audio-path-prefix", help="Explicit old absolute project prefix in CSV audio paths; "
                        "replace with --project-path (for moving Colab data to a local machine).")
    parser.add_argument("--force-extract", action="store_true", help="Rebuild features even with a valid cache")
    args = parser.parse_args()
    run(args.project_path, args.audio_path_prefix, args.force_extract)
