"""Synthetic verification only: these are not ASVspoof performance results."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import pytest
import soundfile as sf
from sklearn.metrics import roc_curve

import extract_mfcc as extraction
from config import MFCCConfig
from evaluate import equal_error_rate, evaluate_test
from train_svm import train_and_select


def test_eer_score_direction_and_ties():
    labels = [0, 0, 1, 1]
    for scores, expected in [([0, 0, 1, 1], 0), ([1, 1, 0, 0], 1), ([0, 0, 0, 0], .5)]:
        fpr, tpr, _ = roc_curve(labels, scores, pos_label=1, drop_intermediate=False)
        assert equal_error_rate(fpr, tpr) == pytest.approx(expected)
    assert equal_error_rate([0, .2, .6, 1], [0, .5, .9, 1]) == pytest.approx(.35)


def tiny_splits(tmp_path):
    splits = {}
    for i, name in enumerate(("train", "val", "test")):
        path = tmp_path / f"{name}.wav"
        t = np.arange(2400) / 24000
        # Stereo input at a different sample rate exercises both conversion steps.
        sf.write(path, np.column_stack([np.sin(2*np.pi*(300+i*100)*t), np.cos(2*np.pi*500*t)]), 24000)
        splits[name] = pd.DataFrame({"file_id": [name], "label": ["spoof"],
                                     "audio_path": [str(path)], "resolved_path": [str(path)]})
    return splits


def test_audio_cache_and_alignment(tmp_path, monkeypatch):
    splits, config = tiny_splits(tmp_path), MFCCConfig()
    first = extraction.load_or_extract(splits, config, tmp_path)
    assert first["X_train"].shape == (1, 40)
    def forbidden(*args):
        raise AssertionError("valid cache should avoid extraction")
    monkeypatch.setattr(extraction, "extract_one", forbidden)
    cached = extraction.load_or_extract(splits, config, tmp_path)
    np.testing.assert_array_equal(first["X_test"], cached["X_test"])
    cached["X_train"][0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN/infinite"):
        extraction.validate_features(cached, splits, config)
    cached["X_train"] = first["X_train"].copy()
    cached["train_file_ids"] = np.array(["wrong"])
    with pytest.raises(ValueError, match="alignment"):
        extraction.validate_features(cached, splits, config)
    # A changed label invalidates cache, leading to reported extraction failures.
    splits["train"].loc[0, "label"] = "bonafide"
    with pytest.raises(RuntimeError, match="3 audio files failed"):
        extraction.load_or_extract(splits, config, tmp_path)
    assert len(json.loads((tmp_path / "extraction_failures.json").read_text())) == 3


def test_missing_audio_never_saves_partial_cache(tmp_path):
    splits = tiny_splits(tmp_path)
    splits["val"].loc[0, "resolved_path"] = str(tmp_path / "absent.wav")
    with pytest.raises(RuntimeError, match="1 audio files failed"):
        extraction.load_or_extract(splits, MFCCConfig(), tmp_path)
    assert not (tmp_path / "mfcc_features.npz").exists()
    assert json.loads((tmp_path / "extraction_failures.json").read_text())[0]["file_id"] == "val"


def test_split_validation_and_overlap(tmp_path, monkeypatch):
    (tmp_path / "dataset").mkdir()
    monkeypatch.setattr(extraction, "EXPECTED", {n: {"bonafide": 1, "spoof": 1} for n in extraction.SPLIT_FILES})
    for name, filename in extraction.SPLIT_FILES.items():
        pd.DataFrame({"file_id": [name+"0", name+"1"], "label": ["bonafide", "spoof"],
                      "audio_path": [name+"0.flac", name+"1.flac"]}).to_csv(tmp_path / "dataset" / filename, index=False)
    splits = extraction.read_splits(tmp_path)
    assert splits["train"].file_id.tolist() == ["train0", "train1"]
    path = tmp_path / "dataset" / "test.csv"
    frame = pd.read_csv(path)
    frame.loc[0, "file_id"] = "train0"
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="Data leakage"):
        extraction.read_splits(tmp_path)
    assert extraction.resolve_audio_path("/old/data/a.flac", tmp_path, "/old") == tmp_path / "data/a.flac"


def test_training_selection_scaler_and_evaluation(tmp_path):
    rng = np.random.default_rng(42)
    x_train = rng.normal(size=(40, 40))
    y_train = np.tile([0, 1], 20)
    x_train[:, 0] += 4 * y_train
    x_val = rng.normal(size=(16, 40)) + .4
    y_val = np.tile([0, 1], 8)
    x_val[:, 0] += 4 * y_val
    model, scaler, table, best = train_and_select(x_train, y_train, x_val, y_val)
    assert len(table) == 20
    assert best["validation_f1"] == table.validation_f1.max()
    np.testing.assert_allclose(scaler.mean_, x_train.mean(axis=0))
    assert scaler.n_samples_seen_ == 40
    np.testing.assert_array_equal(model.classes_, [0, 1])
    x_test = rng.normal(size=(12, 40))
    y_test = np.tile([0, 1], 6)
    x_test[:, 0] += 4 * y_test
    metrics, cm, report = evaluate_test(model, scaler, x_test, y_test, tmp_path)
    assert cm.sum() == 12 and 0 <= metrics["eer"] <= 1
    assert "Spoof" in report
    assert (tmp_path / "roc_curve.png").is_file()
    assert (tmp_path / "confusion_matrix.png").is_file()
    np.testing.assert_allclose(scaler.mean_, x_train.mean(axis=0))


def test_runner_writes_all_artifacts(tmp_path, monkeypatch):
    import run_mfcc_experiment as runner
    rng = np.random.default_rng(42)
    splits, data = {}, {}
    for name, count in [("train", 20), ("val", 8), ("test", 8)]:
        y = np.tile([0, 1], count // 2)
        ids = [f"{name}{i}" for i in range(count)]
        splits[name] = pd.DataFrame({"file_id": ids, "label": np.where(y, "spoof", "bonafide"),
                                     "audio_path": ids, "resolved_path": ids})
        data[f"X_{name}"] = rng.normal(size=(count, 40))
        data[f"X_{name}"][:, 0] += y * 4
        data[f"y_{name}"] = y
        data[f"{name}_file_ids"] = np.array(ids)
    monkeypatch.setattr(runner, "read_splits", lambda *args: splits)
    def features(splits, config, output_dir, force=False):
        np.savez_compressed(output_dir / "mfcc_features.npz", **data)
        return data
    monkeypatch.setattr(runner, "load_or_extract", features)
    record = runner.run(tmp_path)
    out = tmp_path / "results" / "mfcc_svm"
    expected = {"mfcc_features.npz", "validation_results.csv", "test_metrics.json",
                "test_metrics.csv", "confusion_matrix.png", "roc_curve.png",
                "svm_model.joblib", "scaler.joblib", "experiment_summary.txt"}
    assert expected <= {path.name for path in out.iterdir()}
    saved = json.loads((out / "test_metrics.json").read_text())
    assert saved["test_metrics"] == record["test_metrics"]
    import joblib
    model = joblib.load(out / "svm_model.joblib")
    scaler = joblib.load(out / "scaler.joblib")
    assert model.predict(scaler.transform(data["X_test"])).shape == (8,)
