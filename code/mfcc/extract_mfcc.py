"""Strict split validation and order-preserving, versioned feature caching."""
import hashlib
import json
from pathlib import Path

import librosa
import numpy as np
import pandas as pd

from config import EXPECTED, LABELS, SPLIT_FILES


def resolve_audio_path(value, project_path, audio_path_prefix=None):
    path = Path(value).expanduser()
    if audio_path_prefix is not None and path.is_absolute():
        try:
            path = path.relative_to(Path(audio_path_prefix))
        except ValueError:
            pass
    return (path if path.is_absolute() else project_path / path).resolve()


def read_splits(project_path, audio_path_prefix=None):
    splits = {}
    for name, filename in SPLIT_FILES.items():
        path = project_path / "dataset" / filename
        if not path.is_file():
            raise FileNotFoundError(f"Required existing split missing: {path}. No splits are generated.")
        frame = pd.read_csv(path, dtype=str, keep_default_na=False)
        required = ["file_id", "label", "audio_path"]
        if not set(required).issubset(frame.columns):
            raise ValueError(f"{path}: required columns are {required}")
        if frame[required].apply(lambda col: col.str.strip().eq("")).any().any():
            raise ValueError(f"{path}: empty required value")
        counts = frame.label.value_counts().to_dict()
        print(f"{name}: shape={frame.shape}, class distribution={counts}", flush=True)
        if counts != EXPECTED[name]:
            raise ValueError(f"{name}: expected {EXPECTED[name]}, received {counts}")
        if frame.file_id.duplicated().any():
            raise ValueError(f"{name}: duplicate file_ids within split")
        frame = frame.copy()
        frame["resolved_path"] = [str(resolve_audio_path(p, project_path, audio_path_prefix))
                                  for p in frame.audio_path]
        if frame.resolved_path.duplicated().any():
            raise ValueError(f"{name}: duplicate audio paths within split")
        for previous_name, previous in splits.items():
            for column in ("file_id", "resolved_path"):
                overlap = set(frame[column]) & set(previous[column])
                if overlap:
                    raise ValueError(f"Data leakage: {previous_name} vs {name} overlapping {column}: "
                                     f"{sorted(overlap)[:10]}")
        splits[name] = frame
    print("Verified: exact existing splits, class counts, and no ID/path overlap.")
    return splits


def extract_one(path, config):
    audio, sr = librosa.load(path, sr=config.sample_rate, mono=True,
                             dtype=np.float32, res_type=config.res_type)
    if audio.size == 0 or not np.isfinite(audio).all():
        raise ValueError("Empty or nonfinite waveform")
    if config.peak_normalize:
        peak = np.max(np.abs(audio))
        if peak > 0:
            audio = audio / peak
    mfcc = librosa.feature.mfcc(
        y=audio, sr=sr, n_mfcc=config.n_mfcc, n_fft=config.n_fft,
        hop_length=config.hop_length, n_mels=config.n_mels, window=config.window,
        center=config.center, pad_mode=config.pad_mode, power=config.power,
        dct_type=config.dct_type, norm=config.norm, lifter=config.lifter,
        fmin=0.0, fmax=config.sample_rate / 2, htk=False, mel_norm="slaney",
    )
    vector = np.concatenate((mfcc.mean(axis=1), mfcc.std(axis=1, ddof=0)))
    if vector.shape != (2 * config.n_mfcc,) or not np.isfinite(vector).all():
        raise ValueError("Invalid MFCC feature vector")
    return vector.astype(np.float32)


def cache_metadata(splits, config):
    # Ordered CSV content and resolved paths detect changes without reading audio.
    manifests = {name: frame[["file_id", "label", "audio_path", "resolved_path"]].to_dict("list")
                 for name, frame in splits.items()}
    digest = hashlib.sha256(json.dumps(manifests, sort_keys=True).encode()).hexdigest()
    return {"schema_version": 1, "config": config.to_dict(), "split_sha256": digest,
            "librosa_version": librosa.__version__, "numpy_version": np.__version__,
            "label_mapping": LABELS, "feature_order": "20 means, then 20 population standard deviations"}


def validate_features(data, splits, config):
    for name, frame in splits.items():
        x, y, ids = data[f"X_{name}"], data[f"y_{name}"], data[f"{name}_file_ids"]
        if x.shape != (len(frame), 2 * config.n_mfcc) or not np.isfinite(x).all():
            raise ValueError(f"{name}: wrong feature shape or NaN/infinite features")
        if not np.array_equal(y, frame.label.map(LABELS).to_numpy(dtype=np.int64)):
            raise ValueError(f"{name}: feature/label alignment mismatch")
        if not np.array_equal(ids, frame.file_id.to_numpy(dtype=str)):
            raise ValueError(f"{name}: feature/file_id alignment mismatch")


def load_or_extract(splits, config, output_dir, force=False):
    cache = output_dir / "mfcc_features.npz"
    metadata = cache_metadata(splits, config)
    if cache.exists() and not force:
        try:
            with np.load(cache, allow_pickle=False) as archive:
                if json.loads(str(archive["metadata"].item())) != metadata:
                    raise ValueError("configuration, split manifest, or library version changed")
                data = {key: archive[key] for key in archive.files if key != "metadata"}
            validate_features(data, splits, config)
        except Exception as exc:
            print(f"Invalid cache: {exc}. Re-extracting.", flush=True)
        else:
            print(f"Loaded validated cache: {cache}")
            for name, frame in splits.items():
                print(f"{name}: requested={len(frame)}, successfully processed={len(frame)}, "
                      f"failed=0, shape={data[f'X_{name}'].shape} (cached)")
            return data
    data, failures = {}, []
    for name, frame in splits.items():
        vectors = []
        for index, row in enumerate(frame.itertuples(index=False), start=1):
            try:
                vectors.append(extract_one(row.resolved_path, config))
            except Exception as exc:
                failure = {"split": name, "file_id": row.file_id,
                           "audio_path": row.resolved_path, "error": str(exc)}
                failures.append(failure)
                print(f"FAILED: {failure}", flush=True)
            if index % 250 == 0:
                print(f"{name}: attempted {index}/{len(frame)}", flush=True)
        x = np.asarray(vectors, dtype=np.float32).reshape(-1, 2 * config.n_mfcc)
        print(f"{name}: requested={len(frame)}, successfully processed={len(vectors)}, "
              f"failed={len(frame)-len(vectors)}, shape={x.shape}", flush=True)
        data[f"X_{name}"] = x
        data[f"y_{name}"] = frame.label.map(LABELS).to_numpy(dtype=np.int64)
        data[f"{name}_file_ids"] = frame.file_id.to_numpy(dtype=str)
    (output_dir / "extraction_failures.json").write_text(json.dumps(failures, indent=2))
    if failures:
        raise RuntimeError(f"{len(failures)} audio files failed; see extraction_failures.json. "
                           "Training stopped. No partial cache saved; fix paths/files and rerun.")
    validate_features(data, splits, config)
    temporary = cache.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **data, metadata=json.dumps(metadata, sort_keys=True))
    temporary.replace(cache)
    print(f"Saved feature cache: {cache}")
    return data
