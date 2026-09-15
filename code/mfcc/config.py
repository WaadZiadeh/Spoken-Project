"""Shared, recorded parameters for Waad's independent experiment."""
from dataclasses import asdict, dataclass

LABELS = {"bonafide": 0, "spoof": 1}
EXPECTED = {
    "train": {"bonafide": 3874, "spoof": 3875},
    "val": {"bonafide": 830, "spoof": 830},
    "test": {"bonafide": 831, "spoof": 830},
}
SPLIT_FILES = {"train": "train.csv", "val": "validation.csv", "test": "test.csv"}
RANDOM_STATE = 42

@dataclass(frozen=True)
class MFCCConfig:
    sample_rate: int = 16000
    n_mfcc: int = 20
    n_fft: int = 512
    hop_length: int = 256
    n_mels: int = 40
    window: str = "hann"
    center: bool = True
    pad_mode: str = "constant"
    power: float = 2.0
    dct_type: int = 2
    norm: str = "ortho"
    lifter: int = 0
    res_type: str = "soxr_hq"
    # Match the peak normalization used in the existing LFCC notebook.
    peak_normalize: bool = True

    def to_dict(self):
        return asdict(self)
