from typing import Dict

from dp.tri.base import TRIDetector
from dp.tri.with_bk import TRIDetectorWithBK
# from dp.tri.with_deid import TRIDetectorWithDeid
# from dp.tri.with_split import TRIDetectorWithSplit

TRI_DETECTOR_REGISTRY: Dict[str, type[TRIDetector]] = {
    "bk": TRIDetectorWithBK,
    # "deid": TRIDetectorWithDeid,
    # "split": TRIDetectorWithSplit,
}

def get_tri_detector(name: str, *args, **kwargs) -> TRIDetector:
    key = (name or "").lower()
    if key not in TRI_DETECTOR_REGISTRY:
        raise ValueError(f"Unknown TRI detector: {name}")
    return TRI_DETECTOR_REGISTRY[key](*args, **kwargs)

__all__ = [
    "TRIDetectorWithBK",
    # "TRIDetectorWithDeid",
    # "TRIDetectorWithSplit",
    "get_tri_detector",
    "TRI_DETECTOR_REGISTRY",
]
