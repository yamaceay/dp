from typing import Sequence, Tuple, Optional, Dict, Any
from pathlib import Path
import json
import numpy as np

class TokenExplainer:
    def __init__(self, *args, **kwargs):
        pass

    def explain(self, text: str, offsets: Sequence[Tuple[int, int]]) -> np.ndarray:
        raise NotImplementedError("TokenExplainer is a stub.")

def load_tri_label_mapping(explainer: TokenExplainer, label_mapping: Optional[Dict[str, int]] = None, label_mapping_source: Optional[str] = None) -> Dict[str, int]:
    if explainer is None:
        raise ValueError("It is required to load TRI label mapping")
    model_name = getattr(explainer, "model_name", None)
    if not model_name:
        raise ValueError("No valid explainer.model_name to load TRI label mapping")
    source = str(model_name)
    if label_mapping is not None and label_mapping_source == source:
        return label_mapping, source

    mapping_path = Path(source) / "label_mapping.json"
    if not mapping_path.exists():
        raise ValueError(f"TRI label mapping not found at {mapping_path}")
    with mapping_path.open("r", encoding="utf-8") as handle:
        mapping = json.load(handle)
    if not isinstance(mapping, dict) or not mapping:
        raise ValueError(f"Invalid TRI label mapping at {mapping_path}")

    normalized: Dict[str, int] = {}
    for name, value in mapping.items():
        if not isinstance(name, str):
            continue
        try:
            normalized[name] = int(value)
        except (TypeError, ValueError):
            continue
    if not normalized:
        raise ValueError(f"TRI label mapping at {mapping_path} has no usable entries")

    return normalized, source