from typing import Optional, Sequence, Tuple, Dict, List, Any, Union
from enum import Enum
import numpy as np
from dp.utils.explainer.base import TokenExplainer
from dp.utils.device import resolve_device
from dp.tri import TRIDetector

class _SpanTokenizer:
    def __init__(self, text: str, spans: Sequence[Tuple[int, int]]):
        if text is None:
            raise ValueError("text cannot be None")
        if spans is None:
            raise ValueError("spans cannot be None")
        self._text = text
        self._spans = list(spans)
        self._tokens = [text[s:e] for s, e in spans]
        self._sorted_order = sorted(range(len(spans)), key=lambda i: spans[i][0])
        self._gaps = self._compute_gaps(text, spans)

    def _compute_gaps(self, text: str, spans: Sequence[Tuple[int, int]]) -> List[str]:
        if not spans:
            return [text]
        gaps = []
        prev_end = 0
        for idx in self._sorted_order:
            start, end = spans[idx]
            gaps.append(text[prev_end:start])
            prev_end = end
        gaps.append(text[prev_end:])
        return gaps

    def __call__(self, s: str, return_offsets_mapping: bool = True, **_: Any) -> Dict[str, Any]:
        if s is None:
            raise ValueError("input string cannot be None")
        if s == "":
            out: Dict[str, Any] = {"input_ids": []}
            if return_offsets_mapping:
                out["offset_mapping"] = []
            return out
        if s == self._text:
            tokens = self._tokens
        else:
            tokens = [s[start:end] if end <= len(s) else "" for start, end in self._spans]
        out = {"input_ids": tokens}
        if return_offsets_mapping:
            out["offset_mapping"] = self._spans
        return out

    def convert_ids_to_tokens(self, ids: Sequence[int]) -> List[str]:
        if ids is None:
            raise ValueError("ids cannot be None")
        return [self._tokens[i] if 0 <= i < len(self._tokens) else "" for i in ids]

    def decode(self, ids: Sequence[int], **_: Any) -> str:
        if ids is None:
            raise ValueError("ids cannot be None")
        if not self._spans:
            return self._gaps[0] if self._gaps else ""
        id_set = set(ids)
        parts = [self._gaps[0]]
        for pos, original_idx in enumerate(self._sorted_order):
            if original_idx in id_set:
                parts.append(self._tokens[original_idx])
            parts.append(self._gaps[pos + 1])
        return "".join(parts)


class ShapType(Enum):
    DEFAULT = "shap"
    PERMUTATION = "shap_permutation"

class ShapExplainer(TokenExplainer):
    def __init__(self, model_name: str = None, device: Optional[Union[str, int]] = None, use_chunking: bool = False, explainer_type: Optional[ShapType] = None, **kwargs):
        super().__init__(**kwargs)
        if model_name is None:
            raise ValueError("ShapExplainer requires model_name")
        self.model_name = model_name
        self.device = resolve_device(device)
        self.pipeline = None
        self.tri_detector = TRIDetector(model_name=model_name, device=self.device, use_chunking=use_chunking)
        self._tri_mapping_attempted = False
        self.id_to_label: Dict[int, str] = {}
        self.label_to_id: Dict[str, int] = {}
        self.explainer_type: ShapType = explainer_type or ShapType.DEFAULT
    
    def _load_pipeline(self):
        if self.pipeline is None:
            from transformers import pipeline
            self.pipeline = pipeline("text-classification", model=self.model_name, tokenizer=self.model_name, device=self.device if self.device != "cpu" else -1, top_k=None, max_length=512, truncation=True)
            config = getattr(self.pipeline.model, "config", None)
            if config is not None and hasattr(config, "id2label"):
                self.id_to_label = dict(config.id2label)
                self.label_to_id = {label: idx for idx, label in self.id_to_label.items()}

    def _ensure_tri_mapping(self) -> None:
        if self._tri_mapping_attempted:
            return
        self._tri_mapping_attempted = True
        try:
            self.tri_detector.load(self.model_name)
        except Exception:
            return
        self.label_to_id.update(self.tri_detector.name_to_label)
        for name, idx in self.tri_detector.name_to_label.items():
            self.id_to_label[idx] = name

    def explain(self, text: str, offsets: Sequence[Tuple[int, int]], target_label: Optional[str] = None) -> np.ndarray:
        import shap
        normalized_offsets = self._normalize_offsets(text, offsets)
        if not normalized_offsets:
            return np.zeros(0, dtype=float)
        self._load_pipeline()
        
        if target_label is None:
            raise ValueError("target_label must be provided for ShapExplainer")
        label_name = str(target_label)
        
        label_int: Optional[int] = None
        self._ensure_tri_mapping()
        if label_name in self.label_to_id:
            label_int = self.label_to_id[label_name]
        if label_int is None:
            try:
                label_int = int(label_name.split("_")[-1])
            except (ValueError, AttributeError):
                pass
        if label_int is None:
            raise ValueError(f"target label '{label_name}' cannot be mapped to an output index")

        tokenizer = _SpanTokenizer(text, normalized_offsets)
        masker = shap.maskers.Text(tokenizer=tokenizer, collapse_mask_token=True)
        explainer = None
        explainer_args = dict(silent=True)
        match self.explainer_type:
            case ShapType.PERMUTATION:
                num_features = max(2 * len(normalized_offsets) + 1, 500)
                explainer = shap.PermutationExplainer(self.pipeline, masker, max_evals=num_features, **explainer_args)
            case _:
                explainer = shap.Explainer(self.pipeline, masker, **explainer_args)
        shap_values = explainer([text], batch_size=1)

        values = shap_values.values[0, :, label_int]
        if values.shape[0] != len(normalized_offsets):
            raise ValueError(f"SHAP produced {values.shape[0]} values, expected {len(normalized_offsets)}")

        return values.astype(float)

    def _normalize_offsets(self, text: str, offsets: Sequence[Tuple[int, int]]) -> List[Tuple[int, int]]:
        if offsets is None:
            raise ValueError("ShapExplainer requires offsets")
        length = len(text)
        normalized: List[Tuple[int, int]] = []
        for start, end in offsets:
            start_int = int(start)
            end_int = int(end)
            if start_int < 0 or end_int < start_int or end_int > length:
                raise ValueError(f"Invalid offset span ({start}, {end}) for text of length {length}")
            normalized.append((start_int, end_int))
        return normalized
