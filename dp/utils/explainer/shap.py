from typing import Optional, Sequence, Tuple, Dict, List, Any
import numpy as np
from transformers import pipeline
import shap
from dp.utils.explainer.base import TokenExplainer
from dp.tri.with_bk import TRIDetectorWithBK
    
class _SpanTokenizer:
    def __init__(self, text: str, spans: Sequence[Tuple[int, int]]):
        self._text = text
        self._spans = list(spans)

    def encode(self, text: str, **_: Any) -> List[int]:
        if text == "":
            return []
        return list(range(len(self._fallback_offsets(text))))

    def __call__(self, text: str, **kwargs: Any) -> Dict[str, Any]:
        return_offsets_mapping = bool(kwargs.get("return_offsets_mapping", True))
        spans = self._spans if text is self._text or text == self._text else self._fallback_offsets(text)
        input_ids = list(range(len(spans)))
        out: Dict[str, Any] = {"input_ids": input_ids}
        if return_offsets_mapping:
            out["offset_mapping"] = spans
        return out

    def convert_ids_to_tokens(self, ids: Sequence[int]) -> List[str]:
        return [self._text[a:b] if 0 <= i < len(self._spans) else "" for i, (a, b) in zip(ids, self._spans)]

    def decode(self, ids: Sequence[int], **_: Any) -> str:
        toks = self.convert_ids_to_tokens(ids)
        return " ".join([t for t in toks if t])

    def _fallback_offsets(self, text: str) -> List[Tuple[int, int]]:
        spans: List[Tuple[int, int]] = []
        n = len(text)
        i = 0
        while i < n:
            while i < n and text[i].isspace():
                i += 1
            if i >= n:
                break
            j = i + 1
            while j < n and not text[j].isspace():
                j += 1
            spans.append((i, j))
            i = j
        return spans

class ShapExplainer(TokenExplainer):
    def __init__(self, model_name: str = None, device: str = "auto", use_chunking: bool = False, **kwargs):
        super().__init__(**kwargs)
        if model_name is None:
            raise ValueError("ShapExplainer requires model_name")
        self.model_name = model_name
        self.device = self._resolve_device(device)
        self.pipeline = None
        self.tri_detector = TRIDetectorWithBK(model_name=model_name, device=device, use_chunking=use_chunking)
        self._tri_mapping_attempted = False
        self.id_to_label: Dict[int, str] = {}
        self.label_to_id: Dict[str, int] = {}

    def _resolve_device(self, device: str) -> str:
        if device == "auto":
            import torch
            if torch.cuda.is_available():
                return "cuda"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
            return "cpu"
        return device

    def _load_pipeline(self):
        if self.pipeline is None:
            self.pipeline = pipeline(
                "text-classification",
                model=self.model_name,
                tokenizer=self.model_name,
                device=self.device if self.device != "cpu" else -1,
                top_k=None,
                max_length=512,
                truncation=True,
            )
            config = getattr(self.pipeline.model, "config", None)
            if config is not None and hasattr(config, "id2label"):
                self.id_to_label = dict(config.id2label)
                self.label_to_id = {label: idx for idx, label in self.id_to_label.items()}

    def _ensure_tri_mapping(self) -> None:
        if self._tri_mapping_attempted:
            return
        self._tri_mapping_attempted = True
        self.tri_detector.load(self.model_name)
        self.label_to_id.update(self.tri_detector.name_to_label)
        for name, idx in self.tri_detector.name_to_label.items():
            self.id_to_label[idx] = name

    def explain(self, text: str, offsets: Sequence[Tuple[int, int]], target_label: Optional[str] = None) -> np.ndarray:
        spans = self._normalize_offsets(text, offsets)
        if not spans:
            return np.zeros(0, dtype=float)
        if target_label is None:
            raise ValueError("target_label must be provided for ShapExplainer")

        self._load_pipeline()
        self._ensure_tri_mapping()

        label_name = str(target_label)
        label_int: Optional[int] = self.label_to_id.get(label_name)
        if label_int is None:
            try:
                label_int = int(label_name.split("_")[-1])
            except (ValueError, AttributeError):
                label_int = None
        if label_int is None:
            raise ValueError(f"target label '{label_name}' cannot be mapped to an output index")

        tokenizer = _SpanTokenizer(text=text, spans=spans)
        masker = shap.maskers.Text(tokenizer=tokenizer, collapse_mask_token=True)
        explainer = shap.Explainer(self.pipeline, masker, silent=True)

        shap_values = explainer([text], batch_size=1)
        values = shap_values.values[0, :, label_int]

        if values.shape[0] != len(spans):
            raise ValueError(f"SHAP produced {values.shape[0]} tokens, expected {len(spans)}")

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