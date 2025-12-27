from typing import Optional, Sequence, Tuple, Dict, List
import numpy as np
from dp.utils.explainer.base import TokenExplainer
from dp.tri.with_bk import TRIDetectorWithBK

class ShapExplainer(TokenExplainer):
    def __init__(self, model_name: str = None, device: str = "auto", use_chunking: bool = False, **kwargs):
        super().__init__(**kwargs)
        if model_name is None:
            raise ValueError("ShapExplainer requires model_name")
        self.model_name = model_name
        self.device = self._resolve_device(device)
        self.pipeline = None
        self.shap_explainer = None
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
            from transformers import pipeline
            import shap
            self.pipeline = pipeline("text-classification", model=self.model_name, tokenizer=self.model_name, device=self.device if self.device != "cpu" else -1, top_k=None, max_length=512, truncation=True)
            self.shap_explainer = shap.Explainer(self.pipeline, silent=True)
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
        normalized_offsets = self._normalize_offsets(text, offsets)
        if not normalized_offsets:
            return np.zeros(0, dtype=float)
        self._load_pipeline()
        
        label_name = None
        if target_label is not None:
            label_name = str(target_label)
        else:
            raise ValueError("target_label must be provided for ShapExplainer")
            predictions = self.pipeline(text)
            if isinstance(predictions, list) and predictions:
                prediction = predictions[0]
                if isinstance(prediction, dict):
                    label_name = prediction.get("label")
                elif isinstance(prediction, list) and prediction:
                    label_name = prediction[0].get("label")

        label_int: Optional[int] = None
        self._ensure_tri_mapping()
        if label_int is None and label_name in self.label_to_id:
            label_int = self.label_to_id[label_name]
        else:
            try:
                label_int = int(label_name.split("_")[-1])
            except (ValueError, AttributeError):
                pass
        if label_int is None:
            raise ValueError(f"target label '{label_name}' cannot be mapped to an output index")
        term_spans = normalized_offsets
        if self.shap_explainer is None:
            raise ValueError("ShapExplainer pipeline is not loaded properly")
        shap_values = self.shap_explainer([text], batch_size=1)
        subword_weights = shap_values.values[0, :, label_int]
        shap_tokens = shap_values.data[0]
        
        term_weights = np.zeros(len(term_spans), dtype=float)
        
        for term_idx, (term_start, term_end) in enumerate(term_spans):
            overlapping_weights = []
            
            current_pos = 0
            for subword_idx, subword in enumerate(shap_tokens):
                subword_start = text.find(subword, current_pos)
                if subword_start == -1:
                    continue
                subword_end = subword_start + len(subword)
                
                if not (subword_end <= term_start or subword_start >= term_end):
                    overlapping_weights.append(subword_weights[subword_idx])
                current_pos = subword_end
            
            term_weights[term_idx] = sum(overlapping_weights) if overlapping_weights else 0.0
        
        return term_weights

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
