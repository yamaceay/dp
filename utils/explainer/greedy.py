from typing import Optional, Sequence, Tuple, List
import numpy as np
from dp.utils.explainer.base import TokenExplainer
from dp.utils.memory import clear_memory
from dp.tri.with_deid import TRIDetectorWithDeid

class GreedyExplainer(TokenExplainer):
    def __init__(self, model_name: str = None, mask_token: str = "[MASK]", batch_size: int = 128, device: str = "auto", use_chunking: bool = False, **kwargs):
        super().__init__(**kwargs)
        if model_name is None:
            raise ValueError("GreedyExplainer requires model_name")
        self.model_name = model_name
        self.mask_token = mask_token
        self.batch_size = batch_size
        self.device = self._resolve_device(device)
        self.pipeline = None
        self.tri_detector = TRIDetectorWithDeid(model_name=model_name, device=device, use_chunking=use_chunking)
    
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
            self.pipeline = pipeline("text-classification", model=self.model_name, tokenizer=self.model_name, device=self.device if self.device != "cpu" else -1, top_k=None, max_length=512, truncation=True)
    
    def explain(self, text: str, offsets: Sequence[Tuple[int, int]], target_label: Optional[str] = None) -> np.ndarray:
        normalized_offsets = self._normalize_offsets(text, offsets)
        if not normalized_offsets:
            return np.zeros(0, dtype=float)
        self._load_pipeline()
        baseline_result = self.pipeline([text], batch_size=1)[0]
        if target_label is None:
            target_label = baseline_result[0]["label"]
        baseline_prob = 0.0
        for pred in baseline_result:
            if pred["label"] == target_label:
                baseline_prob = pred["score"]
                break
        masked_texts = [text[:start] + self.mask_token + text[end:] for start, end in normalized_offsets]
        masked_results = self.pipeline(masked_texts, batch_size=self.batch_size)
        clear_memory()
        scores = np.zeros(len(normalized_offsets), dtype=float)
        for i, result in enumerate(masked_results):
            masked_prob = 0.0
            for pred in result:
                if pred["label"] == target_label:
                    masked_prob = pred["score"]
                    break
            scores[i] = baseline_prob - masked_prob
        return scores

    def _normalize_offsets(self, text: str, offsets: Sequence[Tuple[int, int]]) -> List[Tuple[int, int]]:
        if offsets is None:
            raise ValueError("GreedyExplainer requires offsets")
        length = len(text)
        normalized: List[Tuple[int, int]] = []
        for start, end in offsets:
            start_int = int(start)
            end_int = int(end)
            if start_int < 0 or end_int < start_int or end_int > length:
                raise ValueError(f"Invalid offset span ({start}, {end}) for text of length {length}")
            normalized.append((start_int, end_int))
        return normalized
