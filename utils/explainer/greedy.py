from typing import Optional, List
import numpy as np
from dp.utils.explainer.base import TokenExplainer
from dp.utils.tri_detector import TRIDetector

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
        self.tri_detector = TRIDetector(model_name=model_name, device=device, use_chunking=use_chunking)
    
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
    
    def explain(self, text: str, tokens: Optional[List[str]] = None, target_label: Optional[int] = None) -> np.ndarray:
        if tokens is None:
            raise ValueError("GreedyExplainer requires explicit tokens")
        self._load_pipeline()
        baseline_result = self.pipeline([text], batch_size=1)[0]
        if target_label is None:
            target_label = baseline_result[0]["label"]
        baseline_prob = 0.0
        for pred in baseline_result:
            if pred["label"] == target_label:
                baseline_prob = pred["score"]
                break
        masked_texts = [text.replace(token, self.mask_token, 1) for token in tokens]
        masked_results = self.pipeline(masked_texts, batch_size=self.batch_size)
        scores = np.zeros(len(tokens))
        for i, result in enumerate(masked_results):
            masked_prob = 0.0
            for pred in result:
                if pred["label"] == target_label:
                    masked_prob = pred["score"]
                    break
            scores[i] = baseline_prob - masked_prob
        return scores
