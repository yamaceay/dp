from typing import Optional, List
import numpy as np
from dp.utils.explainer.base import TokenExplainer
from dp.utils.tri_detector import TRIDetector

class ShapExplainer(TokenExplainer):
    def __init__(self, model_name: str = None, device: str = "auto", use_chunking: bool = False, **kwargs):
        super().__init__(**kwargs)
        if model_name is None:
            raise ValueError("ShapExplainer requires model_name")
        self.model_name = model_name
        self.device = self._resolve_device(device)
        self.pipeline = None
        self.shap_explainer = None
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
            import shap
            self.pipeline = pipeline("text-classification", model=self.model_name, tokenizer=self.model_name, device=self.device if self.device != "cpu" else -1, top_k=None, max_length=512, truncation=True)
            self.shap_explainer = shap.Explainer(self.pipeline, silent=True)
    
    def explain(self, text: str, tokens: Optional[List[str]] = None, target_label: Optional[int] = None) -> np.ndarray:
        self._load_pipeline()
        shap_values = self.shap_explainer([text], batch_size=1)
        
        if target_label is None:
            result = self.pipeline([text], batch_size=1)[0]
            label_str = result[0]["label"]
            if label_str.startswith("LABEL_"):
                target_label = int(label_str.split("_")[1])
            else:
                target_label = 0
        
        token_scores = shap_values.values[0, :, target_label]
        if tokens is not None:
            return np.ones(len(tokens)) * np.mean(np.abs(token_scores))
        return np.abs(token_scores)
