from typing import Optional, List
import numpy as np
from dp.utils.explainer.base import TokenExplainer
from dp.utils.tri_detector import TRIDetector
from dp.utils.splitter import TextSplitter

class ShapExplainer(TokenExplainer):
    def __init__(self, model_name: str = None, device: str = "auto", use_chunking: bool = False, **kwargs):
        super().__init__(**kwargs)
        if model_name is None:
            raise ValueError("ShapExplainer requires model_name")
        self.model_name = model_name
        self.device = self._resolve_device(device)
        self.pipeline = None
        self.shap_explainer = None
        self.splitter = TextSplitter()
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
    
    def explain(self, text: str, tokens: Optional[List[str]] = None, target_label: Optional[str] = None) -> np.ndarray:
        self._load_pipeline()
        
        if target_label is None:
            raise ValueError("target_label required")
        
        label_int = int(target_label.split("_")[1]) if "_" in target_label else int(target_label)
        
        if tokens is None:
            raise ValueError("tokens required")
        
        term_spans = self.splitter.tokenize_with_spans(text)
        shap_values = self.shap_explainer([text], batch_size=1)
        subword_weights = shap_values.values[0, :, label_int]
        shap_tokens = shap_values.data[0]
        
        term_weights = np.zeros(len(tokens))
        
        for term_idx, token in enumerate(tokens):
            if term_idx >= len(term_spans):
                break
            
            term_start, term_end, _ = term_spans[term_idx]
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
