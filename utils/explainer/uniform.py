from typing import Optional, List
import numpy as np
from dp.utils.explainer.base import TokenExplainer

class UniformExplainer(TokenExplainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def explain(self, text: str, tokens: Optional[List[str]] = None) -> np.ndarray:
        if tokens is None:
            tokens = text.split()
        return np.ones(len(tokens), dtype=float)
