from typing import Sequence, Tuple
import numpy as np
from dp.utils.explainer.base import TokenExplainer

class UniformExplainer(TokenExplainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def explain(self, text: str, offsets: Sequence[Tuple[int, int]]) -> np.ndarray:
        if offsets is None:
            raise ValueError("UniformExplainer requires offsets")
        return np.ones(len(offsets), dtype=float)
