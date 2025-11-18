from typing import Sequence, Tuple
import numpy as np

class TokenExplainer:
    def __init__(self, *args, **kwargs):
        pass

    def explain(self, text: str, offsets: Sequence[Tuple[int, int]]) -> np.ndarray:
        raise NotImplementedError("TokenExplainer is a stub.")
