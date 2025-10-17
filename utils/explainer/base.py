"""
Base stub for explainers. Extend for uniform, shap, greedy, etc.
"""

from typing import Optional, List

class TokenExplainer:
    def __init__(self, *args, **kwargs):
        pass
    def explain(self, text: str, tokens: Optional[List[str]] = None):
        """Stub explain method. Implement in subclass."""
        raise NotImplementedError("TokenExplainer is a stub.")
