from typing import Optional, List
import numpy as np

from dp.utils.explainer.base import TokenExplainer

class UniformExplainer(TokenExplainer):
    """
    Explainer that assigns uniform (equal) importance scores to all tokens.
    
    This is the simplest explainability strategy, treating all tokens as
    equally important for differential privacy budget allocation.
    """
    
    def __init__(self, **kwargs):
        """
        Initialize UniformExplainer.
        
        Args:
            **kwargs: Additional configuration parameters (unused for uniform)
        """
        super().__init__(**kwargs)

    def explain(self, text: str, tokens: Optional[List[str]] = None) -> np.ndarray:
        """
        Return uniform scores for all tokens.
        
        Args:
            text: Input text to analyze
            tokens: Optional list of tokens (if not provided, will tokenize)
            
        Returns:
            Array of uniform scores (all ones)
        """
        if tokens is None:
            # Simple word tokenization if no tokens provided
            tokens = text.split()
        
        # Return uniform scores (all 1.0)
        return np.ones(len(tokens), dtype=float)
