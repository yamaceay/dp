"""
Explainer utilities for explainability-based scoring.

This module provides explainer classes that compute importance scores for tokens
to guide differential privacy budget allocation:

- UniformExplainer: Assigns equal importance to all tokens
- GreedyExplainer: Measures importance by masking impact (stub - needs implementation)
- ShapExplainer: Uses SHAP values for importance (stub - needs implementation)
"""

from dp.utils.explainer.base import TokenExplainer
from dp.utils.explainer.uniform import UniformExplainer
from dp.utils.explainer.shap import ShapExplainer
from dp.utils.explainer.greedy import GreedyExplainer

__all__ = [
    "TokenExplainer",
    "UniformExplainer",
    "GreedyExplainer",
    "ShapExplainer",
]