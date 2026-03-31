from typing import Any, List, Optional
import numpy as np

def combine_privacy_utility_scores(
    privacy_scores: np.ndarray,
    utility_scores: np.ndarray,
    alpha: float,
) -> np.ndarray:
    def _minmax(arr: np.ndarray) -> np.ndarray:
        lo, hi = float(arr.min()), float(arr.max())
        if hi == lo:
            return np.zeros_like(arr)
        return (arr - lo) / (hi - lo)
    return _minmax(privacy_scores) - alpha * _minmax(utility_scores)

def top_predicted_label(explainer: Any, text: str) -> str:
    entries: List[Any] = explainer.predict_entries([text], batch_size=1)[0]
    top = max(entries, key=lambda e: float(e["score"]))
    return str(top["label"])

def _scores_to_probs(scores: np.ndarray, temperature: Optional[float] = None) -> np.ndarray:
    scores = np.asarray(scores, dtype=float)
    scores = (scores - np.min(scores)) / (np.max(scores) - np.min(scores) + 1e-10)
    if temperature is not None and temperature >= 0.0:
        scores = np.exp(scores / temperature)
    scores = scores / np.sum(scores)
    return scores

def _scores_to_inverse_probs(scores: np.ndarray, temperature: Optional[float] = None) -> np.ndarray:
    scores = np.asarray(scores, dtype=float)
    scores = (np.max(scores) - scores) / (np.max(scores) - np.min(scores) + 1e-10)
    if temperature is not None and temperature >= 0.0:
        scores = np.exp(scores / temperature)
    scores = scores / np.sum(scores)
    return scores