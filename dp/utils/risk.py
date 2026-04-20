from typing import Optional
import numpy as np

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