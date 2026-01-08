import numpy as np

def compute_coral_bias_init(labels: np.ndarray, n_classes: int) -> np.ndarray:
    cumulative_probs = np.array([
        (labels > t).mean() for t in range(n_classes - 1)
    ])
    eps = 1e-7
    cumulative_probs = np.clip(cumulative_probs, eps, 1 - eps)
    return np.log(cumulative_probs / (1 - cumulative_probs))
