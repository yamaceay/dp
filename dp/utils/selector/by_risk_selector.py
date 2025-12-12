import string
from typing import Any, List, Tuple

import numpy as np

from dp.utils.selector.base import AnonymizerUnit


class ByRiskUnit(AnonymizerUnit):
    def __init__(self, temperature: float = 1.0, **kwargs: Any) -> None:
        super().__init__(temperature=temperature)

    def order_thresholds(self, thresholds: List[Any]) -> List[Any]:
        return sorted([float(t) for t in thresholds], reverse=True)

    def select_indices(
        self,
        text: str,
        offsets: List[Tuple[int, int]],
        threshold: Any,
        already_processed: set[int],
        **context: Any,
    ) -> List[int]:
        if not text or not text.strip():
            return []

        if self._risk_scores is None or len(self._risk_scores) != len(offsets):
            return []

        rho = float(threshold)
        removal_limit = max(0.0, min(1.0, 1.0 - rho))
        if removal_limit <= 0:
            return []

        probs = self._scores_to_probs(self._risk_scores)
        pairs = [(idx, probs[idx]) for idx in range(len(offsets)) if idx not in already_processed]
        pairs.sort(key=lambda x: x[1], reverse=True)

        cumulative = sum(probs[idx] for idx in already_processed)
        indices: List[int] = []
        for idx, prob in pairs:
            if cumulative >= removal_limit:
                break
            token = text[offsets[idx][0] : offsets[idx][1]]
            if not token or token in string.punctuation:
                continue
            indices.append(idx)
            cumulative += prob

        return indices


ByRiskSelector = ByRiskUnit
