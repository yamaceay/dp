from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

import numpy as np

from dp.loaders.base import TextAnnotation
from dp.utils.token_ledger import TokenLedger


@dataclass
class AnonymizationStep:
    threshold: Any
    text: str
    ledger: TokenLedger
    new_indices: List[int]
    metadata: Dict[str, Any] = field(default_factory=dict)


ApplyFn = Callable[[int, TokenLedger], None]


class AnonymizerUnit(ABC):
    def __init__(self, temperature: float = 1.0) -> None:
        self._thresholds: List[Any] = []
        self._risk_scores: Optional[np.ndarray] = None
        self._temperature = float(temperature) if temperature > 0 else 1.0

    def set_thresholds(self, thresholds: List[Any]) -> None:
        self._thresholds = list(thresholds)

    def set_risk_scores(self, scores: np.ndarray) -> None:
        self._risk_scores = scores

    def _scores_to_probs(self, scores: np.ndarray) -> np.ndarray:
        if scores.size == 0:
            return scores
        scaled = scores / self._temperature
        scaled = scaled - np.max(scaled)
        exps = np.exp(scaled)
        total = np.sum(exps)
        if total <= 0:
            return np.ones(len(scores)) / len(scores)
        return exps / total

    def _sort_by_risk(self, indices: List[int], n_offsets: int) -> List[int]:
        if not indices:
            return indices
        if self._risk_scores is None or len(self._risk_scores) != n_offsets:
            return indices
        return sorted(indices, key=lambda i: float(self._risk_scores[i]), reverse=True)

    @abstractmethod
    def order_thresholds(self, thresholds: List[Any]) -> List[Any]:
        pass

    @abstractmethod
    def select_indices(
        self,
        text: str,
        offsets: List[Tuple[int, int]],
        threshold: Any,
        already_processed: set[int],
        **context: Any,
    ) -> List[int]:
        pass

    def anonymize(
        self,
        text: str,
        offsets: List[Tuple[int, int]],
        apply_fn: ApplyFn,
        **context: Any,
    ) -> Iterator[AnonymizationStep]:
        if not self._thresholds:
            return

        ledger = TokenLedger(text, offsets)
        processed: set[int] = set()
        ordered = self.order_thresholds(self._thresholds)

        for threshold in ordered:
            indices = self.select_indices(text, offsets, threshold, processed, ledger=ledger, **context)
            sorted_indices = self._sort_by_risk(indices, len(offsets))
            new_indices: List[int] = []
            for idx in sorted_indices:
                if idx in processed:
                    continue
                apply_fn(idx, ledger)
                processed.add(idx)
                new_indices.append(idx)

            if new_indices:
                yield AnonymizationStep(
                    threshold=threshold,
                    text=ledger.render_offsets(text),
                    ledger=ledger,
                    new_indices=new_indices,
                    metadata={"processed_count": len(processed)},
                )
