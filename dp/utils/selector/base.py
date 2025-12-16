from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

import numpy as np

from dp.loaders.base import TextAnnotation
from dp.utils.token_ledger import TokenLedger


@dataclass
class AnonymizationStep:
    threshold_type: Optional[str]
    threshold: Any
    text: str
    ledger: TokenLedger
    new_indices: List[int]
    metadata: Dict[str, Any] = field(default_factory=dict)


ApplyFn = Callable[[int, TokenLedger], None]


class AnonymizerUnit(ABC):
    def __init__(self, temperature: float = 1.0) -> None:
        self._thresholds: List[Any] = []
        self._threshold_name: Optional[str] = None
        self._risk_scores: Optional[np.ndarray] = None
        self._temperature = float(temperature) if temperature > 0 else 1.0

    def set_thresholds(self, thresholds: List[Any], name: str) -> None:
        self._thresholds = list(thresholds)
        if not name or not str(name).strip():
            raise ValueError("threshold name must be a non-empty string")
        self._threshold_name = str(name)

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

    def _apply_starting_indices(
        self,
        n_offsets: int,
        ledger: TokenLedger,
        processed: set[int],
        apply_fn: ApplyFn,
        **context: Any,
    ) -> List[int]:
        starting_indices = context.get("starting_indices")
        if starting_indices is None:
            return []
        if not isinstance(starting_indices, list):
            raise ValueError("starting_indices must be a list of ints")
        applied: List[int] = []
        for idx in starting_indices:
            if not isinstance(idx, int):
                raise ValueError("starting_indices must be a list of ints")
            if idx < 0 or idx >= n_offsets:
                raise IndexError(f"starting index {idx} is out of bounds")
            if idx in processed:
                continue
            apply_fn(idx, ledger)
            processed.add(idx)
            applied.append(idx)
        return applied

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

        if self._threshold_name is None:
            raise ValueError("threshold name must be set before anonymization")

        ledger = TokenLedger(text, offsets)
        processed: set[int] = set()
        self._apply_starting_indices(len(offsets), ledger, processed, apply_fn, **context)
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
                metadata: Dict[str, Any] = {"processed_count": len(processed)}
                yield AnonymizationStep(
                    threshold_type=self._threshold_name,
                    threshold=threshold,
                    text=ledger.render_offsets(text),
                    ledger=ledger,
                    new_indices=new_indices,
                    metadata=metadata,
                )
