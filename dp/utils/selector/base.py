from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

import numpy as np

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
    def __init__(self, temperature: float = 1.0, sort_by_risk: bool = True, selector_name: Optional[str] = None) -> None:
        self._thresholds: List[Any] = []
        self._threshold_name: Optional[str] = None
        self._risk_scores: Optional[np.ndarray] = None
        self._temperature = float(temperature) if temperature > 0 else 1.0
        self._sort_by_risk_enabled = bool(sort_by_risk)
        self._selector_name = selector_name

    def set_thresholds(self, thresholds: List[Any], name: str) -> None:
        self._thresholds = list(thresholds)
        if not name or not str(name).strip():
            raise ValueError("threshold name must be a non-empty string")
        self._threshold_name = str(name)

    def set_risk_scores(self, scores: np.ndarray) -> None:
        self._risk_scores = scores

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
        **context: Any,
    ) -> List[int]:
        pass

    def anonymize(
        self,
        text: str,
        offsets: List[Tuple[int, int]],
        apply_fn: ApplyFn,
        ledger: Optional[TokenLedger] = None,
        prior_edits: Optional[List[Dict[str, object]]] = None,
        **context: Any,
    ) -> Iterator[AnonymizationStep]:
        if not self._thresholds:
            return

        if self._threshold_name is None:
            raise ValueError("threshold name must be set before anonymization")

        if ledger is None:
            ledger = TokenLedger(text, offsets)
        
        if prior_edits:
            ledger.apply_prior_edits(prior_edits)

        processed: set[int] = set()
        for idx in range(len(offsets)):
            if ledger.is_modified(idx):
                processed.add(idx)
        
        ordered = self.order_thresholds(self._thresholds)

        for threshold in ordered:
            applied_indices = self.select_indices(text, offsets, threshold, **context)
            new_indices: List[int] = []
            for idx in applied_indices:
                if idx in processed:
                    continue
                apply_fn(idx, ledger)
                processed.add(idx)
                new_indices.append(idx)

            metadata: Dict[str, Any] = {
                "selector": self._selector_name,
                "processed_count": len(processed)
            }
            yield AnonymizationStep(
                threshold_type=self._threshold_name,
                threshold=threshold,
                text=ledger.render_offsets(text),
                ledger=ledger,
                new_indices=new_indices,
                metadata=metadata,
            )
