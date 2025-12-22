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

        starting_already_applied = context.get("starting_already_applied")
        if starting_already_applied is not None and not isinstance(starting_already_applied, bool):
            raise ValueError("starting_already_applied must be a bool")

        starting_edit_source = context.get("starting_edit_source")
        if starting_edit_source is not None and not isinstance(starting_edit_source, str):
            starting_edit_source = str(starting_edit_source)
        if isinstance(starting_edit_source, str) and starting_edit_source == "":
            starting_edit_source = None
        applied: List[int] = []
        prev_source = ledger.active_edit_source
        if starting_edit_source is not None and starting_already_applied is not True:
            ledger.set_active_edit_source(starting_edit_source)
        for idx in starting_indices:
            if not isinstance(idx, int):
                raise ValueError("starting_indices must be a list of ints")
            if idx < 0 or idx >= n_offsets:
                raise IndexError(f"starting index {idx} is out of bounds")
            if idx in processed:
                continue
            if starting_already_applied is not True:
                apply_fn(idx, ledger)
            processed.add(idx)
            applied.append(idx)
        if starting_edit_source is not None and starting_already_applied is not True:
            ledger.set_active_edit_source(prev_source)
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

        ledger_value = context.get("ledger")
        if ledger_value is None:
            ledger = TokenLedger(text, offsets)
        else:
            if not isinstance(ledger_value, TokenLedger):
                raise ValueError("ledger must be a TokenLedger")
            ledger = ledger_value

        processed: set[int] = set()
        context_without_ledger = {k: v for k, v in context.items() if k != "ledger"}
        seeded_indices = self._apply_starting_indices(len(offsets), ledger, processed, apply_fn, **context_without_ledger)
        seeded_pending = list(seeded_indices)
        ordered = self.order_thresholds(self._thresholds)

        starting_annotations_name = context.get("starting_annotations_name")
        if starting_annotations_name is not None and not isinstance(starting_annotations_name, str):
            starting_annotations_name = str(starting_annotations_name)
        starting_meta: Dict[str, Any] = {
            "starting_annotations_name": starting_annotations_name,
            "starting_applied_count": len(seeded_indices),
        }

        for threshold in ordered:
            indices = self.select_indices(text, offsets, threshold, processed, ledger=ledger, **context_without_ledger)
            sorted_indices = self._sort_by_risk(indices, len(offsets))
            new_indices: List[int] = []
            for idx in sorted_indices:
                if idx in processed:
                    continue
                apply_fn(idx, ledger)
                processed.add(idx)
                new_indices.append(idx)

            step_indices: List[int] = []
            if seeded_pending:
                step_indices.extend(seeded_pending)
                seeded_pending.clear()
            if new_indices:
                step_indices.extend(new_indices)

            if step_indices:
                metadata: Dict[str, Any] = {"processed_count": len(processed), **starting_meta}
                yield AnonymizationStep(
                    threshold_type=self._threshold_name,
                    threshold=threshold,
                    text=ledger.render_offsets(text),
                    ledger=ledger,
                    new_indices=step_indices,
                    metadata=metadata,
                )
