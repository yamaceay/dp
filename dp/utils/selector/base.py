from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

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
    def __init__(self) -> None:
        self._thresholds: List[Any] = []

    def set_thresholds(self, thresholds: List[Any]) -> None:
        self._thresholds = list(thresholds)

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
            new_indices: List[int] = []
            for idx in indices:
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
