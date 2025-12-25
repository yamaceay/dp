from typing import Any, Iterator, List, Tuple

from dp.utils.selector.base import AnonymizerUnit, AnonymizationStep, ApplyFn
from dp.utils.token_ledger import TokenLedger


class AllUnit(AnonymizerUnit):
    SELECTOR_NAME = "all"

    def __init__(self, temperature: float = 1.0, **kwargs: Any) -> None:
        super().__init__(temperature=temperature, selector_name=self.SELECTOR_NAME)

    def order_thresholds(self, thresholds: List[Any]) -> List[Any]:
        return thresholds

    def select_indices(
        self,
        text: str,
        offsets: List[Tuple[int, int]],
        threshold: Any,
        **context: Any,
    ) -> List[int]:
        if not text or not text.strip():
            return []
        return offsets

    def anonymize(
        self,
        text: str,
        offsets: List[Tuple[int, int]],
        apply_fn: ApplyFn,
        **context: Any,
    ) -> Iterator[AnonymizationStep]:
        if not text or not text.strip():
            return

        ledger = TokenLedger(text, offsets)
        
        processed: set[int] = set()

        step_metadata: dict = {
            "selector": self.SELECTOR_NAME,
            "processed_count": len(processed),
        }

        indices = self.select_indices(text, offsets, None, processed, **context)
        sorted_indices = self._sort_by_risk(indices, len(offsets))
        applied: List[int] = []
        for idx in sorted_indices:
            if idx in processed:
                continue
            apply_fn(idx, ledger)
            processed.add(idx)
            applied.append(idx)

        yield AnonymizationStep(
            threshold_type=None,
            threshold=None,
            text=ledger.render_offsets(text),
            ledger=ledger,
            new_indices=applied,
            metadata=step_metadata,
        )