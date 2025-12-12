from typing import Any, Iterator, List, Tuple

from dp.utils.selector.base import AnonymizerUnit, AnonymizationStep, ApplyFn
from dp.utils.token_ledger import TokenLedger


class AllUnit(AnonymizerUnit):
    def __init__(self, temperature: float = 1.0, **kwargs: Any) -> None:
        super().__init__(temperature=temperature)

    def order_thresholds(self, thresholds: List[Any]) -> List[Any]:
        return thresholds

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
        return [i for i in range(len(offsets)) if i not in already_processed]

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
        indices = self.select_indices(text, offsets, None, set(), **context)
        sorted_indices = self._sort_by_risk(indices, len(offsets))
        
        for idx in sorted_indices:
            apply_fn(idx, ledger)

        yield AnonymizationStep(
            threshold=None,
            text=ledger.render_offsets(text),
            ledger=ledger,
            new_indices=sorted_indices,
            metadata={"processed_count": len(sorted_indices)},
        )


AllSelector = AllUnit