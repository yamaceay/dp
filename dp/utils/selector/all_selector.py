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
        processed: set[int] = set()
        seeded = self._apply_starting_indices(len(offsets), ledger, processed, apply_fn, **context)

        starting_annotations_name = context.get("starting_annotations_name")
        if starting_annotations_name is not None and not isinstance(starting_annotations_name, str):
            starting_annotations_name = str(starting_annotations_name)
        step_metadata: dict = {
            "processed_count": len(processed),
            "starting_annotations_name": starting_annotations_name,
            "starting_applied_count": len(seeded),
        }

        indices = self.select_indices(text, offsets, None, processed, **context)
        sorted_indices = self._sort_by_risk(indices, len(offsets))
        remaining: List[int] = []
        for idx in sorted_indices:
            if idx in processed:
                continue
            apply_fn(idx, ledger)
            processed.add(idx)
            remaining.append(idx)

        applied = seeded + remaining

        yield AnonymizationStep(
            threshold_type=None,
            threshold=None,
            text=ledger.render_offsets(text),
            ledger=ledger,
            new_indices=applied,
            metadata=step_metadata,
        )


AllSelector = AllUnit