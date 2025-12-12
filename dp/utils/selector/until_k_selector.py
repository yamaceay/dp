from typing import Any, Callable, Iterator, List, Optional, Tuple

import numpy as np

from dp.utils.selector.base import AnonymizerUnit, AnonymizationStep, ApplyFn
from dp.utils.token_ledger import TokenLedger


RankEvaluator = Callable[[str, int], int]


class UntilKUnit(AnonymizerUnit):
    def __init__(
        self,
        rank_evaluator: Optional[RankEvaluator] = None,
        temperature: float = 1.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(temperature=temperature)
        self._rank_evaluator = rank_evaluator
        self._target_label: Optional[int] = None

    def set_rank_evaluator(self, evaluator: RankEvaluator) -> None:
        self._rank_evaluator = evaluator

    def set_target_label(self, label: int) -> None:
        self._target_label = label

    def order_thresholds(self, thresholds: List[Any]) -> List[Any]:
        return sorted([int(t) for t in thresholds])

    def select_indices(
        self,
        text: str,
        offsets: List[Tuple[int, int]],
        threshold: Any,
        already_processed: set[int],
        **context: Any,
    ) -> List[int]:
        candidates = [i for i in range(len(offsets)) if i not in already_processed]
        return self._sort_by_risk(candidates, len(offsets))

    def anonymize(
        self,
        text: str,
        offsets: List[Tuple[int, int]],
        apply_fn: ApplyFn,
        **context: Any,
    ) -> Iterator[AnonymizationStep]:
        if not self._thresholds:
            return

        if self._rank_evaluator is None:
            raise ValueError("UntilKUnit requires rank_evaluator to be set")
        if self._target_label is None:
            raise ValueError("UntilKUnit requires target_label to be set")

        ledger = TokenLedger(text, offsets)
        processed: set[int] = set()
        k_values = self.order_thresholds(self._thresholds)
        
        current_text = ledger.render_offsets(text)
        current_rank = self._rank_evaluator(current_text, self._target_label)

        for target_k in k_values:
            if current_rank >= target_k:
                yield AnonymizationStep(
                    threshold=target_k,
                    text=current_text,
                    ledger=ledger,
                    new_indices=[],
                    metadata={"processed_count": len(processed), "rank": current_rank},
                )
                continue

            candidates = self._sort_by_risk(
                [i for i in range(len(offsets)) if i not in processed],
                len(offsets),
            )
            new_indices: List[int] = []

            for idx in candidates:
                if current_rank >= target_k:
                    break

                apply_fn(idx, ledger)
                processed.add(idx)
                new_indices.append(idx)

                current_text = ledger.render_offsets(text)
                current_rank = self._rank_evaluator(current_text, self._target_label)

            yield AnonymizationStep(
                threshold=target_k,
                text=current_text,
                ledger=ledger,
                new_indices=new_indices,
                metadata={"processed_count": len(processed), "rank": current_rank},
            )


UntilKSelector = UntilKUnit
