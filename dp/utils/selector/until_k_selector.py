from typing import List, Tuple
from dp.loaders.base import TextAnnotation
from dp.utils.selector.base import TokenSelector


class UntilKSelector(TokenSelector):
    def __init__(self, k: int, **kwargs):
        super().__init__(**kwargs)
        self.set_k(k)

    def set_k(self, k: int) -> None:
        if k < 1:
            raise ValueError("k must be at least 1")
        self._k = k

    def select(self, text: str, offsets: List[Tuple[int, int]]) -> List[TextAnnotation]:
        if not text or not text.strip() or not offsets:
            return []
        
        return [
            TextAnnotation(
                text=text[start:end],
                start_offset=start,
                end_offset=end,
                label="PII"
            )
            for start, end in offsets[:self._k]
        ]
