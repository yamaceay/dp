from typing import List, Optional, Tuple
from dp.loaders.base import TextAnnotation
from dp.utils.selector.base import TokenSelector

class AllSelector(TokenSelector):
    """Selector that returns no PII spans, allowing all tokens to be privatized."""

    def select(self, text: str, offsets: Optional[List[Tuple[int, int]]] = None) -> list:
        """
        Return all tokens indicating no tokens should be skipped.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Empty list (all tokens will be privatized)
        """
        if not text or not text.strip():
            return []
        
        if offsets is None:
            return []
        
        annotations = []
        for offset in offsets:
            token = text[offset[0]:offset[1]]
            annotations.append(TextAnnotation(*offset, text=token))
        return annotations
