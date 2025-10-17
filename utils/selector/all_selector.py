from dp.utils.selector.base import TokenSelector

class AllSelector(TokenSelector):
    """Selector that returns no PII spans, allowing all tokens to be privatized."""
    
    def select(self, text: str) -> list:
        """
        Return empty list indicating no tokens should be skipped.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Empty list (all tokens will be privatized)
        """
        return []
