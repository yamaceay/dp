"""
Base stub for selectors. Extend for all_selector, pii_only_selector, etc.
"""
class TokenSelector:
    def __init__(self, *args, **kwargs):
        pass
    def select(self, text):
        """Stub select method. Implement in subclass."""
        raise NotImplementedError("TokenSelector is a stub.")
