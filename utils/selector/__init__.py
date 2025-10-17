"""
Selector utilities for PII detection and token filtering.

This module provides selector classes that determine which tokens should be
protected (skipped) during anonymization:

- AllSelector: Returns no PII spans (all tokens are privatized)
- PIIOnlySelector: Identifies PII spans to skip during privatization (stub - needs implementation)
"""

from dp.utils.selector.base import TokenSelector
from dp.utils.selector.all_selector import AllSelector
from dp.utils.selector.pii_only_selector import PIIOnlySelector

__all__ = [
    "TokenSelector",
    "AllSelector",
    "PIIOnlySelector",
]