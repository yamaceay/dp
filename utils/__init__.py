"""Utils package re-exporting utility types and registry.

This module exposes utility classes for selectors and explainers
used in anonymization strategies.
"""

from dp.utils.pii_detector import PIIDetector, PIIDataset
from dp.utils.tri_detector import TRIDetector
from dp.utils.selector import TokenSelector, AllSelector, PIIOnlySelector
from dp.utils.explainer import TokenExplainer, UniformExplainer, GreedyExplainer, ShapExplainer
from dp.utils.chunking import (
    Chunk,
    ChunkAggregator,
    TruncateChunker,
    SlidingWindowChunker,
    TokenAwareChunker,
    MaxScoreAggregator,
    AverageAggregator,
    SpanMergeAggregator,
    ProbabilityAggregator,
    process_with_chunking,
)

__all__ = [
    "PIIDetector",
    "PIIDataset",
    "TRIDetector",
    "TokenSelector",
    "AllSelector",
    "PIIOnlySelector",
    "TokenExplainer",
    "UniformExplainer",
    "GreedyExplainer",
    "ShapExplainer",
    "Chunk",
    "ChunkAggregator",
    "TruncateChunker",
    "SlidingWindowChunker",
    "TokenAwareChunker",
    "MaxScoreAggregator",
    "AverageAggregator",
    "SpanMergeAggregator",
    "ProbabilityAggregator",
    "process_with_chunking",
]
