"""Utils package re-exporting utility types and registry.

This module exposes utility classes for selectors and explainers
used in anonymization strategies.

Note: TRIDetectorWithDeid is now available from dp.tri, not dp.utils.
"""

from dp.utils.pii_detector import PIIDetector, PIIDataset
from dp.utils.selector import AnonymizerUnit, AllSelector, PIIOnlySelector, ByRiskSelector
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
    "AnonymizerUnit",
    "AllSelector",
    "PIIOnlySelector",
    "ByRiskSelector",
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
