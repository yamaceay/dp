from abc import ABC, abstractmethod
from typing import Any, List, Callable, TypeVar, Generic
from dataclasses import dataclass

T = TypeVar('T')
R = TypeVar('R')


@dataclass
class Chunk:
    text: str
    start: int
    end: int


class ChunkAggregator(ABC, Generic[R]):
    @abstractmethod
    def aggregate(self, results: List[R], chunks: List[Chunk]) -> R:
        pass


class TruncateChunker:
    def __init__(self, max_length: int):
        self.max_length = max_length
    
    def chunk(self, text: str) -> List[Chunk]:
        if len(text) <= self.max_length:
            return [Chunk(text=text, start=0, end=len(text))]
        return [Chunk(text=text[:self.max_length], start=0, end=self.max_length)]


class SlidingWindowChunker:
    def __init__(self, max_length: int, overlap: int = 50):
        self.max_length = max_length
        self.overlap = overlap
    
    def chunk(self, text: str) -> List[Chunk]:
        if len(text) <= self.max_length:
            return [Chunk(text=text, start=0, end=len(text))]
        
        chunks = []
        stride = self.max_length - self.overlap
        start = 0
        
        while start < len(text):
            end = min(start + self.max_length, len(text))
            chunks.append(Chunk(text=text[start:end], start=start, end=end))
            if end == len(text):
                break
            start += stride
        
        return chunks


class TokenAwareChunker:
    def __init__(self, tokenizer: Any, max_tokens: int):
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens
    
    def chunk(self, text: str) -> List[Chunk]:
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        
        if len(tokens) <= self.max_tokens:
            return [Chunk(text=text, start=0, end=len(text))]
        
        chunks = []
        token_chunks = [tokens[i:i + self.max_tokens] for i in range(0, len(tokens), self.max_tokens)]
        
        pos = 0
        for token_chunk in token_chunks:
            chunk_text = self.tokenizer.decode(token_chunk, skip_special_tokens=True)
            start = pos
            end = start + len(chunk_text)
            chunks.append(Chunk(text=chunk_text, start=start, end=end))
            pos = end
        
        return chunks


class MaxScoreAggregator(ChunkAggregator[float]):
    def aggregate(self, results: List[float], chunks: List[Chunk]) -> float:
        return max(results) if results else 0.0


class AverageAggregator(ChunkAggregator[float]):
    def aggregate(self, results: List[float], chunks: List[Chunk]) -> float:
        return sum(results) / len(results) if results else 0.0


class SpanMergeAggregator(ChunkAggregator[List[dict]]):
    def aggregate(self, results: List[List[dict]], chunks: List[Chunk]) -> List[dict]:
        all_spans = []
        for chunk_spans, chunk in zip(results, chunks):
            for span in chunk_spans:
                adjusted_span = dict(span)
                adjusted_span['start'] = span.get('start', 0) + chunk.start
                adjusted_span['end'] = span.get('end', 0) + chunk.start
                all_spans.append(adjusted_span)
        
        all_spans.sort(key=lambda s: s.get('start', 0))
        
        if not all_spans:
            return []
        
        merged = [all_spans[0]]
        for span in all_spans[1:]:
            last = merged[-1]
            if span['start'] <= last['end'] and span.get('label') == last.get('label'):
                last['end'] = max(last['end'], span['end'])
            else:
                merged.append(span)
        
        return merged


class ProbabilityAggregator(ChunkAggregator[dict]):
    def aggregate(self, results: List[dict], chunks: List[Chunk]) -> dict:
        if not results:
            return {}
        
        label_probs = {}
        for result in results:
            for label, prob in result.items():
                if label not in label_probs:
                    label_probs[label] = []
                label_probs[label].append(prob)
        
        return {label: max(probs) for label, probs in label_probs.items()}


def process_with_chunking(
    text: str,
    chunker: Any,
    processor: Callable[[str], R],
    aggregator: ChunkAggregator[R]
) -> R:
    chunks = chunker.chunk(text)
    results = [processor(chunk.text) for chunk in chunks]
    return aggregator.aggregate(results, chunks)
