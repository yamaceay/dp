from abc import ABC, abstractmethod
from typing import Any, List, Callable, TypeVar, Generic, Optional, Tuple
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
    def __init__(self, tokenizer: Any, max_tokens: int, overlap_tokens: int = 0):
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens
        self.overlap_tokens = max(0, int(overlap_tokens))

    def _tokenize(self, text: str) -> Tuple[List[int], List[Tuple[int, int]]]:
        encoding = self.tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        tokens = list(encoding.get('input_ids', []))
        offsets = list(encoding.get('offset_mapping', []))
        return tokens, offsets

    def _find_target_token_index(self, offsets: List[Tuple[int, int]], span: Tuple[int, int]) -> Optional[int]:
        target_start, target_end = int(span[0]), int(span[1])
        for i, (start, end) in enumerate(offsets):
            if int(start) == target_start and int(end) == target_end:
                return i
        for i, (start, end) in enumerate(offsets):
            if target_start >= int(start) and target_end <= int(end):
                return i
        return None

    def centered_chunk(self, text: str, span: Tuple[int, int]) -> Chunk:
        tokens, offsets = self._tokenize(text)
        if len(tokens) <= self.max_tokens or not offsets:
            return Chunk(text=text, start=0, end=len(text))

        target_start, target_end = int(span[0]), int(span[1])

        if self.overlap_tokens > 0:
            candidates: List[Chunk] = []
            for chunk in self.chunk(text):
                if target_start >= int(chunk.start) and target_end <= int(chunk.end):
                    candidates.append(chunk)
            if candidates:
                target_center = (target_start + target_end) / 2.0
                return min(
                    candidates,
                    key=lambda c: (abs((((int(c.start) + int(c.end)) / 2.0) - target_center)), int(c.start)),
                )

        target_idx = self._find_target_token_index(offsets, span)
        if target_idx is None:
            return Chunk(text=text, start=0, end=len(text))

        window = int(self.max_tokens)
        half_window = window // 2
        token_start_idx = max(0, target_idx - half_window)
        token_end_idx = min(len(offsets), token_start_idx + window)
        token_start_idx = max(0, token_end_idx - window)

        chunk_char_start = int(offsets[token_start_idx][0])
        if token_end_idx >= len(offsets):
            chunk_char_end = len(text)
        else:
            chunk_char_end = int(offsets[token_end_idx][0])

        return Chunk(text=text[chunk_char_start:chunk_char_end], start=chunk_char_start, end=chunk_char_end)
    
    def chunk(self, text: str) -> List[Chunk]:
        tokens, offsets = self._tokenize(text)
        
        if len(tokens) <= self.max_tokens:
            return [Chunk(text=text, start=0, end=len(text))]
        
        chunks = []
        stride = max(1, int(self.max_tokens) - int(self.overlap_tokens))
        i = 0
        while i < len(tokens):
            end_idx = min(i + self.max_tokens, len(tokens))
            
            if offsets:
                start_char = offsets[i][0]
                end_char = offsets[end_idx - 1][1]
                while end_idx > i:
                    chunk_text = text[start_char:end_char]
                    chunk_tokens = self.tokenizer(chunk_text, add_special_tokens=False)["input_ids"]
                    if len(chunk_tokens) <= self.max_tokens:
                        break
                    end_idx -= 1
                    end_char = offsets[end_idx - 1][1]
            else:
                chunk_tokens = tokens[i:end_idx]
                decoded = self.tokenizer.decode(chunk_tokens, skip_special_tokens=True)
                start_char = text.find(decoded, 0 if not chunks else chunks[-1].end)
                if start_char == -1:
                    start_char = 0 if not chunks else chunks[-1].end
                end_char = start_char + len(decoded)
            
            chunk_text = text[start_char:end_char]
            chunks.append(Chunk(text=chunk_text, start=start_char, end=end_char))
            if end_idx >= len(tokens):
                break
            i = min(end_idx, i + stride)
        
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


def process_with_chunking(
    text: str,
    chunker: Any,
    processor: Callable[[str], R],
    aggregator: ChunkAggregator[R]
) -> R:
    chunks = chunker.chunk(text)
    results = [processor(chunk.text) for chunk in chunks]
    return aggregator.aggregate(results, chunks)
