from typing import List, Tuple
import re
from nltk.tokenize import TreebankWordTokenizer


class TextSplitter:
    def __init__(self):
        self.tokenizer = TreebankWordTokenizer()
    
    def split_sentences(self, text: str) -> List[Tuple[int, int]]:
        pattern = re.compile(r'[.!?]+\s+')
        splits = []
        start = 0
        
        for match in pattern.finditer(text):
            end = match.end()
            splits.append((start, end))
            start = end
        
        if start < len(text):
            splits.append((start, len(text)))
        
        if not splits:
            splits.append((0, len(text)))
        
        return splits
    
    def tokenize_with_spans(self, text: str) -> List[Tuple[int, int, str]]:
        terms = []
        for start, end in self.tokenizer.span_tokenize(text):
            term_text = text[start:end]
            terms.append((start, end, term_text))
        return terms
    
    def map_tokens_to_nltk(
        self, 
        text: str, 
        token_spans: List[Tuple[int, int]]
    ) -> List[List[int]]:
        nltk_spans = list(self.tokenizer.span_tokenize(text))
        mapping = []
        
        for token_start, token_end in token_spans:
            overlapping_idxs = []
            for idx, (nltk_start, nltk_end) in enumerate(nltk_spans):
                if not (nltk_end <= token_start or nltk_start >= token_end):
                    overlapping_idxs.append(idx)
            mapping.append(overlapping_idxs)
        
        return mapping
