from abc import ABC, abstractmethod
from transformers import pipeline

from dp.utils.chunking import TokenAwareChunker

class Rewriter(ABC):
    @abstractmethod
    def rewrite(self, text: str, **kwargs) -> str:
        raise NotImplementedError

class BartRewriter(Rewriter):
    def __init__(self, model_name: str = "facebook/bart-large-cnn", device: int = -1, max_input_tokens: int = 512):
        self.rewriting_pipeline = pipeline("summarization", model=model_name, device=device)
        self.max_input_tokens = max_input_tokens
        self.chunker = TokenAwareChunker(
            tokenizer=self.rewriting_pipeline.tokenizer,
            max_tokens=self.max_input_tokens,
        )

    def rewrite(self, text: str, max_length: int = 150, min_length: int = 40, **kwargs) -> str:
        chunks = self.chunker.chunk(text)
        
        summaries = []
        for chunk in chunks:
            summary = self.rewriting_pipeline(
                chunk.text,
                max_length=max_length,
                min_length=min_length,
                **kwargs
            )
            summaries.append(summary[0]['summary_text'])
        
        return " ".join(summaries)