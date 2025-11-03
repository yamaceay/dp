from abc import ABC, abstractmethod
from transformers import pipeline

from dp.utils.chunking import TokenAwareChunker

class Summarizer(ABC):
    @abstractmethod
    def summarize(self, text: str, **kwargs) -> str:
        raise NotImplementedError

class BartSummarizer(Summarizer):
    def __init__(self, model_name: str = "facebook/bart-large-cnn", device: int = -1, max_input_tokens: int = 512):
        self.summarization_pipeline = pipeline("summarization", model=model_name, device=device)
        self.max_input_tokens = max_input_tokens
        self.chunker = TokenAwareChunker(
            tokenizer=self.summarization_pipeline.tokenizer,
            max_tokens=self.max_input_tokens,
        )

    def summarize(self, text: str, max_length: int = 150, min_length: int = 40, do_sample: bool = False) -> str:
        chunks = self.chunker.chunk(text)
        
        summaries = []
        for chunk in chunks:
            summary = self.summarization_pipeline(
                chunk.text,
                max_length=max_length,
                min_length=min_length,
                do_sample=do_sample,
            )
            summaries.append(summary[0]['summary_text'])
        
        return " ".join(summaries)