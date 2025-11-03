from abc import ABC, abstractmethod
from transformers import pipeline

class Summarizer(ABC):
    @abstractmethod
    def summarize(self, text: str, **kwargs) -> str:
        raise NotImplementedError

class BartSummarizer(Summarizer):
    def __init__(self, model_name: str = "facebook/bart-large-cnn", device: int = -1):
        self.summarization_pipeline = pipeline("summarization", model=model_name, device=device)

    def summarize(self, text: str, max_length: int = 150, min_length: int = 40, do_sample: bool = False, *args, **kwargs) -> str:
        summary = self.summarization_pipeline(
            text,
            max_length=max_length,
            min_length=min_length,
            do_sample=do_sample,
        )
        return summary[0]['summary_text']