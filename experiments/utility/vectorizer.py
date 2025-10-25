from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Sequence, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer

from transformers import AutoModel, AutoTokenizer
import torch

class SelfSupervisedFeatureExtractor(ABC):
    @abstractmethod
    def setup(self) -> "SelfSupervisedFeatureExtractor":
        raise NotImplementedError

    def describe(self) -> Dict[str, Any]:
        raise NotImplementedError
    
    @abstractmethod
    def cleanup(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def fit(self, texts: Sequence[str]) -> None:
        raise NotImplementedError

    @abstractmethod
    def transform(self, texts: Sequence[str]) -> Any:
        raise NotImplementedError


class TfidfTextVectorizer(SelfSupervisedFeatureExtractor):
    def __init__(self, 
                 ngram_range: Tuple[int, int] = (1, 2), 
                 max_features: int = 20000, 
                 min_df: int = 2, 
                 **params: Any
    ):
        self._params = {
            "ngram_range": ngram_range,
            "max_features": max_features,
            "min_df": min_df,
            **params
        }
        self._model: Optional[TfidfVectorizer] = None

    def describe(self) -> Dict[str, Any]:
        return self._params

    def setup(self) -> None:
        self._model = TfidfVectorizer(**self.describe())

    def fit(self, texts: Sequence[str]) -> None:
        self.setup()
        self._model.fit(list(texts))
        self._model = model

    def transform(self, texts: Sequence[str]) -> Any:
        if self._model is None:
            self.setup()
        return self._model.transform(list(texts))

    def cleanup(self) -> None:
        self._model = None

class BERTVectorizer(SelfSupervisedFeatureExtractor):
    def __init__(self, model_name: str = "bert-base-uncased", batch_size: int = 32, device: str = "cpu"):
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device

    def setup(self) -> None:
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModel.from_pretrained(self.model_name).to(self.device)

    def fit(self, texts: Sequence[str]) -> None:
        pass  # No fitting required for BERT embeddings

    def cleanup(self) -> None:
        if hasattr(self, '_model') and self._model is not None:
            if self.device == "cuda":
                torch.cuda.empty_cache()
            elif self.device == "mps":
                torch.mps.empty_cache()
            self._model = None
        if hasattr(self, '_tokenizer'):
            self._tokenizer = None

    def transform(self, texts: Sequence[str]) -> Any:
        all_embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i : i + self.batch_size]
            encodings = self._tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self._model(**encodings)
                embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
                all_embeddings.append(embeddings)
        return np.vstack(all_embeddings)