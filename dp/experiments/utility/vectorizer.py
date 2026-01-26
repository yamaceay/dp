from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoModel, AutoTokenizer
import torch
try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

class SelfSupervisedFeatureExtractor(ABC):
    @abstractmethod
    def setup(self) -> "SelfSupervisedFeatureExtractor":
        raise NotImplementedError

    @abstractmethod
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

    @abstractmethod
    def clone(self) -> "SelfSupervisedFeatureExtractor":
        raise NotImplementedError


class TfidfTextVectorizer(SelfSupervisedFeatureExtractor):
    def __init__(
        self,
        ngram_range: Tuple[int, int] = (1, 2),
        max_features: int = 20000,
        min_df: int = 2,
        **params: Any,
    ):
        if isinstance(ngram_range, (list, tuple)) and len(ngram_range) == 2:
            ngram_range = (int(ngram_range[0]), int(ngram_range[1]))
        if "ngram_range" in params:
            nr = params["ngram_range"]
            if isinstance(nr, (list, tuple)) and len(nr) == 2:
                params["ngram_range"] = (int(nr[0]), int(nr[1]))
        self._params = {
            "ngram_range": ngram_range,
            "max_features": max_features,
            "min_df": min_df,
            **params,
        }
        self._model: Optional[TfidfVectorizer] = None

    def describe(self) -> Dict[str, Any]:
        return dict(self._params)

    def setup(self) -> "SelfSupervisedFeatureExtractor":
        self._model = TfidfVectorizer(**self.describe())
        return self

    def fit(self, texts: Sequence[str]) -> None:
        if self._model is None:
            self.setup()
        self._model.fit(list(texts))

    def transform(self, texts: Sequence[str]) -> Any:
        if self._model is None:
            self.setup()
        return self._model.transform(list(texts))

    def cleanup(self) -> None:
        self._model = None

    def clone(self) -> "SelfSupervisedFeatureExtractor":
        return TfidfTextVectorizer(**self.describe())

class BERTVectorizer(SelfSupervisedFeatureExtractor):
    def __init__(self, model_name: str = "roberta-base", batch_size: int = 32, device: str = "cpu"):
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device
        self._tokenizer: Optional[Any] = None
        self._model: Optional[Any] = None

    def describe(self) -> Dict[str, Any]:
        return {"model_name": self.model_name, "batch_size": self.batch_size, "device": self.device}

    def setup(self) -> "SelfSupervisedFeatureExtractor":
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModel.from_pretrained(self.model_name).to(self.device)
        return self

    def fit(self, texts: Sequence[str]) -> None:
        if self._model is None:
            self.setup()

    def cleanup(self) -> None:
        if self._model is not None:
            if self.device == "cuda":
                torch.cuda.empty_cache()
            elif self.device == "mps":
                torch.mps.empty_cache()
        self._model = None
        self._tokenizer = None

    def transform(self, texts: Sequence[str]) -> Any:
        if self._model is None or self._tokenizer is None:
            self.setup()
        all_embeddings: list[np.ndarray] = []
        for i in range(0, len(texts), self.batch_size):
            batch_texts = list(texts)[i : i + self.batch_size]
            encodings = self._tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self._model(**encodings)
                embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
                all_embeddings.append(embeddings)
        return np.vstack(all_embeddings) if all_embeddings else np.empty((0,))

    def clone(self) -> "SelfSupervisedFeatureExtractor":
        return BERTVectorizer(**self.describe())

class SentenceEmbeddingVectorizer(SelfSupervisedFeatureExtractor):
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        batch_size: int = 32,
        device: str = "cpu",
        backend: Optional[str] = None,
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device
        self.backend = backend
        self._sentence_model: Optional[Any] = None
        self._hf_vectorizer: Optional[BERTVectorizer] = None
        self._backend: Optional[str] = None

    def describe(self) -> Dict[str, Any]:
        payload = {
            "model_name": self.model_name,
            "batch_size": self.batch_size,
            "device": self.device,
        }
        if self.backend:
            payload["backend"] = self.backend
        return payload

    def setup(self) -> "SelfSupervisedFeatureExtractor":
        backend = (self.backend or "auto").lower()
        if backend in {"sentence_transformers", "sentence-transformers", "sbert", "st"}:
            if SentenceTransformer is None:
                raise RuntimeError("sentence-transformers is not available")
            self._sentence_model = SentenceTransformer(self.model_name, device=self.device)
            self._backend = "sentence_transformers"
            return self
        if backend in {"transformers", "hf", "bert"}:
            self._hf_vectorizer = BERTVectorizer(self.model_name, self.batch_size, self.device)
            self._hf_vectorizer.setup()
            self._backend = "transformers"
            return self
        if SentenceTransformer is not None and self.model_name.startswith("sentence-transformers/"):
            self._sentence_model = SentenceTransformer(self.model_name, device=self.device)
            self._backend = "sentence_transformers"
            return self
        self._hf_vectorizer = BERTVectorizer(self.model_name, self.batch_size, self.device)
        self._hf_vectorizer.setup()
        self._backend = "transformers"
        return self

    def fit(self, texts: Sequence[str]) -> None:
        if self._backend is None:
            self.setup()
        if self._backend == "transformers" and self._hf_vectorizer is not None:
            self._hf_vectorizer.fit(texts)

    def transform(self, texts: Sequence[str]) -> Any:
        if self._backend is None:
            self.setup()
        if self._backend == "sentence_transformers":
            if self._sentence_model is None:
                self.setup()
            return self._sentence_model.encode(
                list(texts),
                batch_size=self.batch_size,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
        if self._hf_vectorizer is None:
            self.setup()
        return self._hf_vectorizer.transform(texts)

    def cleanup(self) -> None:
        if self._hf_vectorizer is not None:
            self._hf_vectorizer.cleanup()
        self._sentence_model = None
        self._hf_vectorizer = None
        self._backend = None

    def clone(self) -> "SelfSupervisedFeatureExtractor":
        return SentenceEmbeddingVectorizer(**self.describe())

FEATURE_EXTRACTOR_REGISTRY: Dict[str, type[SelfSupervisedFeatureExtractor]] = {
    "tfidf": TfidfTextVectorizer,
    "bert": BERTVectorizer,
    "sentence": SentenceEmbeddingVectorizer,
    "sbert": SentenceEmbeddingVectorizer,
    "text": None,  # placeholder, filled below
}


class IdentityTextVectorizer(SelfSupervisedFeatureExtractor):
    def describe(self) -> Dict[str, Any]:
        return {"type": "text"}

    def setup(self) -> "SelfSupervisedFeatureExtractor":
        return self

    def fit(self, texts: Sequence[str]) -> None:
        return None

    def transform(self, texts: Sequence[str]) -> Any:
        return list(texts)

    def cleanup(self) -> None:
        return None

    def clone(self) -> "SelfSupervisedFeatureExtractor":
        return IdentityTextVectorizer()

FEATURE_EXTRACTOR_REGISTRY["text"] = IdentityTextVectorizer
