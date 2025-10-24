from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Sequence

from sklearn.feature_extraction.text import TfidfVectorizer


class TextVectorizer(ABC):
    @abstractmethod
    def clone(self) -> "TextVectorizer":
        raise NotImplementedError

    @abstractmethod
    def fit(self, texts: Sequence[str]) -> None:
        raise NotImplementedError

    @abstractmethod
    def transform(self, texts: Sequence[str]) -> Any:
        raise NotImplementedError

    def fit_transform(self, texts: Sequence[str]) -> Any:
        self.fit(texts)
        return self.transform(texts)

    def describe(self) -> Dict[str, Any]:
        return {}


class TfidfTextVectorizer(TextVectorizer):
    def __init__(self, vectorizer: Optional[TfidfVectorizer] = None, **params: Any):
        if vectorizer is not None and params:
            raise ValueError("vectorizer and params are mutually exclusive")
        if vectorizer is not None:
            self._params = vectorizer.get_params(deep=False)
        else:
            defaults = {"ngram_range": (1, 2), "max_features": 20000, "min_df": 2}
            self._params = dict(defaults)
            self._params.update(params)
        self._model: Optional[TfidfVectorizer] = None

    def clone(self) -> "TfidfTextVectorizer":
        return TfidfTextVectorizer(**self._params)

    def fit(self, texts: Sequence[str]) -> None:
        model = TfidfVectorizer(**self._params)
        model.fit(list(texts))
        self._model = model

    def transform(self, texts: Sequence[str]) -> Any:
        if self._model is None:
            raise RuntimeError("vectorizer is not fitted")
        return self._model.transform(list(texts))

    def fit_transform(self, texts: Sequence[str]) -> Any:
        model = TfidfVectorizer(**self._params)
        matrix = model.fit_transform(list(texts))
        self._model = model
        return matrix

    def describe(self) -> Dict[str, Any]:
        return dict(self._params)
