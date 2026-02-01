from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Sequence


class SupervisedDownstreamHead(ABC):
    def __init__(self, name: str, primary_metric: str):
        if not name:
            raise ValueError("model name is required")
        if not primary_metric:
            raise ValueError("primary metric is required")
        self.name = name
        self.primary_metric = primary_metric

    @abstractmethod
    def setup(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def fit(self, x: Any, y: Sequence[Any]) -> None:
        raise NotImplementedError

    @abstractmethod
    def predict(self, x: Any) -> Sequence[Any]:
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, x: Any, y: Sequence[Any]) -> Dict[str, float]:
        raise NotImplementedError

    @abstractmethod
    def cleanup(self) -> None:
        raise NotImplementedError
