from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Sequence
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import f1_score, mean_squared_error, r2_score
from sklearn.neural_network import MLPClassifier, MLPRegressor

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


class LogisticClassifier(SupervisedDownstreamHead):
    def __init__(self, multi_class: str = "auto"):
        super().__init__(name="logistic_classifier", primary_metric="f1")
        self.multi_class = multi_class
        self._estimator: Optional[LogisticRegression] = None

    def setup(self) -> None:
        self._estimator = LogisticRegression(
            max_iter=1000,
            multi_class=self.multi_class,
            n_jobs=None,
        )

    def fit(self, x: Any, y: Sequence[Any]) -> None:
        self.setup()
        self._estimator.fit(x, y)

    def predict(self, x: Any) -> Sequence[Any]:
        if self._estimator is None:
            raise RuntimeError("logistic classifier not fitted")
        return self._estimator.predict(x)

    def evaluate(self, x: Any, y: Sequence[Any]) -> Dict[str, float]:
        predictions = self.predict(x)
        label_count = len(set(y))
        average = "binary" if label_count == 2 else "macro"
        f1 = float(f1_score(y, predictions, average=average, zero_division=0))
        return {
            "f1": f1,
            "loss": 1.0 - f1,
        }

    def cleanup(self) -> None:
        self._estimator = None


class LinearRegressor(SupervisedDownstreamHead):
    def __init__(self):
        super().__init__(name="linear_regressor", primary_metric="r2")
        self._estimator: Optional[LinearRegression] = None

    def setup(self) -> None:
        self._estimator = LinearRegression()

    def fit(self, x: Any, y: Sequence[Any]) -> None:
        self.setup()
        self._estimator.fit(x, y)

    def predict(self, x: Any) -> Sequence[float]:
        if self._estimator is None:
            raise RuntimeError("linear regressor not fitted")
        values = self._estimator.predict(x)
        return np.asarray(values, dtype=float)

    def evaluate(self, x: Any, y: Sequence[Any]) -> Dict[str, float]:
        predictions = self.predict(x)
        r2 = float(r2_score(y, predictions))
        mse = float(mean_squared_error(y, predictions))
        return {
            "r2": r2,
            "rmse": float(np.sqrt(mse)),
        }

    def cleanup(self) -> None:
        self._estimator = None

class FeedForwardClassifier(SupervisedDownstreamHead):
    def __init__(self):
        super().__init__(name="feedforward_classifier", primary_metric="f1")
        self._estimator: Optional[MLPClassifier] = None

    def setup(self) -> None:
        self._estimator = MLPClassifier(hidden_layer_sizes=(128, 64), activation="relu", max_iter=200)

    def fit(self, x: Any, y: Sequence[Any]) -> None:
        self.setup()
        self._estimator.fit(x, y)

    def predict(self, x: Any) -> Sequence[Any]:
        if self._estimator is None:
            raise RuntimeError("feedforward classifier not fitted")
        return self._estimator.predict(x)

    def evaluate(self, x: Any, y: Sequence[Any]) -> Dict[str, float]:
        predictions = self.predict(x)
        label_count = len(set(y))
        average = "binary" if label_count == 2 else "macro"
        f1 = float(f1_score(y, predictions, average=average, zero_division=0))
        return {"f1": f1, "loss": 1.0 - f1}

    def cleanup(self) -> None:
        self._estimator = None

class FeedForwardRegressor(SupervisedDownstreamHead):
    def __init__(self):
        super().__init__(name="feedforward_regressor", primary_metric="r2")
        self._estimator: Optional[MLPRegressor] = None

    def setup(self) -> None:
        self._estimator = MLPRegressor(hidden_layer_sizes=(128, 64), activation="relu", max_iter=500)

    def fit(self, x: Any, y: Sequence[Any]) -> None:
        self.setup()
        self._estimator.fit(x, y)

    def predict(self, x: Any) -> Sequence[float]:
        if self._estimator is None:
            raise RuntimeError("feedforward regressor not fitted")
        values = self._estimator.predict(x)
        return np.asarray(values, dtype=float)

    def evaluate(self, x: Any, y: Sequence[Any]) -> Dict[str, float]:
        predictions = self.predict(x)
        mse = float(mean_squared_error(y, predictions))
        r2 = float(r2_score(y, predictions))
        return {"rmse": float(np.sqrt(mse)), "r2": r2}

    def cleanup(self) -> None:
        self._estimator = None

DOWNSTREAM_CLASSIFIER_HEAD_REGISTRY: Dict[str, type[SupervisedDownstreamHead]] = {
    "logistic_classifier": LogisticClassifier,
    "feedforward_classifier": FeedForwardClassifier,
}

DOWNSTREAM_REGRESSOR_HEAD_REGISTRY: Dict[str, type[SupervisedDownstreamHead]] = {
    "linear_regressor": LinearRegressor,
    "feedforward_regressor": FeedForwardRegressor,
}

DOWNSTREAM_HEAD_REGISTRY: Dict[str, type[SupervisedDownstreamHead]] = {
    **DOWNSTREAM_CLASSIFIER_HEAD_REGISTRY,
    **DOWNSTREAM_REGRESSOR_HEAD_REGISTRY,
}