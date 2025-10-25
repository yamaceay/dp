from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import f1_score, mean_squared_error, r2_score

from torch.nn import Sequential, Dense

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
        super().__init__(name="feed_forward_classifier", primary_metric="f1")
        self._model: Optional[Sequential] = None

    def setup(self) -> None:
        self._model = Sequential()
        self._model.add(Dense(128, activation='relu', input_shape=(self.input_dim,)))
        self._model.add(Dense(64, activation='relu'))
        self._model.add(Dense(self.output_dim, activation='softmax'))
        self._model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    def fit(self, x: Any, y: Sequence[Any]) -> None:
        self.setup()
        self._model.fit(x, y, epochs=10, batch_size=32)

    def predict(self, x: Any) -> Sequence[Any]:
        if self._model is None:
            raise RuntimeError("feed-forward classifier not fitted")
        return self._model.predict(x)

    def evaluate(self, x: Any, y: Sequence[Any]) -> Dict[str, float]:
        if self._model is None:
            raise RuntimeError("feed-forward classifier not fitted")
        loss, accuracy = self._model.evaluate(x, y)
        return {
            "loss": loss,
            "accuracy": accuracy,
        }

    def cleanup(self) -> None:
        if self._model is not None:
            self._model.delete()
            self._model = None

class FeedForwardRegressor(SupervisedDownstreamHead):
    def __init__(self):
        super().__init__(name="feed_forward_regressor", primary_metric="r2")
        self._model: Optional[Sequential] = None

    def setup(self) -> None:
        self._model = Sequential()
        self._model.add(Dense(128, activation='relu', input_shape=(self.input_dim,)))
        self._model.add(Dense(64, activation='relu'))
        self._model.add(Dense(1))
        self._model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

    def fit(self, x: Any, y: Sequence[Any]) -> None:
        self.setup()
        self._model.fit(x, y, epochs=10, batch_size=32)

    def predict(self, x: Any) -> Sequence[float]:
        if self._model is None:
            raise RuntimeError("feed-forward regressor not fitted")
        values = self._model.predict(x)
        return np.asarray(values.flatten(), dtype=float)

    def evaluate(self, x: Any, y: Sequence[Any]) -> Dict[str, float]:
        if self._model is None:
            raise RuntimeError("feed-forward regressor not fitted")
        loss, mae = self._model.evaluate(x, y)
        return {
            "loss": loss,
            "mae": mae,
        }

    def cleanup(self) -> None:
        if self._model is not None:
            self._model.delete()
            self._model = None