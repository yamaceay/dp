from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import f1_score, mean_squared_error, r2_score

from dp.experiments.utility.base import DownstreamModel

class LogisticClassifier(DownstreamModel):
    def __init__(self, multi_class: str = "auto"):
        super().__init__(name="logistic_classifier", primary_metric="f1")
        self.multi_class = multi_class
        self._estimator: Optional[LogisticRegression] = None

    def clone(self) -> "LogisticClassifier":
        return LogisticClassifier(multi_class=self.multi_class)

    def fit(self, x: Any, y: Sequence[Any]) -> None:
        estimator = LogisticRegression(
            max_iter=1000,
            multi_class=self.multi_class,
            n_jobs=None,
        )
        estimator.fit(x, y)
        self._estimator = estimator

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


class LinearRegressor(DownstreamModel):
    def __init__(self):
        super().__init__(name="linear_regressor", primary_metric="r2")
        self._estimator: Optional[LinearRegression] = None

    def clone(self) -> "LinearRegressor":
        return LinearRegressor()

    def fit(self, x: Any, y: Sequence[Any]) -> None:
        estimator = LinearRegression()
        estimator.fit(x, y)
        self._estimator = estimator

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
