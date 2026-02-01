from __future__ import annotations
from typing import Any, Dict, Optional, Sequence
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import f1_score, mean_squared_error, r2_score
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import LabelEncoder

from dp.bert import SupervisedDownstreamHead, BertClassifierHead, BertOrdinalHead, BertRegressorHead

class LogisticClassifier(SupervisedDownstreamHead):
    def __init__(
        self,
        multi_class: str = "auto",
        max_iter: int = 1000,
        class_weight: Optional[str | dict] = None,
        C: float = 1.0,
        solver: str = "lbfgs",
        n_jobs: Optional[int] = None,
        primary_metric: str = "f1",
    ):
        super().__init__(name="logistic_classifier", primary_metric=primary_metric)
        self.multi_class = multi_class
        self.max_iter = int(max_iter)
        self.class_weight = class_weight
        self.C = float(C)
        self.solver = solver
        self.n_jobs = n_jobs
        self._estimator: Optional[LogisticRegression] = None

    def setup(self) -> None:
        self._estimator = LogisticRegression(
            max_iter=self.max_iter,
            multi_class=self.multi_class,
            class_weight=self.class_weight,
            C=self.C,
            solver=self.solver,
            n_jobs=self.n_jobs,
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
        def _is_numeric(vals: Sequence[Any]) -> bool:
            try:
                for v in set(vals):
                    float(v)
                return True
            except Exception:
                return False
        average = "binary" if label_count == 2 and _is_numeric(y) else "macro"
        f1 = float(f1_score(y, predictions, average=average, zero_division=0))
        accuracy = float(np.mean(np.array(y) == np.array(predictions)))
        return {"f1": f1, "acc": accuracy}

    def cleanup(self) -> None:
        self._estimator = None

class LinearRegressor(SupervisedDownstreamHead):
    def __init__(self, primary_metric: str = "r2"):
        super().__init__(name="linear_regressor", primary_metric=primary_metric)
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
        return {"rmse": float(np.sqrt(mse)), "r2": r2}

    def cleanup(self) -> None:
        self._estimator = None

class FeedForwardClassifier(SupervisedDownstreamHead):
    def __init__(self, mlp_params: Optional[Dict[str, Any]] = None, primary_metric: str = "f1"):
        super().__init__(name="feedforward_classifier", primary_metric=primary_metric)
        self._estimator: Optional[MLPClassifier] = None
        self._mlp_params: Dict[str, Any] = dict(mlp_params or {})
        self._label_encoder: Optional[LabelEncoder] = None

    def setup(self) -> None:
        defaults: Dict[str, Any] = {"hidden_layer_sizes": (128, 64), "activation": "relu", "max_iter": 200}
        params = {**defaults, **self._mlp_params}
        self._estimator = MLPClassifier(**params)

    def fit(self, x: Any, y: Sequence[Any]) -> None:
        self.setup()
        self._label_encoder = LabelEncoder()
        y_enc = self._label_encoder.fit_transform(list(y))
        self._estimator.fit(x, y_enc)

    def predict(self, x: Any) -> Sequence[Any]:
        if self._estimator is None:
            raise RuntimeError("feedforward classifier not fitted")
        preds = self._estimator.predict(x)
        if self._label_encoder is None:
            return preds
        return self._label_encoder.inverse_transform(preds)

    def evaluate(self, x: Any, y: Sequence[Any]) -> Dict[str, float]:
        predictions = self.predict(x)
        label_count = len(set(y))
        def _is_numeric(vals: Sequence[Any]) -> bool:
            try:
                for v in set(vals):
                    float(v)
                return True
            except Exception:
                return False
        average = "binary" if label_count == 2 and _is_numeric(y) else "macro"
        f1 = float(f1_score(y, predictions, average=average, zero_division=0))
        accuracy = float(np.mean(np.array(y) == np.array(predictions)))
        return {"f1": f1, "acc": accuracy}

    def cleanup(self) -> None:
        self._estimator = None

class FeedForwardRegressor(SupervisedDownstreamHead):
    def __init__(self, mlp_params: Optional[Dict[str, Any]] = None, primary_metric: str = "r2"):
        super().__init__(name="feedforward_regressor", primary_metric=primary_metric)
        self._estimator: Optional[MLPRegressor] = None
        self._mlp_params: Dict[str, Any] = dict(mlp_params or {})

    def setup(self) -> None:
        defaults: Dict[str, Any] = {"hidden_layer_sizes": (128, 64), "activation": "relu", "max_iter": 500}
        params = {**defaults, **self._mlp_params}
        self._estimator = MLPRegressor(**params)

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
    "bert_classifier": BertClassifierHead,
    "bert_ordinal": BertOrdinalHead,
}

DOWNSTREAM_REGRESSOR_HEAD_REGISTRY: Dict[str, type[SupervisedDownstreamHead]] = {
    "linear_regressor": LinearRegressor,
    "feedforward_regressor": FeedForwardRegressor,
    "bert_regressor": BertRegressorHead,
}

DOWNSTREAM_HEAD_REGISTRY: Dict[str, type[SupervisedDownstreamHead]] = {
    **DOWNSTREAM_CLASSIFIER_HEAD_REGISTRY,
    **DOWNSTREAM_REGRESSOR_HEAD_REGISTRY,
}
