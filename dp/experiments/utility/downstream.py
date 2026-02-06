from __future__ import annotations
from typing import Any, Dict, Optional, Sequence
import json
import re
import urllib.request
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

    def fit(self, x: Any, y: Sequence[Any], *args: Any) -> None:
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

    def fit(self, x: Any, y: Sequence[Any], *args: Any) -> None:
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

    def fit(self, x: Any, y: Sequence[Any], *args: Any) -> None:
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


def _normalize_text_label(value: Any) -> str:
    return str(value).strip()


def _extract_json_field(raw: str, field: str) -> Optional[Any]:
    text = raw.strip()
    try:
        data = json.loads(text)
        if isinstance(data, dict) and field in data:
            return data[field]
    except Exception:
        pass
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        return None
    try:
        data = json.loads(m.group(0))
        if isinstance(data, dict):
            return data.get(field)
    except Exception:
        return None
    return None


class _QwenRestBaseHead(SupervisedDownstreamHead):
    def __init__(
        self,
        *,
        name: str,
        primary_metric: str,
        endpoint_id: str = "qwen",
        server_info_file: str = "logs/vllm_server.json",
        url: str = "",
        timeout: float = 600.0,
        temperature: float = 0.0,
        top_p: float = 1.0,
        max_new_tokens: int = 64,
        system_prompt: str = "You are a strict and concise classifier. Return only valid JSON.",
    ):
        super().__init__(name=name, primary_metric=primary_metric)
        self.endpoint_id = endpoint_id
        self.server_info_file = server_info_file
        self.url = url
        self.timeout = float(timeout)
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.max_new_tokens = int(max_new_tokens)
        self.system_prompt = system_prompt
        self._base_url: Optional[str] = None

    def setup(self) -> None:
        if self.url:
            self._base_url = self.url.rstrip("/")
            return
        with open(self.server_info_file, "r", encoding="utf-8") as f:
            info = json.load(f)
        self._base_url = str(info["base_url"]).rstrip("/")

    def _post_infer(self, prompt: str) -> str:
        if not self._base_url:
            self.setup()
        payload = {
            "endpoint_id": self.endpoint_id,
            "prompt": prompt,
            "chat": True,
            "system_prompt": self.system_prompt,
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
        }
        body = json.dumps(payload, ensure_ascii=True).encode("utf-8")
        req = urllib.request.Request(
            url=f"{self._base_url}/infer",
            data=body,
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        if not data.get("ok", False):
            raise RuntimeError(f"provider infer failed: {data}")
        return str(data.get("response", "")).strip()

    def cleanup(self) -> None:
        self._base_url = None


class QwenRestClassifierHead(_QwenRestBaseHead):
    def __init__(self, **kwargs: Any):
        super().__init__(name="qwen_classifier", primary_metric="macro_f1", **kwargs)
        self._labels: list[str] = []

    def fit(self, x: Any, y: Sequence[Any]) -> None:
        self.setup()
        labels = sorted({_normalize_text_label(v) for v in y if _normalize_text_label(v)})
        if not labels:
            raise ValueError("qwen_classifier requires non-empty labels")
        self._labels = labels

    def _parse_label(self, response: str) -> str:
        candidate = _extract_json_field(response, "label")
        if candidate is not None:
            s = _normalize_text_label(candidate)
            if s in self._labels:
                return s
        low = response.lower()
        for lbl in self._labels:
            if lbl.lower() in low:
                return lbl
        return self._labels[0]

    def predict(self, x: Any) -> Sequence[Any]:
        if not self._labels:
            raise RuntimeError("qwen_classifier not fitted")
        preds: list[str] = []
        for text in list(x):
            prompt = (
                "Choose one label and return JSON only.\n"
                f"Labels: {self._labels}\n"
                f"Text: {text}\n"
                "Output format: {\"label\": \"<one_label>\"}"
            )
            response = self._post_infer(prompt)
            preds.append(self._parse_label(response))
        return preds

    def evaluate(self, x: Any, y: Sequence[Any]) -> Dict[str, float]:
        predictions = self.predict(x)
        label_count = len(set(y))
        average = "binary" if label_count == 2 else "macro"
        f1 = float(f1_score(y, predictions, average=average, zero_division=0))
        accuracy = float(np.mean(np.array(y) == np.array(predictions)))
        return {"f1": f1, "acc": accuracy, "macro_f1": f1}


class QwenRestOrdinalHead(_QwenRestBaseHead):
    def __init__(self, **kwargs: Any):
        super().__init__(name="qwen_ordinal", primary_metric="macro_mae", **kwargs)
        self._label_order: list[str] = []
        self._label_to_idx: dict[str, int] = {}

    def set_label_order(self, labels: Sequence[Any]) -> None:
        self._label_order = [_normalize_text_label(v) for v in labels if _normalize_text_label(v)]
        self._label_to_idx = {v: i for i, v in enumerate(self._label_order)}

    def fit(self, x: Any, y: Sequence[Any]) -> None:
        self.setup()
        if not self._label_order:
            self.set_label_order(sorted({_normalize_text_label(v) for v in y if _normalize_text_label(v)}))
        if not self._label_order:
            raise ValueError("qwen_ordinal requires non-empty labels")

    def _parse_label(self, response: str) -> str:
        candidate = _extract_json_field(response, "label")
        if candidate is not None:
            s = _normalize_text_label(candidate)
            if s in self._label_to_idx:
                return s
        low = response.lower()
        for lbl in self._label_order:
            if lbl.lower() in low:
                return lbl
        return self._label_order[0]

    def predict(self, x: Any) -> Sequence[Any]:
        if not self._label_order:
            raise RuntimeError("qwen_ordinal not fitted")
        preds: list[str] = []
        for text in list(x):
            prompt = (
                "Choose one ordered label and return JSON only.\n"
                f"Ordered labels (low->high): {self._label_order}\n"
                f"Text: {text}\n"
                "Output format: {\"label\": \"<one_label>\"}"
            )
            response = self._post_infer(prompt)
            preds.append(self._parse_label(response))
        return preds

    def evaluate(self, x: Any, y: Sequence[Any]) -> Dict[str, float]:
        predictions = self.predict(x)
        y_idx = np.array([self._label_to_idx[_normalize_text_label(v)] for v in y], dtype=float)
        p_idx = np.array([self._label_to_idx[_normalize_text_label(v)] for v in predictions], dtype=float)
        macro_mae = float(np.mean(np.abs(y_idx - p_idx)))
        accuracy = float(np.mean(y_idx == p_idx))
        return {"macro_mae": macro_mae, "acc": accuracy}


class QwenRestRegressorHead(_QwenRestBaseHead):
    def __init__(self, **kwargs: Any):
        super().__init__(name="qwen_regressor", primary_metric="r2", **kwargs)

    def fit(self, x: Any, y: Sequence[Any]) -> None:
        self.setup()

    def _parse_value(self, response: str) -> float:
        candidate = _extract_json_field(response, "value")
        if candidate is not None:
            try:
                return float(candidate)
            except Exception:
                pass
        m = re.search(r"-?\d+(?:\.\d+)?", response)
        if m:
            return float(m.group(0))
        return 0.0

    def predict(self, x: Any) -> Sequence[float]:
        preds: list[float] = []
        for text in list(x):
            prompt = (
                "Estimate a numeric value and return JSON only.\n"
                f"Text: {text}\n"
                "Output format: {\"value\": <number>}"
            )
            response = self._post_infer(prompt)
            preds.append(self._parse_value(response))
        return np.asarray(preds, dtype=float)

    def evaluate(self, x: Any, y: Sequence[Any]) -> Dict[str, float]:
        predictions = self.predict(x)
        mse = float(mean_squared_error(y, predictions))
        r2 = float(r2_score(y, predictions))
        return {"rmse": float(np.sqrt(mse)), "r2": r2}


DOWNSTREAM_CLASSIFIER_HEAD_REGISTRY: Dict[str, type[SupervisedDownstreamHead]] = {
    "logistic_classifier": LogisticClassifier,
    "feedforward_classifier": FeedForwardClassifier,
    "bert_classifier": BertClassifierHead,
    "qwen_classifier": QwenRestClassifierHead,
    "bert_ordinal": BertOrdinalHead,
    "qwen_ordinal": QwenRestOrdinalHead,
}

DOWNSTREAM_REGRESSOR_HEAD_REGISTRY: Dict[str, type[SupervisedDownstreamHead]] = {
    "linear_regressor": LinearRegressor,
    "feedforward_regressor": FeedForwardRegressor,
    "bert_regressor": BertRegressorHead,
    "qwen_regressor": QwenRestRegressorHead,
}

DOWNSTREAM_HEAD_REGISTRY: Dict[str, type[SupervisedDownstreamHead]] = {
    **DOWNSTREAM_CLASSIFIER_HEAD_REGISTRY,
    **DOWNSTREAM_REGRESSOR_HEAD_REGISTRY,
}
