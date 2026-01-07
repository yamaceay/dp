from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Sequence
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import f1_score, mean_squared_error, r2_score
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import LabelEncoder
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

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

def _resolve_device(preferred: Optional[str] = None) -> str:
    if preferred and preferred in {"cpu", "cuda", "mps"}:
        return preferred
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

class BertClassifierHead(SupervisedDownstreamHead):
    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        batch_size: int = 8,
        epochs: int = 2,
        lr: float = 2e-5,
        device: Optional[str] = None,
        primary_metric: str = "f1",
        target_acc: Optional[float] = None,
        init_checkpoint: Optional[str] = None,
        checkpoint_dir: Optional[str] = None,
    ):
        super().__init__(name="bert_classifier", primary_metric=primary_metric)
        self.model_name = model_name
        self.batch_size = int(batch_size)
        self.epochs = int(epochs)
        self.lr = float(lr)
        self.device = _resolve_device(device)
        self.target_acc: Optional[float] = float(target_acc) if target_acc is not None else None
        self.init_checkpoint = init_checkpoint
        self.checkpoint_dir = checkpoint_dir
        self._tokenizer: Optional[AutoTokenizer] = None
        self._model: Optional[AutoModelForSequenceClassification] = None
        self._label_encoder: Optional[LabelEncoder] = None

    def setup(self) -> None:
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(self.model_name, num_labels=2)
        self._model.to(self.device)
        if self.init_checkpoint:
            try:
                src = AutoModelForSequenceClassification.from_pretrained(self.init_checkpoint)
                src.to(self.device)
                state = src.state_dict()
                base_keys = [k for k in state.keys() if not k.startswith("classifier")]
                filtered = {k: state[k] for k in base_keys}
                if hasattr(self._model, "bert") and hasattr(src, "bert"):
                    self._model.bert.load_state_dict(filtered, strict=False)
                elif hasattr(self._model, "roberta") and hasattr(src, "roberta"):
                    self._model.roberta.load_state_dict(filtered, strict=False)
                elif hasattr(self._model, "distilbert") and hasattr(src, "distilbert"):
                    self._model.distilbert.load_state_dict(filtered, strict=False)
            except Exception:
                pass

    def fit(self, x: Any, y: Sequence[Any]) -> None:
        texts = list(x)
        if not texts or not isinstance(texts[0], str):
            raise TypeError("bert classifier expects raw texts via 'text' vectorizer")
        self.setup()
        self._label_encoder = LabelEncoder()
        y_arr = list(y)
        labels = torch.tensor(self._label_encoder.fit_transform(y_arr), dtype=torch.long)
        num_labels = int(len(set(y_arr)))
        try:
            hidden = int(getattr(self._model.config, "hidden_size"))
            new_head = torch.nn.Linear(hidden, num_labels)
            new_head.to(self.device)
            setattr(self._model, "classifier", new_head)
        except Exception:
            pass
        enc = self._tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        enc = {k: v.to(self.device) for k, v in enc.items()}
        optimizer = torch.optim.AdamW(self._model.parameters(), lr=self.lr)
        self._model.train()
        for _ in range(self.epochs):
            optimizer.zero_grad()
            outputs = self._model(**enc, labels=labels.to(self.device))
            loss = outputs.loss
            loss.backward()
            optimizer.step()
        if self.device == "mps":
            torch.mps.empty_cache()

    def fit_with_validation(self, x_train: Any, y_train: Sequence[Any], x_val: Any, y_val: Sequence[Any], target_acc: float | None = None) -> None:
        train_texts = list(x_train)
        val_texts = list(x_val)
        if not train_texts or not isinstance(train_texts[0], str):
            raise TypeError("bert classifier expects raw texts via 'text' vectorizer")
        self.setup()
        union_labels = list(y_train) + list(y_val)
        self._label_encoder = LabelEncoder()
        self._label_encoder.fit(union_labels)
        train_labels = torch.tensor(self._label_encoder.transform(list(y_train)), dtype=torch.long)
        val_labels_np = np.asarray(self._label_encoder.transform(list(y_val)), dtype=int)
        num_labels = int(len(set(union_labels)))
        try:
            hidden = int(getattr(self._model.config, "hidden_size"))
            new_head = torch.nn.Linear(hidden, num_labels)
            new_head.to(self.device)
            setattr(self._model, "classifier", new_head)
        except Exception:
            pass
        train_enc = self._tokenizer(train_texts, padding=True, truncation=True, return_tensors="pt")
        val_enc = self._tokenizer(val_texts, padding=True, truncation=True, return_tensors="pt")
        train_enc = {k: v.to(self.device) for k, v in train_enc.items()}
        val_enc = {k: v.to(self.device) for k, v in val_enc.items()}
        optimizer = torch.optim.AdamW(self._model.parameters(), lr=self.lr)
        best_acc = -1.0
        best_state: Optional[dict] = None
        self._model.train()
        train_labels_list = self._label_encoder.transform(list(y_train))
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            outputs = self._model(**train_enc, labels=train_labels.to(self.device))
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            self._model.eval()
            with torch.no_grad():
                train_logits = self._model(**train_enc).logits
                train_preds = train_logits.argmax(dim=-1).cpu().numpy()
                train_acc = float(np.mean(train_preds == train_labels_list))
                val_logits = self._model(**val_enc).logits
                val_preds = val_logits.argmax(dim=-1).cpu().numpy()
                val_acc = float(np.mean(val_preds == val_labels_np))
            print(f"Epoch {epoch + 1}/{self.epochs} - loss: {loss.item():.4f} - train_acc: {train_acc:.4f} - val_acc: {val_acc:.4f}")
            if val_acc > best_acc:
                best_acc = val_acc
                if self.checkpoint_dir:
                    try:
                        self._model.save_pretrained(self.checkpoint_dir)
                        if self._tokenizer is not None:
                            self._tokenizer.save_pretrained(self.checkpoint_dir)
                    except Exception:
                        pass
                else:
                    best_state = {k: v.detach().cpu().clone() for k, v in self._model.state_dict().items()}
            if (target_acc or self.target_acc) is not None and val_acc >= float(target_acc or self.target_acc):
                print(f"Early stopping: target_acc {target_acc or self.target_acc} reached")
                break
            self._model.train()
        if self.device == "mps":
            torch.mps.empty_cache()
        if self.checkpoint_dir:
            try:
                self._model = AutoModelForSequenceClassification.from_pretrained(self.checkpoint_dir)
                self._model.to(self.device)
            except Exception:
                pass
        elif best_state is not None:
            self._model.load_state_dict({k: v.to(self.device) for k, v in best_state.items()})

    def predict(self, x: Any) -> Sequence[Any]:
        if self._model is None or self._tokenizer is None:
            raise RuntimeError("bert classifier not fitted")
        texts = list(x)
        enc = self._tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        enc = {k: v.to(self.device) for k, v in enc.items()}
        self._model.eval()
        with torch.no_grad():
            logits = self._model(**enc).logits
            preds = logits.argmax(dim=-1).cpu().numpy()
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
        if self._model is not None:
            self._model.cpu()
            del self._model
        if self._tokenizer is not None:
            del self._tokenizer
        if self._label_encoder is not None:
            del self._label_encoder
        self._model = None
        self._tokenizer = None
        self._label_encoder = None
        import gc
        gc.collect()
        if self.device == "cuda":
            torch.cuda.empty_cache()
        elif self.device == "mps":
            torch.mps.empty_cache()

class BertRegressorHead(SupervisedDownstreamHead):
    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        batch_size: int = 8,
        epochs: int = 2,
        lr: float = 2e-5,
        device: Optional[str] = None,
        primary_metric: str = "r2",
    ):
        super().__init__(name="bert_regressor", primary_metric=primary_metric)
        self.model_name = model_name
        self.batch_size = int(batch_size)
        self.epochs = int(epochs)
        self.lr = float(lr)
        self.device = _resolve_device(device)
        self._tokenizer: Optional[AutoTokenizer] = None
        self._model: Optional[AutoModelForSequenceClassification] = None

    def setup(self) -> None:
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(self.model_name, num_labels=1)
        self._model.config.problem_type = "regression"
        self._model.to(self.device)

    def fit(self, x: Any, y: Sequence[Any]) -> None:
        texts = list(x)
        if not texts or not isinstance(texts[0], str):
            raise TypeError("bert regressor expects raw texts via 'text' vectorizer")
        self.setup()
        labels = torch.tensor(np.asarray(list(y), dtype=float), dtype=torch.float)
        enc = self._tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        enc = {k: v.to(self.device) for k, v in enc.items()}
        optimizer = torch.optim.AdamW(self._model.parameters(), lr=self.lr)
        self._model.train()
        for _ in range(self.epochs):
            optimizer.zero_grad()
            outputs = self._model(**enc, labels=labels.to(self.device))
            loss = outputs.loss
            loss.backward()
            optimizer.step()
        if self.device == "mps":
            torch.mps.empty_cache()

    def predict(self, x: Any) -> Sequence[float]:
        if self._model is None or self._tokenizer is None:
            raise RuntimeError("bert regressor not fitted")
        texts = list(x)
        enc = self._tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        enc = {k: v.to(self.device) for k, v in enc.items()}
        self._model.eval()
        with torch.no_grad():
            logits = self._model(**enc).logits
            preds = logits.squeeze(-1).cpu().numpy()
        return np.asarray(preds, dtype=float)

    def evaluate(self, x: Any, y: Sequence[Any]) -> Dict[str, float]:
        preds = self.predict(x)
        mse = float(mean_squared_error(y, preds))
        r2 = float(r2_score(y, preds))
        return {"rmse": float(np.sqrt(mse)), "r2": r2}

    def cleanup(self) -> None:
        if self._model is not None:
            self._model.cpu()
            del self._model
        if self._tokenizer is not None:
            del self._tokenizer
        self._model = None
        self._tokenizer = None
        import gc
        gc.collect()
        if self.device == "cuda":
            torch.cuda.empty_cache()
        elif self.device == "mps":
            torch.mps.empty_cache()

DOWNSTREAM_CLASSIFIER_HEAD_REGISTRY: Dict[str, type[SupervisedDownstreamHead]] = {
    "logistic_classifier": LogisticClassifier,
    "feedforward_classifier": FeedForwardClassifier,
    "bert_classifier": BertClassifierHead,
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

