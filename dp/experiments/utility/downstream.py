from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Sequence, Set
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import f1_score, mean_squared_error, r2_score
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer, AutoModel, Trainer, TrainingArguments, TrainerCallback, EvalPrediction
import torch

from dp.utils.device import resolve_device

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

class EarlyStoppingCallback(TrainerCallback):
    def __init__(self, early_stopping_patience: int, early_stopping_threshold: Optional[float], metric_name: str, minimize: bool):
        self.patience = early_stopping_patience
        self.threshold = early_stopping_threshold
        self.metric_name = metric_name
        self.minimize = minimize
        self.best_metric = float('inf') if minimize else -float('inf')
        self.wait = 0
        self.stopped_epoch = 0

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        current = metrics.get(f"eval_{self.metric_name}")
        if current is None:
            return
        if self.threshold is not None:
            if (self.minimize and current <= self.threshold) or (not self.minimize and current >= self.threshold):
                control.should_training_stop = True
                return
        improved = (current < self.best_metric) if self.minimize else (current > self.best_metric)
        if improved:
            self.best_metric = current
            self.wait = 0
            control.should_save = True
        else:
            self.wait += 1
            if self.wait >= self.patience:
                control.should_training_stop = True

class BertOrdinalHead(SupervisedDownstreamHead):
    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        batch_size: int = 8,
        epochs: int = 5,
        encoder_lr: float = 1e-5,
        head_lr: float = 5e-5,
        warmup_steps: int = 10,
        gradient_clip: float = 1.0,
        device: Optional[str] = None,
        primary_metric: str = "macro_mae",
        early_stop_threshold: Optional[float] = None,
        early_stop_patience: int = 2,
        init_checkpoint: Optional[str] = None,
        checkpoint_dir: Optional[str] = None,
        mask_stopwords: bool = False,
        optimizer_type: str = "adamw",
        scheduler_type: str = "linear",
        weight_decay: Optional[float] = None,
        warmup_ratio: Optional[float] = None,
        macro_loss_weight: float = 0.0,
    ):
        super().__init__(name="bert_ordinal", primary_metric=primary_metric)
        self.model_name = model_name
        self.batch_size = int(batch_size)
        self.epochs = int(epochs)
        self.encoder_lr = float(encoder_lr)
        self.head_lr = float(head_lr)
        self.warmup_steps = int(warmup_steps)
        self.gradient_clip = float(gradient_clip)
        self.device = resolve_device(device)
        self.early_stop_threshold = float(early_stop_threshold) if early_stop_threshold is not None else None
        self.early_stop_patience = int(early_stop_patience)
        self.init_checkpoint = init_checkpoint
        self.checkpoint_dir = checkpoint_dir or "tmp_hf_checkpoint"
        self.mask_stopwords = bool(mask_stopwords)
        self.optimizer_type = str(optimizer_type)
        self.scheduler_type = str(scheduler_type)
        self.weight_decay = float(weight_decay) if weight_decay is not None else None
        self.warmup_ratio = float(warmup_ratio) if warmup_ratio is not None else None
        self.macro_loss_weight = float(macro_loss_weight)
        self._tokenizer: Optional[AutoTokenizer] = None
        self._model: Optional[torch.nn.Module] = None
        self._trainer: Optional[Trainer] = None
        self._label_order: Optional[List[str]] = None
        self._label_to_index: Optional[Dict[str, int]] = None
        self._index_to_label: Optional[Dict[int, str]] = None
        self._num_classes: int = 0
        self._stopwords: Set[str] = set()

    def _mask_stopword_tokens(self, encodings, texts):
        for idx, text in enumerate(texts):
            tokens = self._tokenizer.tokenize(text)
            token_ids = encodings["input_ids"][idx]
            attention_mask = encodings["attention_mask"][idx]
            
            for token_pos in range(len(token_ids)):
                token_id = token_ids[token_pos].item() if hasattr(token_ids[token_pos], 'item') else token_ids[token_pos]
                token_str = self._tokenizer.convert_ids_to_tokens(token_id)
                normalized_token = token_str.lower().strip("#").replace("##", "")
                
                if normalized_token in self._stopwords:
                    attention_mask[token_pos] = 0
        return encodings

    def _create_model(self, num_classes: int):
        base_model = AutoModel.from_pretrained(self.model_name)
        if self.init_checkpoint:
            try:
                checkpoint_model = AutoModel.from_pretrained(self.init_checkpoint)
                base_model.load_state_dict(checkpoint_model.state_dict(), strict=False)
            except Exception:
                pass
        
        num_thresholds = num_classes - 1
        hidden_size = base_model.config.hidden_size
        
        class OrdinalModel(torch.nn.Module):
            def __init__(self, base, hidden_size, num_thresholds, macro_loss_weight):
                super().__init__()
                self.base_model = base
                self.pre_classifier = torch.nn.Linear(hidden_size, hidden_size)
                self.dropout = torch.nn.Dropout(0.1)
                self.weight = torch.nn.Parameter(torch.randn(hidden_size) * 0.01)
                self.biases = torch.nn.Parameter(torch.zeros(num_thresholds))
                self.macro_loss_weight = macro_loss_weight
            
            def forward(self, input_ids, attention_mask, labels=None):
                outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
                hidden = outputs.last_hidden_state[:, 0, :]
                hidden = self.pre_classifier(hidden)
                hidden = torch.nn.functional.relu(hidden)
                hidden = self.dropout(hidden)
                logits = torch.matmul(hidden, self.weight.unsqueeze(1))
                logits = logits + self.biases.unsqueeze(0)
                
                loss = None
                if labels is not None:
                    base_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                        logits,
                        labels,
                        reduction="none",
                    )
                    base_loss_mean = base_loss.mean()
                    if self.macro_loss_weight > 0:
                        sample_loss = base_loss.mean(dim=1)
                        class_ids = labels.sum(dim=1).long()
                        classes = torch.unique(class_ids)
                        per_class = []
                        for cls in classes:
                            mask = class_ids == cls
                            if mask.any():
                                per_class.append(sample_loss[mask].mean())
                        macro_loss = torch.stack(per_class).mean() if per_class else base_loss_mean
                        loss = base_loss_mean + self.macro_loss_weight * macro_loss
                    else:
                        loss = base_loss_mean
                
                return {"loss": loss, "logits": logits}
        
        model = OrdinalModel(base_model, hidden_size, num_thresholds, self.macro_loss_weight)
        return model

    def _prepare_targets(self, train_labels, val_labels, union_labels):
        from .initializers import compute_coral_bias_init
        from collections import Counter
        if self._label_order is None:
            unique_labels = sorted(set(union_labels), key=str)
            self._label_order = [str(label) for label in unique_labels]
        self._label_to_index = {label: idx for idx, label in enumerate(self._label_order)}
        self._index_to_label = {idx: label for idx, label in enumerate(self._label_order)}
        train_encoded = np.array([self._label_to_index[str(label)] for label in train_labels])
        val_encoded = np.array([self._label_to_index[str(label)] for label in val_labels])
        num_classes = len(self._label_order)
        num_thresholds = num_classes - 1
        
        bias_init = compute_coral_bias_init(train_encoded, num_classes)
        print(f"CORAL bias init: {bias_init}")
        train_dist = Counter(train_labels)
        val_dist = Counter(val_labels)
        print(f"Training on {len(train_labels)} samples: {dict(train_dist)}")
        print(f"Validating on {len(val_labels)} samples: {dict(val_dist)}")
        
        def encode_cumulative(labels_encoded):
            cumulative = np.zeros((len(labels_encoded), num_thresholds), dtype=np.float32)
            for i, label in enumerate(labels_encoded):
                for t in range(num_thresholds):
                    if label > t:
                        cumulative[i, t] = 1.0
            return cumulative
        
        train_targets = encode_cumulative(train_encoded)
        val_targets = encode_cumulative(val_encoded)
        
        return train_targets, val_targets, num_classes, bias_init

    def fit(self, x_train: Any, y_train: Sequence[Any], x_val: Any, y_val: Sequence[Any]) -> None:
        train_texts = list(x_train)
        val_texts = list(x_val)
        train_labels = list(y_train)
        val_labels = list(y_val)
        union_labels = train_labels + val_labels
        
        train_targets, val_targets, num_classes, bias_init = self._prepare_targets(train_labels, val_labels, union_labels)
        self._num_classes = num_classes
        
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        if self.mask_stopwords:
            from dp.utils.stopwords import DEFAULT_STOPWORDS
            self._stopwords = DEFAULT_STOPWORDS
            print(f"Stopword masking enabled: {len(self._stopwords)} stopwords will be masked")
        
        self._model = self._create_model(num_classes)
        
        with torch.no_grad():
            self._model.biases.data = torch.from_numpy(bias_init).float()
        
        train_encodings = self._tokenizer(train_texts, padding=True, truncation=True, return_tensors="pt")
        val_encodings = self._tokenizer(val_texts, padding=True, truncation=True, return_tensors="pt")
        
        if self.mask_stopwords:
            train_encodings = self._mask_stopword_tokens(train_encodings, train_texts)
            val_encodings = self._mask_stopword_tokens(val_encodings, val_texts)
        
        class OrdinalDataset(torch.utils.data.Dataset):
            def __init__(self, encodings, labels):
                self.encodings = encodings
                self.labels = labels
            
            def __len__(self):
                return len(self.labels)
            
            def __getitem__(self, idx):
                item = {key: val[idx] for key, val in self.encodings.items()}
                item['labels'] = torch.tensor(self.labels[idx], dtype=torch.float)
                return item
        
        train_dataset = OrdinalDataset(train_encodings, train_targets)
        val_dataset = OrdinalDataset(val_encodings, val_targets)
        
        def compute_metrics(eval_pred: EvalPrediction):
            logits = eval_pred.predictions
            labels = eval_pred.label_ids
            probs = 1 / (1 + np.exp(-logits))
            class_preds = (probs > 0.5).sum(axis=1).astype(int)
            label_classes = (labels > 0.5).sum(axis=1).astype(int)
            
            unique_classes = sorted(self._label_to_index.values())
            per_class_mae = []
            for cls in unique_classes:
                mask = label_classes == cls
                if mask.sum() > 0:
                    cls_mae = np.abs(label_classes[mask] - class_preds[mask]).mean()
                    per_class_mae.append(cls_mae)
            
            macro_mae = float(np.mean(per_class_mae)) if per_class_mae else float('inf')
            overall_mae = float(np.mean(np.abs(label_classes - class_preds)))
            
            return {
                "macro_mae": macro_mae,
                "mae": overall_mae,
            }
        
        training_args = TrainingArguments(
            output_dir=self.checkpoint_dir,
            num_train_epochs=self.epochs,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            warmup_steps=self.warmup_steps,
            learning_rate=self.head_lr,
            weight_decay=0.01,
            logging_steps=10,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="macro_mae",
            greater_is_better=False,
            max_grad_norm=self.gradient_clip,
            report_to="none",
        )
        
        early_stopping = EarlyStoppingCallback(
            early_stopping_patience=self.early_stop_patience,
            early_stopping_threshold=self.early_stop_threshold,
            metric_name="macro_mae",
            minimize=True,
        )
        
        self._trainer = Trainer(
            model=self._model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
            callbacks=[early_stopping],
        )
        
        self._trainer.train()
        print(f"Restored best model with macro_mae: {early_stopping.best_metric:.4f}")

    def predict(self, x: Any) -> Sequence[Any]:
        if self._model is None or self._tokenizer is None:
            raise RuntimeError("Model not fitted")
        texts = list(x)
        self._model.eval()
        all_preds = []
        with torch.no_grad():
            for i in range(0, len(texts), self.batch_size):
                batch_texts = texts[i:i + self.batch_size]
                encodings = self._tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt")
                if self.mask_stopwords:
                    encodings = self._mask_stopword_tokens(encodings, batch_texts)
                encodings = {k: v.to(self._model.base_model.device) for k, v in encodings.items()}
                outputs = self._model(**encodings)
                logits = outputs["logits"]
                probs = torch.sigmoid(logits)
                class_preds = (probs > 0.5).sum(dim=1).cpu().numpy().astype(int)
                if self._index_to_label is not None:
                    preds = [self._index_to_label[idx] for idx in class_preds]
                else:
                    preds = class_preds
                all_preds.extend(preds)
        return np.array(all_preds)

    def evaluate(self, x: Any, y: Sequence[Any]) -> Dict[str, float]:
        predictions = self.predict(x)
        if self._label_to_index is not None:
            y_encoded = np.array([self._label_to_index[str(label)] for label in y])
            pred_encoded = np.array([self._label_to_index[str(label)] for label in predictions])
            unique_classes = sorted(self._label_to_index.values())
        else:
            y_encoded = np.array(y, dtype=int)
            pred_encoded = np.array(predictions, dtype=int)
            unique_classes = sorted(set(y_encoded))
        
        per_class_mae = []
        per_class_recall = []
        per_class_within1 = []
        for cls in unique_classes:
            mask = y_encoded == cls
            if mask.sum() > 0:
                cls_mae = np.abs(y_encoded[mask] - pred_encoded[mask]).mean()
                cls_recall = (y_encoded[mask] == pred_encoded[mask]).mean()
                cls_within1 = (np.abs(y_encoded[mask] - pred_encoded[mask]) <= 1).mean()
                per_class_mae.append(cls_mae)
                per_class_recall.append(cls_recall)
                per_class_within1.append(cls_within1)
        
        macro_mae = float(np.mean(per_class_mae)) if per_class_mae else float('inf')
        macro_within1 = float(np.mean(per_class_within1)) if per_class_within1 else 0.0
        worst_recall = float(np.min(per_class_recall)) if per_class_recall else 0.0
        overall_mae = float(np.mean(np.abs(y_encoded - pred_encoded)))
        overall_acc = float(np.mean(y_encoded == pred_encoded))
        overall_within1 = float(np.mean(np.abs(y_encoded - pred_encoded) <= 1))
        
        return {
            "macro_mae": macro_mae,
            "macro_within1": macro_within1,
            "worst_recall": worst_recall,
            "mae": overall_mae,
            "acc": overall_acc,
            "within1": overall_within1,
        }

    def set_label_order(self, label_order: List[str]) -> None:
        self._label_order = list(label_order)

    def setup(self) -> None:
        pass

    def cleanup(self) -> None:
        if self._model is not None:
            del self._model
        if self._tokenizer is not None:
            del self._tokenizer
        if self._trainer is not None:
            del self._trainer
        self._model = None
        self._tokenizer = None
        self._trainer = None
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

class BertClassifierHead(SupervisedDownstreamHead):
    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        batch_size: int = 8,
        epochs: int = 5,
        encoder_lr: float = 1e-5,
        head_lr: float = 5e-5,
        warmup_steps: int = 10,
        gradient_clip: float = 1.0,
        label_smoothing: float = 0.0,
        device: Optional[str] = None,
        primary_metric: str = "macro_f1",
        early_stop_threshold: Optional[float] = None,
        early_stop_patience: int = 2,
        init_checkpoint: Optional[str] = None,
        checkpoint_dir: Optional[str] = None,
        mask_stopwords: bool = False,
        macro_loss_weight: float = 0.0,
    ):
        super().__init__(name="bert_classifier", primary_metric=primary_metric)
        self.model_name = model_name
        self.batch_size = int(batch_size)
        self.epochs = int(epochs)
        self.encoder_lr = float(encoder_lr)
        self.head_lr = float(head_lr)
        self.warmup_steps = int(warmup_steps)
        self.gradient_clip = float(gradient_clip)
        self.label_smoothing = float(label_smoothing)
        self.device = resolve_device(device)
        self.early_stop_threshold = float(early_stop_threshold) if early_stop_threshold is not None else None
        self.early_stop_patience = int(early_stop_patience)
        self.init_checkpoint = init_checkpoint
        self.checkpoint_dir = checkpoint_dir or "tmp_hf_checkpoint"
        self.mask_stopwords = bool(mask_stopwords)
        self.macro_loss_weight = float(macro_loss_weight)
        self._tokenizer: Optional[AutoTokenizer] = None
        self._model: Optional[torch.nn.Module] = None
        self._trainer: Optional[Trainer] = None
        self._label_list: Optional[List[str]] = None
        self._label_to_id: Optional[Dict[str, int]] = None
        self._id_to_label: Optional[Dict[int, str]] = None
        self._stopwords: Set[str] = set()

    def _mask_stopword_tokens(self, encodings, texts):
        for idx, text in enumerate(texts):
            tokens = self._tokenizer.tokenize(text)
            token_ids = encodings["input_ids"][idx]
            attention_mask = encodings["attention_mask"][idx]
            
            for token_pos in range(len(token_ids)):
                token_id = token_ids[token_pos].item() if hasattr(token_ids[token_pos], 'item') else token_ids[token_pos]
                token_str = self._tokenizer.convert_ids_to_tokens(token_id)
                normalized_token = token_str.lower().strip("#").replace("##", "")
                
                if normalized_token in self._stopwords:
                    attention_mask[token_pos] = 0
        return encodings

    def _create_model(self, num_labels: int):
        base_model = AutoModel.from_pretrained(self.model_name)
        if self.init_checkpoint:
            try:
                checkpoint_model = AutoModel.from_pretrained(self.init_checkpoint)
                base_model.load_state_dict(checkpoint_model.state_dict(), strict=False)
            except Exception:
                pass
        
        hidden_size = base_model.config.hidden_size
        
        class ClassifierModel(torch.nn.Module):
            def __init__(self, base, hidden_size, num_labels, label_smoothing, macro_loss_weight):
                super().__init__()
                self.base_model = base
                self.classifier = torch.nn.Linear(hidden_size, num_labels)
                self.label_smoothing = label_smoothing
                self.macro_loss_weight = macro_loss_weight
            
            def forward(self, input_ids, attention_mask, labels=None):
                outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
                hidden = outputs.last_hidden_state[:, 0, :]
                logits = self.classifier(hidden)
                
                loss = None
                if labels is not None:
                    base_loss = torch.nn.functional.cross_entropy(
                        logits,
                        labels,
                        label_smoothing=self.label_smoothing,
                        reduction="none",
                    )
                    base_loss_mean = base_loss.mean()
                    if self.macro_loss_weight > 0:
                        classes = torch.unique(labels)
                        per_class = []
                        for cls in classes:
                            mask = labels == cls
                            if mask.any():
                                per_class.append(base_loss[mask].mean())
                        macro_loss = torch.stack(per_class).mean() if per_class else base_loss_mean
                        loss = base_loss_mean + self.macro_loss_weight * macro_loss
                    else:
                        loss = base_loss_mean
                
                return {"loss": loss, "logits": logits}
        
        model = ClassifierModel(base_model, hidden_size, num_labels, self.label_smoothing, self.macro_loss_weight)
        return model

    def fit(self, x_train: Any, y_train: Sequence[Any], x_val: Any, y_val: Sequence[Any]) -> None:
        from collections import Counter
        
        train_texts = list(x_train)
        val_texts = list(x_val)
        train_labels = list(y_train)
        val_labels = list(y_val)
        union_labels = train_labels + val_labels
        
        label_encoder = LabelEncoder()
        label_encoder.fit(union_labels)
        self._label_list = label_encoder.classes_.tolist()
        self._label_to_id = {label: idx for idx, label in enumerate(self._label_list)}
        self._id_to_label = {idx: label for idx, label in enumerate(self._label_list)}
        
        train_encoded = label_encoder.transform(train_labels)
        val_encoded = label_encoder.transform(val_labels)
        
        train_dist = Counter(train_labels)
        val_dist = Counter(val_labels)
        print(f"Training on {len(train_labels)} samples: {dict(train_dist)}")
        print(f"Validating on {len(val_labels)} samples: {dict(val_dist)}")
        
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        if self.mask_stopwords:
            from dp.utils.stopwords import DEFAULT_STOPWORDS
            self._stopwords = DEFAULT_STOPWORDS
            print(f"Stopword masking enabled: {len(self._stopwords)} stopwords will be masked")
        
        self._model = self._create_model(len(self._label_list))
        
        train_encodings = self._tokenizer(train_texts, padding=True, truncation=True, return_tensors="pt")
        val_encodings = self._tokenizer(val_texts, padding=True, truncation=True, return_tensors="pt")
        
        if self.mask_stopwords:
            train_encodings = self._mask_stopword_tokens(train_encodings, train_texts)
            val_encodings = self._mask_stopword_tokens(val_encodings, val_texts)
        
        class ClassifierDataset(torch.utils.data.Dataset):
            def __init__(self, encodings, labels):
                self.encodings = encodings
                self.labels = labels
            
            def __len__(self):
                return len(self.labels)
            
            def __getitem__(self, idx):
                item = {key: val[idx] for key, val in self.encodings.items()}
                item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
                return item
        
        train_dataset = ClassifierDataset(train_encodings, train_encoded)
        val_dataset = ClassifierDataset(val_encodings, val_encoded)
        
        def compute_metrics(eval_pred: EvalPrediction):
            from sklearn.metrics import precision_recall_fscore_support
            logits = eval_pred.predictions
            labels = eval_pred.label_ids
            preds = np.argmax(logits, axis=1)
            
            per_class_precision, per_class_recall, per_class_f1, per_class_support = precision_recall_fscore_support(
                labels, preds, average=None, zero_division=0
            )
            valid_mask = per_class_support > 0
            macro_f1 = float(per_class_f1[valid_mask].mean()) if valid_mask.any() else 0.0
            balanced_acc = float(per_class_recall[valid_mask].mean()) if valid_mask.any() else 0.0
            overall_acc = float((preds == labels).mean())
            
            return {
                "macro_f1": macro_f1,
                "balanced_acc": balanced_acc,
                "acc": overall_acc,
            }
        
        training_args = TrainingArguments(
            output_dir=self.checkpoint_dir,
            num_train_epochs=self.epochs,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            warmup_steps=self.warmup_steps,
            learning_rate=self.head_lr,
            weight_decay=0.01,
            logging_steps=10,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="macro_f1",
            greater_is_better=True,
            max_grad_norm=self.gradient_clip,
            report_to="none",
        )
        
        early_stopping = EarlyStoppingCallback(
            early_stopping_patience=self.early_stop_patience,
            early_stopping_threshold=self.early_stop_threshold,
            metric_name="macro_f1",
            minimize=False,
        )
        
        self._trainer = Trainer(
            model=self._model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
            callbacks=[early_stopping],
        )
        
        self._trainer.train()
        print(f"Restored best model with macro_f1: {early_stopping.best_metric:.4f}")

    def predict(self, x: Any) -> Sequence[Any]:
        if self._model is None or self._tokenizer is None:
            raise RuntimeError("Model not fitted")
        texts = list(x)
        self._model.eval()
        all_preds = []
        with torch.no_grad():
            for i in range(0, len(texts), self.batch_size):
                batch_texts = texts[i:i + self.batch_size]
                encodings = self._tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt")
                if self.mask_stopwords:
                    encodings = self._mask_stopword_tokens(encodings, batch_texts)
                encodings = {k: v.to(self._model.base_model.device) for k, v in encodings.items()}
                outputs = self._model(**encodings)
                logits = outputs["logits"]
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                all_preds.extend([self._id_to_label[int(p)] for p in preds])
        return np.array(all_preds)

    def evaluate(self, x: Any, y: Sequence[Any]) -> Dict[str, float]:
        from sklearn.metrics import precision_recall_fscore_support
        predictions = self.predict(x)
        y_arr = np.array(y)
        pred_arr = np.array(predictions)
        unique_labels = sorted(set(y))
        per_class_precision, per_class_recall, per_class_f1, per_class_support = precision_recall_fscore_support(
            y, predictions, labels=unique_labels, average=None, zero_division=0
        )
        valid_mask = per_class_support > 0
        macro_f1 = float(per_class_f1[valid_mask].mean()) if valid_mask.any() else 0.0
        balanced_acc = float(per_class_recall[valid_mask].mean()) if valid_mask.any() else 0.0
        worst_class_recall = float(per_class_recall[valid_mask].min()) if valid_mask.any() else 0.0
        overall_acc = float(np.mean(y_arr == pred_arr))
        return {
            "macro_f1": macro_f1,
            "balanced_acc": balanced_acc,
            "worst_recall": worst_class_recall,
            "acc": overall_acc,
            "f1": macro_f1,
        }

    def setup(self) -> None:
        pass

    def cleanup(self) -> None:
        if self._model is not None:
            del self._model
        if self._tokenizer is not None:
            del self._tokenizer
        if self._trainer is not None:
            del self._trainer
        self._model = None
        self._tokenizer = None
        self._trainer = None
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

class BertRegressorHead(SupervisedDownstreamHead):
    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        batch_size: int = 8,
        epochs: int = 5,
        encoder_lr: float = 1e-5,
        head_lr: float = 5e-5,
        warmup_steps: int = 10,
        gradient_clip: float = 1.0,
        device: Optional[str] = None,
        primary_metric: str = "r2",
        early_stop_threshold: Optional[float] = None,
        early_stop_patience: int = 2,
        init_checkpoint: Optional[str] = None,
        checkpoint_dir: Optional[str] = None,
        normalize_targets: bool = True,
    ):
        super().__init__(name="bert_regressor", primary_metric=primary_metric)
        self.model_name = model_name
        self.batch_size = int(batch_size)
        self.epochs = int(epochs)
        self.encoder_lr = float(encoder_lr)
        self.head_lr = float(head_lr)
        self.warmup_steps = int(warmup_steps)
        self.gradient_clip = float(gradient_clip)
        self.device = resolve_device(device)
        self.early_stop_threshold = float(early_stop_threshold) if early_stop_threshold is not None else None
        self.early_stop_patience = int(early_stop_patience)
        self.init_checkpoint = init_checkpoint
        self.checkpoint_dir = checkpoint_dir or "tmp_hf_checkpoint"
        self.normalize_targets = normalize_targets
        self._tokenizer: Optional[AutoTokenizer] = None
        self._model: Optional[torch.nn.Module] = None
        self._trainer: Optional[Trainer] = None
        self._target_mean: Optional[float] = None
        self._target_std: Optional[float] = None

    def _create_model(self):
        base_model = AutoModel.from_pretrained(self.model_name)
        if self.init_checkpoint:
            try:
                checkpoint_model = AutoModel.from_pretrained(self.init_checkpoint)
                base_model.load_state_dict(checkpoint_model.state_dict(), strict=False)
            except Exception:
                pass
        
        hidden_size = base_model.config.hidden_size
        
        class RegressorModel(torch.nn.Module):
            def __init__(self, base, hidden_size):
                super().__init__()
                self.base_model = base
                self.regressor = torch.nn.Linear(hidden_size, 1)
            
            def forward(self, input_ids, attention_mask, labels=None):
                outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
                hidden = outputs.last_hidden_state[:, 0, :]
                logits = self.regressor(hidden).squeeze(-1)
                
                loss = None
                if labels is not None:
                    loss = torch.nn.functional.smooth_l1_loss(logits, labels)
                
                return {"loss": loss, "logits": logits}
        
        model = RegressorModel(base_model, hidden_size)
        return model

    def fit(self, x_train: Any, y_train: Sequence[Any], x_val: Any, y_val: Sequence[Any]) -> None:
        from collections import Counter
        
        train_texts = list(x_train)
        val_texts = list(x_val)
        train_labels = np.asarray(list(y_train), dtype=float)
        val_labels = np.asarray(list(y_val), dtype=float)
        
        if self.normalize_targets:
            self._target_mean = float(train_labels.mean())
            self._target_std = float(train_labels.std() + 1e-7)
            train_labels_normalized = (train_labels - self._target_mean) / self._target_std
            val_labels_normalized = (val_labels - self._target_mean) / self._target_std
        else:
            train_labels_normalized = train_labels
            val_labels_normalized = val_labels
        
        print(f"Training on {len(train_labels)} samples: mean={train_labels.mean():.2f}, std={train_labels.std():.2f}")
        print(f"Validating on {len(val_labels)} samples: mean={val_labels.mean():.2f}, std={val_labels.std():.2f}")
        
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = self._create_model()
        
        train_encodings = self._tokenizer(train_texts, padding=True, truncation=True, return_tensors="pt")
        val_encodings = self._tokenizer(val_texts, padding=True, truncation=True, return_tensors="pt")
        
        class RegressorDataset(torch.utils.data.Dataset):
            def __init__(self, encodings, labels):
                self.encodings = encodings
                self.labels = labels
            
            def __len__(self):
                return len(self.labels)
            
            def __getitem__(self, idx):
                item = {key: val[idx] for key, val in self.encodings.items()}
                item['labels'] = torch.tensor(self.labels[idx], dtype=torch.float)
                return item
        
        train_dataset = RegressorDataset(train_encodings, train_labels_normalized)
        val_dataset = RegressorDataset(val_encodings, val_labels_normalized)
        
        def compute_metrics(eval_pred: EvalPrediction):
            preds = eval_pred.predictions
            labels = eval_pred.label_ids
            
            if self.normalize_targets and self._target_mean is not None and self._target_std is not None:
                preds = preds * self._target_std + self._target_mean
                labels = labels * self._target_std + self._target_mean
            
            mse = float(np.mean((preds - labels) ** 2))
            rmse = float(np.sqrt(mse))
            r2 = float(1 - (np.sum((labels - preds) ** 2) / np.sum((labels - labels.mean()) ** 2)))
            
            return {
                "rmse": rmse,
                "r2": r2,
            }
        
        training_args = TrainingArguments(
            output_dir=self.checkpoint_dir,
            num_train_epochs=self.epochs,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            warmup_steps=self.warmup_steps,
            learning_rate=self.head_lr,
            weight_decay=0.01,
            logging_steps=10,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="r2",
            greater_is_better=True,
            max_grad_norm=self.gradient_clip,
            report_to="none",
        )
        
        early_stopping = EarlyStoppingCallback(
            early_stopping_patience=self.early_stop_patience,
            early_stopping_threshold=self.early_stop_threshold,
            metric_name="r2",
            minimize=False,
        )
        
        self._trainer = Trainer(
            model=self._model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
            callbacks=[early_stopping],
        )
        
        self._trainer.train()
        print(f"Restored best model with r2: {early_stopping.best_metric:.4f}")

    def predict(self, x: Any) -> Sequence[float]:
        if self._model is None or self._tokenizer is None:
            raise RuntimeError("Model not fitted")
        texts = list(x)
        self._model.eval()
        all_preds = []
        with torch.no_grad():
            for i in range(0, len(texts), self.batch_size):
                batch_texts = texts[i:i + self.batch_size]
                encodings = self._tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt")
                encodings = {k: v.to(self._model.base_model.device) for k, v in encodings.items()}
                outputs = self._model(**encodings)
                preds = outputs["logits"].cpu().numpy()
                if self.normalize_targets and self._target_mean is not None and self._target_std is not None:
                    preds = preds * self._target_std + self._target_mean
                all_preds.extend(preds)
        return np.array(all_preds, dtype=float)

    def evaluate(self, x: Any, y: Sequence[Any]) -> Dict[str, float]:
        predictions = self.predict(x)
        y_arr = np.asarray(y, dtype=float)
        mse = float(np.mean((predictions - y_arr) ** 2))
        rmse = float(np.sqrt(mse))
        r2 = float(1 - (np.sum((y_arr - predictions) ** 2) / np.sum((y_arr - y_arr.mean()) ** 2)))
        return {
            "rmse": rmse,
            "r2": r2,
        }

    def setup(self) -> None:
        pass

    def cleanup(self) -> None:
        if self._model is not None:
            del self._model
        if self._tokenizer is not None:
            del self._tokenizer
        if self._trainer is not None:
            del self._trainer
        self._model = None
        self._tokenizer = None
        self._trainer = None
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

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
