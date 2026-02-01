from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Set

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer, EvalPrediction, Trainer, TrainingArguments

from dp.bert.base import SupervisedDownstreamHead
from dp.bert.common import EarlyStoppingCallback, mask_stopword_tokens
from dp.utils.device import resolve_device


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
        macro_loss_weight = self.macro_loss_weight

        class OrdinalModel(torch.nn.Module):
            def __init__(self, base, hidden_size, num_thresholds):
                super().__init__()
                self.base_model = base
                self._keys_to_ignore_on_save = []
                self.pre_classifier = torch.nn.Linear(hidden_size, hidden_size)
                self.dropout = torch.nn.Dropout(0.1)
                self.weight = torch.nn.Parameter(torch.randn(hidden_size) * 0.01)
                self.biases = torch.nn.Parameter(torch.zeros(num_thresholds))

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
                    if macro_loss_weight > 0:
                        sample_loss = base_loss.mean(dim=1)
                        class_ids = labels.sum(dim=1).long()
                        classes = torch.unique(class_ids)
                        per_class = []
                        for cls in classes:
                            mask = class_ids == cls
                            if mask.any():
                                per_class.append(sample_loss[mask].mean())
                        macro_loss = torch.stack(per_class).mean() if per_class else base_loss_mean
                        loss = base_loss_mean + macro_loss_weight * macro_loss
                    else:
                        loss = base_loss_mean

                return {"loss": loss, "logits": logits}

        model = OrdinalModel(base_model, hidden_size, num_thresholds)
        return model

    def _prepare_targets(self, train_labels, val_labels, union_labels):
        from dp.experiments.utility.initializers import compute_coral_bias_init
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
            train_encodings = mask_stopword_tokens(self._tokenizer, self._stopwords, train_encodings, train_texts)
            val_encodings = mask_stopword_tokens(self._tokenizer, self._stopwords, val_encodings, val_texts)

        class OrdinalDataset(torch.utils.data.Dataset):
            def __init__(self, encodings, labels):
                self.encodings = encodings
                self.labels = labels

            def __len__(self):
                return len(self.labels)

            def __getitem__(self, idx):
                item = {key: val[idx] for key, val in self.encodings.items()}
                item["labels"] = torch.tensor(self.labels[idx], dtype=torch.float)
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

            macro_mae = float(np.mean(per_class_mae)) if per_class_mae else float("inf")
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
                    encodings = mask_stopword_tokens(self._tokenizer, self._stopwords, encodings, batch_texts)
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

        macro_mae = float(np.mean(per_class_mae)) if per_class_mae else float("inf")
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
