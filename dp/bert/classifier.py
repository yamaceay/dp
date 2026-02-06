from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set

import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder
from transformers import AutoModel, AutoTokenizer, EvalPrediction, Trainer, TrainingArguments

from dp.bert.base import SupervisedDownstreamHead
from dp.bert.common import (
    EarlyStoppingCallback,
    build_optimizer_and_scheduler,
    mask_stopword_tokens,
    pretrain_backbone_with_mlm,
)
from dp.bert.losses import compute_focal_loss
from dp.utils.device import resolve_device


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
        pretraining_output_dir: Optional[str] = None,
        mask_stopwords: bool = False,
        macro_loss_weight: float = 0.0,
        loss_type: str = "cross_entropy",
        focal_gamma: float = 2.0,
        focal_alpha: Optional[float] = None,
        focal_ignore_pt: Optional[float] = None,
        use_pretraining: bool = False,
        pretraining_epochs: int = 1,
        pretraining_batch_size: Optional[int] = None,
        pretraining_learning_rate: float = 5e-5,
        pretraining_mlm_probability: float = 0.15,
        optimizer_type: str = "adamw",
        scheduler_type: str = "linear",
        weight_decay: float = 0.01,
        warmup_ratio: Optional[float] = None,
    ):
        super().__init__(name="bert_classifier", primary_metric=primary_metric)
        if loss_type not in {"cross_entropy", "focal"}:
            raise ValueError(f"Unknown loss_type: {loss_type}")
        if focal_gamma < 0:
            raise ValueError(f"focal_gamma must be >= 0, got {focal_gamma}")
        if focal_ignore_pt is not None and (focal_ignore_pt <= 0.0 or focal_ignore_pt >= 1.0):
            raise ValueError(f"focal_ignore_pt must be in (0, 1), got {focal_ignore_pt}")
        if pretraining_epochs <= 0:
            raise ValueError(f"pretraining_epochs must be positive, got {pretraining_epochs}")
        if pretraining_batch_size is not None and pretraining_batch_size <= 0:
            raise ValueError(f"pretraining_batch_size must be positive, got {pretraining_batch_size}")
        if pretraining_learning_rate <= 0:
            raise ValueError(f"pretraining_learning_rate must be positive, got {pretraining_learning_rate}")
        if pretraining_mlm_probability <= 0.0 or pretraining_mlm_probability >= 1.0:
            raise ValueError(f"pretraining_mlm_probability must be in (0, 1), got {pretraining_mlm_probability}")
        if weight_decay < 0:
            raise ValueError(f"weight_decay must be >= 0, got {weight_decay}")
        if warmup_ratio is not None and (warmup_ratio < 0.0 or warmup_ratio >= 1.0):
            raise ValueError(f"warmup_ratio must be in [0, 1), got {warmup_ratio}")
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
        self.pretraining_output_dir = pretraining_output_dir or self.checkpoint_dir
        self.mask_stopwords = bool(mask_stopwords)
        self.macro_loss_weight = float(macro_loss_weight)
        self.loss_type = str(loss_type)
        self.focal_gamma = float(focal_gamma)
        self.focal_alpha = float(focal_alpha) if focal_alpha is not None else None
        self.focal_ignore_pt = float(focal_ignore_pt) if focal_ignore_pt is not None else None
        self.use_pretraining = bool(use_pretraining)
        self.pretraining_epochs = int(pretraining_epochs)
        self.pretraining_batch_size = int(pretraining_batch_size) if pretraining_batch_size is not None else self.batch_size
        self.pretraining_learning_rate = float(pretraining_learning_rate)
        self.pretraining_mlm_probability = float(pretraining_mlm_probability)
        self.optimizer_type = str(optimizer_type)
        self.scheduler_type = str(scheduler_type)
        self.weight_decay = float(weight_decay)
        self.warmup_ratio = float(warmup_ratio) if warmup_ratio is not None else None
        self._tokenizer: Optional[AutoTokenizer] = None
        self._model: Optional[torch.nn.Module] = None
        self._trainer: Optional[Trainer] = None
        self._label_list: Optional[List[str]] = None
        self._label_to_id: Optional[Dict[str, int]] = None
        self._id_to_label: Optional[Dict[int, str]] = None
        self._stopwords: Set[str] = set()
        self._active_checkpoint: Optional[str] = None

    def _create_model(self, num_labels: int):
        base_model = AutoModel.from_pretrained(self.model_name)
        checkpoint_source = self._active_checkpoint or self.init_checkpoint
        if checkpoint_source:
            try:
                checkpoint_model = AutoModel.from_pretrained(checkpoint_source)
                base_model.load_state_dict(checkpoint_model.state_dict(), strict=False)
            except Exception:
                pass

        hidden_size = base_model.config.hidden_size
        loss_type = self.loss_type
        label_smoothing = self.label_smoothing
        macro_loss_weight = self.macro_loss_weight
        focal_gamma = self.focal_gamma
        focal_alpha = self.focal_alpha
        focal_ignore_pt = self.focal_ignore_pt

        class ClassifierModel(torch.nn.Module):
            def __init__(self, base, hidden_size, num_labels):
                super().__init__()
                self.base_model = base
                self._keys_to_ignore_on_save = []
                self.classifier = torch.nn.Linear(hidden_size, num_labels)

            def forward(self, input_ids, attention_mask, labels=None):
                outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
                hidden = outputs.last_hidden_state[:, 0, :]
                logits = self.classifier(hidden)

                loss = None
                if labels is not None:
                    if loss_type == "cross_entropy":
                        base_loss = torch.nn.functional.cross_entropy(
                            logits,
                            labels,
                            label_smoothing=label_smoothing,
                            reduction="none",
                        )
                        base_loss_mean = base_loss.mean()
                        if macro_loss_weight > 0:
                            classes = torch.unique(labels)
                            per_class = []
                            for cls in classes:
                                mask = labels == cls
                                if mask.any():
                                    per_class.append(base_loss[mask].mean())
                            macro_loss = torch.stack(per_class).mean() if per_class else base_loss_mean
                            loss = base_loss_mean + macro_loss_weight * macro_loss
                        else:
                            loss = base_loss_mean
                    elif loss_type == "focal":
                        base_loss, mask = compute_focal_loss(
                            logits,
                            labels,
                            gamma=focal_gamma,
                            alpha=focal_alpha,
                            ignore_pt=focal_ignore_pt,
                            reduction="none",
                            return_mask=True,
                        )
                        if focal_ignore_pt is not None and mask is not None:
                            denom = mask.sum()
                            base_loss_mean = (base_loss.sum() / denom) if denom > 0 else base_loss.mean() * 0.0
                        else:
                            base_loss_mean = base_loss.mean()
                        if macro_loss_weight > 0:
                            classes = torch.unique(labels)
                            per_class = []
                            for cls in classes:
                                cls_mask = labels == cls
                                if not cls_mask.any():
                                    continue
                                cls_loss = base_loss[cls_mask]
                                if focal_ignore_pt is not None and mask is not None:
                                    cls_mask_weight = mask[cls_mask]
                                    denom = cls_mask_weight.sum()
                                    if denom > 0:
                                        per_class.append(cls_loss.sum() / denom)
                                else:
                                    per_class.append(cls_loss.mean())
                            macro_loss = torch.stack(per_class).mean() if per_class else base_loss_mean
                            loss = base_loss_mean + macro_loss_weight * macro_loss
                        else:
                            loss = base_loss_mean
                    else:
                        raise ValueError(f"Unknown loss_type: {loss_type}")

                return {"loss": loss, "logits": logits}

        model = ClassifierModel(base_model, hidden_size, num_labels)
        return model

    def fit(self, x_train: Any, y_train: Sequence[Any], x_val: Any, y_val: Sequence[Any]) -> None:
        from collections import Counter

        train_texts = list(x_train)
        val_texts = list(x_val)
        train_labels = list(y_train)
        val_labels = list(y_val)
        union_labels = train_labels + val_labels
        self._active_checkpoint = None

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

        if self.use_pretraining:
            self._active_checkpoint = pretrain_backbone_with_mlm(
                model_name=self.model_name,
                texts=train_texts,
                output_dir=self.pretraining_output_dir,
                init_checkpoint=self.init_checkpoint,
                epochs=self.pretraining_epochs,
                batch_size=self.pretraining_batch_size,
                learning_rate=self.pretraining_learning_rate,
                mlm_probability=self.pretraining_mlm_probability,
            )
        tokenizer_source = self._active_checkpoint or self.init_checkpoint or self.model_name
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
        except Exception:
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        if self.mask_stopwords:
            from dp.utils.stopwords import DEFAULT_STOPWORDS
            self._stopwords = DEFAULT_STOPWORDS
            print(f"Stopword masking enabled: {len(self._stopwords)} stopwords will be masked")

        self._model = self._create_model(len(self._label_list))

        train_encodings = self._tokenizer(train_texts, padding=True, truncation=True, return_tensors="pt")
        val_encodings = self._tokenizer(val_texts, padding=True, truncation=True, return_tensors="pt")

        if self.mask_stopwords:
            train_encodings = mask_stopword_tokens(self._tokenizer, self._stopwords, train_encodings, train_texts)
            val_encodings = mask_stopword_tokens(self._tokenizer, self._stopwords, val_encodings, val_texts)

        class ClassifierDataset(torch.utils.data.Dataset):
            def __init__(self, encodings, labels):
                self.encodings = encodings
                self.labels = labels

            def __len__(self):
                return len(self.labels)

            def __getitem__(self, idx):
                item = {key: val[idx] for key, val in self.encodings.items()}
                item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
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
            weight_decay=self.weight_decay,
            logging_steps=10,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="macro_f1",
            greater_is_better=True,
            max_grad_norm=self.gradient_clip,
            report_to="none",
        )
        steps_per_epoch = max(1, int(np.ceil(len(train_dataset) / self.batch_size)))
        num_training_steps = steps_per_epoch * self.epochs
        optimizer, scheduler = build_optimizer_and_scheduler(
            self._model,
            optimizer_type=self.optimizer_type,
            scheduler_type=self.scheduler_type,
            encoder_lr=self.encoder_lr,
            head_lr=self.head_lr,
            weight_decay=self.weight_decay,
            warmup_steps=self.warmup_steps,
            warmup_ratio=self.warmup_ratio,
            num_training_steps=num_training_steps,
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
            optimizers=(optimizer, scheduler),
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
                    encodings = mask_stopword_tokens(self._tokenizer, self._stopwords, encodings, batch_texts)
                encodings = {k: v.to(self._model_device()) for k, v in encodings.items()}
                outputs = self._model(**encodings)
                logits = outputs["logits"]
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                all_preds.extend([self._id_to_label[int(p)] for p in preds])
        return np.array(all_preds)

    def predict_proba(self, x: Any) -> List[Dict[str, float]]:
        if self._model is None or self._tokenizer is None:
            raise RuntimeError("Model not fitted")
        if self._id_to_label is None:
            raise RuntimeError("Label mapping is not initialized")
        texts = list(x)
        self._model.eval()
        all_scores: List[Dict[str, float]] = []
        with torch.no_grad():
            for i in range(0, len(texts), self.batch_size):
                batch_texts = texts[i:i + self.batch_size]
                encodings = self._tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt")
                if self.mask_stopwords:
                    encodings = mask_stopword_tokens(self._tokenizer, self._stopwords, encodings, batch_texts)
                encodings = {k: v.to(self._model_device()) for k, v in encodings.items()}
                outputs = self._model(**encodings)
                probs = torch.nn.functional.softmax(outputs["logits"], dim=1).cpu().numpy()
                for row in probs:
                    row_map: Dict[str, float] = {}
                    for idx, value in enumerate(row.tolist()):
                        row_map[self._id_to_label[idx]] = float(value)
                    all_scores.append(row_map)
        return all_scores

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

    def save(self, model_dir: str) -> None:
        if self._model is None or self._tokenizer is None:
            raise RuntimeError("Model not fitted")
        if self._label_list is None:
            raise RuntimeError("Label mapping is not initialized")
        output_dir = Path(model_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = output_dir / "bert_classifier.pt"
        payload = {
            "model_state_dict": self._model.state_dict(),
            "label_list": self._label_list,
            "model_name": self.model_name,
            "loss_type": self.loss_type,
            "focal_gamma": self.focal_gamma,
            "focal_alpha": self.focal_alpha,
            "focal_ignore_pt": self.focal_ignore_pt,
            "mask_stopwords": self.mask_stopwords,
        }
        torch.save(payload, checkpoint_path)
        self._tokenizer.save_pretrained(str(output_dir))
        with (output_dir / "label_mapping.json").open("w", encoding="utf-8") as f:
            json.dump({label: idx for idx, label in enumerate(self._label_list)}, f, indent=2)

    def load(self, model_dir: str) -> None:
        checkpoint_path = Path(model_dir) / "bert_classifier.pt"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Bert classifier checkpoint not found: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        checkpoint_model_name = checkpoint.get("model_name")
        if isinstance(checkpoint_model_name, str) and checkpoint_model_name:
            self.model_name = checkpoint_model_name
        label_list = checkpoint.get("label_list")
        if not isinstance(label_list, list) or not label_list:
            raise ValueError("Invalid label_list in bert classifier checkpoint")
        self._label_list = [str(label) for label in label_list]
        self._label_to_id = {label: idx for idx, label in enumerate(self._label_list)}
        self._id_to_label = {idx: label for idx, label in enumerate(self._label_list)}
        self._tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self._model = self._create_model(len(self._label_list))
        state_dict = checkpoint.get("model_state_dict")
        if not isinstance(state_dict, dict):
            raise ValueError("Invalid model_state_dict in bert classifier checkpoint")
        self._model.load_state_dict(state_dict)
        self._model.to(self.device)
        self._model.eval()

    def _model_device(self) -> torch.device:
        if self._model is None:
            raise RuntimeError("Model not fitted")
        return next(self._model.parameters()).device

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
