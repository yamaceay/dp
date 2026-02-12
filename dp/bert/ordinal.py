from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
from transformers import EvalPrediction

from dp.bert.base import SupervisedDownstreamHead
from dp.bert.hf_shared import (
    BertHFPlumbing,
    EncodedDataset,
    HFTrainSpec,
    load_backbone_with_optional_checkpoint,
)


class BertOrdinalHead(SupervisedDownstreamHead, BertHFPlumbing):
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
        use_pos_weight: bool = True,
        use_pretraining: bool = False,
        pretraining_epochs: int = 1,
        pretraining_batch_size: Optional[int] = None,
        pretraining_learning_rate: float = 5e-5,
        pretraining_mlm_probability: float = 0.15,
    ):
        SupervisedDownstreamHead.__init__(self, name="bert_ordinal", primary_metric=primary_metric)
        BertHFPlumbing.__init__(self, device=device)
        if pretraining_epochs <= 0:
            raise ValueError(f"pretraining_epochs must be positive, got {pretraining_epochs}")
        if pretraining_batch_size is not None and pretraining_batch_size <= 0:
            raise ValueError(f"pretraining_batch_size must be positive, got {pretraining_batch_size}")
        if pretraining_learning_rate <= 0:
            raise ValueError(f"pretraining_learning_rate must be positive, got {pretraining_learning_rate}")
        if pretraining_mlm_probability <= 0.0 or pretraining_mlm_probability >= 1.0:
            raise ValueError(f"pretraining_mlm_probability must be in (0, 1), got {pretraining_mlm_probability}")
        if weight_decay is not None and weight_decay < 0:
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
        self.use_pos_weight = bool(use_pos_weight)
        self.use_pretraining = bool(use_pretraining)
        self.pretraining_epochs = int(pretraining_epochs)
        self.pretraining_batch_size = int(pretraining_batch_size) if pretraining_batch_size is not None else self.batch_size
        self.pretraining_learning_rate = float(pretraining_learning_rate)
        self.pretraining_mlm_probability = float(pretraining_mlm_probability)
        self._model: Optional[torch.nn.Module] = None
        self._label_order: Optional[List[str]] = None
        self._label_to_index: Optional[Dict[str, int]] = None
        self._index_to_label: Optional[Dict[int, str]] = None
        self._num_classes: int = 0
        self._loss_pos_weight: Optional[np.ndarray] = None

    def _create_model(self, num_classes: int):
        checkpoint_source = self._active_checkpoint or self.init_checkpoint
        base_model = load_backbone_with_optional_checkpoint(self.model_name, checkpoint_source)

        num_thresholds = num_classes - 1
        hidden_size = base_model.config.hidden_size
        macro_loss_weight = self.macro_loss_weight
        loss_pos_weight = self._loss_pos_weight

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
                    pos_weight = None
                    if loss_pos_weight is not None:
                        pos_weight = torch.tensor(loss_pos_weight, dtype=logits.dtype, device=logits.device)
                    base_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                        logits,
                        labels,
                        pos_weight=pos_weight,
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
        self._active_checkpoint = None

        train_targets, val_targets, num_classes, bias_init = self._prepare_targets(train_labels, val_labels, union_labels)
        self._num_classes = num_classes
        self._loss_pos_weight = None
        if self.use_pos_weight and train_targets.size > 0:
            positives = train_targets.sum(axis=0)
            negatives = train_targets.shape[0] - positives
            eps = 1e-8
            self._loss_pos_weight = (negatives / (positives + eps)).astype(np.float32)
            print(f"CORAL pos_weight: {self._loss_pos_weight.tolist()}")

        self._maybe_pretrain(
            use_pretraining=self.use_pretraining,
            model_name=self.model_name,
            texts=train_texts,
            output_dir=self.checkpoint_dir,
            init_checkpoint=self.init_checkpoint,
            pretraining_epochs=self.pretraining_epochs,
            pretraining_batch_size=self.pretraining_batch_size,
            pretraining_learning_rate=self.pretraining_learning_rate,
            pretraining_mlm_probability=self.pretraining_mlm_probability,
        )
        self._load_tokenizer_with_fallback(model_name=self.model_name, init_checkpoint=self.init_checkpoint)
        self._maybe_enable_stopwords(self.mask_stopwords)

        self._model = self._create_model(num_classes)

        with torch.no_grad():
            self._model.biases.data = torch.from_numpy(bias_init).float()

        train_encodings = self._encode_texts(train_texts, mask_stopwords=self.mask_stopwords)
        val_encodings = self._encode_texts(val_texts, mask_stopwords=self.mask_stopwords)
        train_dataset = EncodedDataset(train_encodings, train_targets, label_dtype=torch.float)
        val_dataset = EncodedDataset(val_encodings, val_targets, label_dtype=torch.float)

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

        wd_eff = self.weight_decay if self.weight_decay is not None else 0.01
        spec = HFTrainSpec(
            metric_name="macro_mae",
            minimize_metric=True,
            weight_decay_effective=wd_eff,
            compute_metrics=compute_metrics,
        )
        self._trainer, early_stopping = self._make_trainer(
            model=self._model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            spec=spec,
            checkpoint_dir=self.checkpoint_dir,
            epochs=self.epochs,
            batch_size=self.batch_size,
            warmup_steps=self.warmup_steps,
            head_lr=self.head_lr,
            gradient_clip=self.gradient_clip,
            optimizer_type=self.optimizer_type,
            scheduler_type=self.scheduler_type,
            encoder_lr=self.encoder_lr,
            weight_decay=wd_eff,
            warmup_ratio=self.warmup_ratio,
            early_stop_patience=self.early_stop_patience,
            early_stop_threshold=self.early_stop_threshold,
        )
        self._trainer.train()
        print(f"Restored best model with macro_mae: {early_stopping.best_metric:.4f}")

    def predict(self, x: Any) -> Sequence[Any]:
        if self._model is None or self._tokenizer is None:
            raise RuntimeError("Model not fitted")
        texts = list(x)
        outs = self._predict_in_batches(
            model=self._model,
            texts=texts,
            batch_size=self.batch_size,
            mask_stopwords=self.mask_stopwords,
        )
        all_preds = []
        for out in outs:
            logits = out["logits"]
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
        self._loss_pos_weight = None
        self.cleanup_plumbing(model_attr_names=["_model"])
