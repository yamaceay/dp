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
        encoder_lr: float,
        head_lr: Optional[float] = None,
        model_name: str = "distilbert-base-uncased",
        batch_size: int = 8,
        epochs: int = 5,
        warmup_steps: int = 10,
        gradient_clip: float = 1.0,
        device: Optional[str] = None,
        primary_metric: str = "macro_mae",
        early_stop_threshold: Optional[float] = None,
        early_stop_patience: int = 2,
        init_checkpoint: Optional[str] = None,
        checkpoint_dir: Optional[str] = None,
        save_checkpoints: bool = True,
        mask_stopwords: bool = False,
        optimizer_type: str = "adamw",
        scheduler_type: str = "linear",
        weight_decay: Optional[float] = None,
        warmup_ratio: Optional[float] = None,
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
        if encoder_lr <= 0:
            raise ValueError(f"encoder_lr must be positive, got {encoder_lr}")
        if head_lr is not None and head_lr <= 0:
            raise ValueError(f"head_lr must be positive, got {head_lr}")

        self.model_name = model_name
        self.batch_size = int(batch_size)
        self.epochs = int(epochs)
        self.encoder_lr = float(encoder_lr)
        self.head_lr = float(head_lr) if head_lr is not None else float(encoder_lr)
        self.warmup_steps = int(warmup_steps)
        self.gradient_clip = float(gradient_clip)
        self.early_stop_threshold = float(early_stop_threshold) if early_stop_threshold is not None else None
        self.early_stop_patience = int(early_stop_patience)
        self.init_checkpoint = init_checkpoint
        self.checkpoint_dir = checkpoint_dir or "tmp_hf_checkpoint"
        self.save_checkpoints = bool(save_checkpoints)
        self.mask_stopwords = bool(mask_stopwords)
        self.optimizer_type = str(optimizer_type)
        self.scheduler_type = str(scheduler_type)
        self.weight_decay = float(weight_decay) if weight_decay is not None else None
        self.warmup_ratio = float(warmup_ratio) if warmup_ratio is not None else None
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

    def _create_model(self, num_classes: int) -> torch.nn.Module:
        try:
            from coral_pytorch.dataset import levels_from_labelbatch
            from coral_pytorch.layers import CoralLayer
            from coral_pytorch.losses import coral_loss
        except Exception as exc:
            raise RuntimeError(
                "coral_pytorch is required for BertOrdinalHead. Install with: pip install coral_pytorch"
            ) from exc

        checkpoint_source = self._active_checkpoint or self.init_checkpoint
        base_model = load_backbone_with_optional_checkpoint(self.model_name, checkpoint_source)

        class OrdinalModel(torch.nn.Module):
            def __init__(self, base: torch.nn.Module, hidden_size: int, n_classes: int):
                super().__init__()
                self.base_model = base
                self._keys_to_ignore_on_save = []
                self.coral = CoralLayer(size_in=hidden_size, num_classes=n_classes)
                self.num_classes = n_classes

            def forward(self, input_ids, attention_mask, labels=None):
                outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
                hidden = outputs.last_hidden_state[:, 0, :]
                logits = self.coral(hidden)

                loss = None
                if labels is not None:
                    levels = levels_from_labelbatch(labels.long(), num_classes=self.num_classes).to(logits.device)
                    loss = coral_loss(logits, levels)

                return {"loss": loss, "logits": logits}

        hidden_size = int(base_model.config.hidden_size)
        return OrdinalModel(base_model, hidden_size, num_classes)

    def _prepare_targets(self, train_labels: Sequence[Any], val_labels: Sequence[Any], union_labels: Sequence[Any]):
        from collections import Counter

        if self._label_order is None:
            unique_labels = sorted(set(union_labels), key=str)
            self._label_order = [str(label) for label in unique_labels]
        self._label_to_index = {label: idx for idx, label in enumerate(self._label_order)}
        self._index_to_label = {idx: label for idx, label in enumerate(self._label_order)}

        train_encoded = np.array([self._label_to_index[str(label)] for label in train_labels], dtype=np.int64)
        val_encoded = np.array([self._label_to_index[str(label)] for label in val_labels], dtype=np.int64)
        num_classes = len(self._label_order)

        print(f"Training on {len(train_labels)} samples: {dict(Counter(train_labels))}")
        print(f"Validating on {len(val_labels)} samples: {dict(Counter(val_labels))}")
        return train_encoded, val_encoded, num_classes

    def fit(self, x_train: Any, y_train: Sequence[Any], x_val: Any, y_val: Sequence[Any]) -> None:
        train_texts = list(x_train)
        val_texts = list(x_val)
        train_labels = list(y_train)
        val_labels = list(y_val)
        self._active_checkpoint = None

        train_targets, val_targets, num_classes = self._prepare_targets(train_labels, val_labels, train_labels + val_labels)
        self._num_classes = num_classes

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

        train_encodings = self._encode_texts(train_texts, mask_stopwords=self.mask_stopwords)
        val_encodings = self._encode_texts(val_texts, mask_stopwords=self.mask_stopwords)
        train_dataset = EncodedDataset(train_encodings, train_targets, label_dtype=torch.long)
        val_dataset = EncodedDataset(val_encodings, val_targets, label_dtype=torch.long)

        def compute_metrics(eval_pred: EvalPrediction) -> Dict[str, float]:
            logits = np.asarray(eval_pred.predictions)
            labels = np.asarray(eval_pred.label_ids).astype(int)
            probs = 1.0 / (1.0 + np.exp(-logits))
            class_preds = (probs > 0.5).sum(axis=1).astype(int)

            unique_classes = sorted(set(labels.tolist()))
            per_class_mae: List[float] = []
            for cls in unique_classes:
                mask = labels == cls
                if mask.sum() > 0:
                    per_class_mae.append(float(np.abs(labels[mask] - class_preds[mask]).mean()))

            macro_mae = float(np.mean(per_class_mae)) if per_class_mae else float("inf")
            overall_mae = float(np.mean(np.abs(labels - class_preds)))
            return {"macro_mae": macro_mae, "mae": overall_mae}

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
            save_checkpoints=self.save_checkpoints,
        )
        self._trainer.train()
        if self.save_checkpoints and early_stopping.best_metric is not None:
            print(f"Restored best model with macro_mae: {early_stopping.best_metric:.4f}")
        elif early_stopping.best_metric is not None:
            print(f"Best observed macro_mae: {early_stopping.best_metric:.4f}")

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
        all_preds: List[Any] = []
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

        per_class_mae: List[float] = []
        per_class_recall: List[float] = []
        per_class_within1: List[float] = []
        for cls in unique_classes:
            mask = y_encoded == cls
            if mask.sum() > 0:
                per_class_mae.append(float(np.abs(y_encoded[mask] - pred_encoded[mask]).mean()))
                per_class_recall.append(float((y_encoded[mask] == pred_encoded[mask]).mean()))
                per_class_within1.append(float((np.abs(y_encoded[mask] - pred_encoded[mask]) <= 1).mean()))

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
        self.cleanup_plumbing(model_attr_names=["_model"])
