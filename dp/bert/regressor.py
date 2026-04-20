from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

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


class BertRegressorHead(SupervisedDownstreamHead, BertHFPlumbing):
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
        primary_metric: str = "r2",
        early_stop_threshold: Optional[float] = None,
        early_stop_patience: int = 2,
        init_checkpoint: Optional[str] = None,
        checkpoint_dir: Optional[str] = None,
        save_checkpoints: bool = True,
        normalize_targets: bool = True,
        use_pretraining: bool = False,
        pretraining_epochs: int = 1,
        pretraining_batch_size: Optional[int] = None,
        pretraining_learning_rate: float = 5e-5,
        pretraining_mlm_probability: float = 0.15,
        optimizer_type: str = "adamw",
        scheduler_type: str = "linear",
        weight_decay: float = 0.01,
        warmup_ratio: Optional[float] = None,
        training_status_file: str = "training_status.json",
        wait_for_training_completion: bool = True,
        training_poll_interval_seconds: float = 10.0,
        training_wait_timeout_seconds: Optional[float] = None,
        mark_existing_checkpoint_complete: bool = True,
    ):
        SupervisedDownstreamHead.__init__(self, name="bert_regressor", primary_metric=primary_metric)
        BertHFPlumbing.__init__(self, device=device)
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
        if encoder_lr <= 0:
            raise ValueError(f"encoder_lr must be positive, got {encoder_lr}")
        if head_lr is not None and head_lr <= 0:
            raise ValueError(f"head_lr must be positive, got {head_lr}")
        self.encoder_lr = float(encoder_lr)
        self.head_lr = float(head_lr) if head_lr is not None else float(encoder_lr)
        self.warmup_steps = int(warmup_steps)
        self.gradient_clip = float(gradient_clip)
        self.early_stop_threshold = float(early_stop_threshold) if early_stop_threshold is not None else None
        self.early_stop_patience = int(early_stop_patience)
        self.init_checkpoint = init_checkpoint
        self.checkpoint_dir = checkpoint_dir
        self.save_checkpoints = bool(save_checkpoints)
        self.normalize_targets = normalize_targets
        self.use_pretraining = bool(use_pretraining)
        self.pretraining_epochs = int(pretraining_epochs)
        self.pretraining_batch_size = int(pretraining_batch_size) if pretraining_batch_size is not None else self.batch_size
        self.pretraining_learning_rate = float(pretraining_learning_rate)
        self.pretraining_mlm_probability = float(pretraining_mlm_probability)
        self.optimizer_type = str(optimizer_type)
        self.scheduler_type = str(scheduler_type)
        self.weight_decay = float(weight_decay)
        self.warmup_ratio = float(warmup_ratio) if warmup_ratio is not None else None
        self.training_status_file = str(training_status_file)
        self.wait_for_training_completion = bool(wait_for_training_completion)
        self.training_poll_interval_seconds = float(training_poll_interval_seconds)
        self.training_wait_timeout_seconds = float(training_wait_timeout_seconds) if training_wait_timeout_seconds is not None else None
        self.mark_existing_checkpoint_complete = bool(mark_existing_checkpoint_complete)
        self._model: Optional[torch.nn.Module] = None
        self._target_mean: Optional[float] = None
        self._target_std: Optional[float] = None

    def _create_model(self):
        checkpoint_source = self._active_checkpoint or self.init_checkpoint
        base_model = load_backbone_with_optional_checkpoint(self.model_name, checkpoint_source)

        hidden_size = base_model.config.hidden_size

        class RegressorModel(torch.nn.Module):
            def __init__(self, base, hidden_size):
                super().__init__()
                self.base_model = base
                self._keys_to_ignore_on_save = []
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
        train_texts = list(x_train)
        val_texts = list(x_val)
        train_labels = np.asarray(list(y_train), dtype=float)
        val_labels = np.asarray(list(y_val), dtype=float)
        self._active_checkpoint = None

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
        self._model = self._create_model()
        def _train_impl() -> None:
            train_encodings = self._encode_texts(train_texts, mask_stopwords=False)
            val_encodings = self._encode_texts(val_texts, mask_stopwords=False)
            train_dataset = EncodedDataset(train_encodings, train_labels_normalized, label_dtype=torch.float)
            val_dataset = EncodedDataset(val_encodings, val_labels_normalized, label_dtype=torch.float)

            def compute_metrics(eval_pred: EvalPrediction):
                preds = eval_pred.predictions
                labels = eval_pred.label_ids

                if self.normalize_targets and self._target_mean is not None and self._target_std is not None:
                    preds = preds * self._target_std + self._target_mean
                    labels = labels * self._target_std + self._target_mean

                mse = float(np.mean((preds - labels) ** 2))
                rmse = float(np.sqrt(mse))
                denom = float(np.sum((labels - labels.mean()) ** 2))
                r2 = float(1 - (np.sum((labels - preds) ** 2) / denom)) if denom > 0 else 0.0

                return {
                    "rmse": rmse,
                    "r2": r2,
                }

            spec = HFTrainSpec(
                metric_name="r2",
                minimize_metric=False,
                weight_decay_effective=self.weight_decay,
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
                weight_decay=self.weight_decay,
                warmup_ratio=self.warmup_ratio,
                early_stop_patience=self.early_stop_patience,
                early_stop_threshold=self.early_stop_threshold,
                save_checkpoints=self.save_checkpoints,
            )
            self._trainer.train()
            if self.save_checkpoints and early_stopping.best_metric is not None:
                print(f"Restored best model with r2: {early_stopping.best_metric:.4f}")
            elif early_stopping.best_metric is not None:
                print(f"Best observed r2: {early_stopping.best_metric:.4f}")
        reused = self._run_training_with_reuse(
            model=self._model,
            checkpoint_dir=self.checkpoint_dir,
            status_file=self.training_status_file,
            wait_for_completion=self.wait_for_training_completion,
            poll_interval_seconds=self.training_poll_interval_seconds,
            wait_timeout_seconds=self.training_wait_timeout_seconds,
            mark_existing_complete=self.mark_existing_checkpoint_complete,
            train_fn=_train_impl,
        )
        if reused:
            return

    def predict(self, x: Any) -> Sequence[float]:
        if self._model is None or self._tokenizer is None:
            raise RuntimeError("Model not fitted")
        texts = list(x)
        outs = self._predict_in_batches(
            model=self._model,
            texts=texts,
            batch_size=self.batch_size,
            mask_stopwords=False,
        )
        all_preds = []
        for out in outs:
            preds = out["logits"].cpu().numpy()
            if self.normalize_targets and self._target_mean is not None and self._target_std is not None:
                preds = preds * self._target_std + self._target_mean
            all_preds.extend(preds)
        return np.array(all_preds, dtype=float)

    def evaluate(self, x: Any, y: Sequence[Any]) -> Dict[str, float]:
        predictions = self.predict(x)
        y_arr = np.asarray(y, dtype=float)
        mse = float(np.mean((predictions - y_arr) ** 2))
        rmse = float(np.sqrt(mse))
        denom = float(np.sum((y_arr - y_arr.mean()) ** 2))
        r2 = float(1 - (np.sum((y_arr - predictions) ** 2) / denom)) if denom > 0 else 0.0
        return {
            "rmse": rmse,
            "r2": r2,
        }

    def setup(self) -> None:
        pass

    def cleanup(self) -> None:
        self.cleanup_plumbing(model_attr_names=["_model"])
