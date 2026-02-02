from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer, EvalPrediction, Trainer, TrainingArguments

from dp.bert.base import SupervisedDownstreamHead
from dp.bert.common import EarlyStoppingCallback, build_optimizer_and_scheduler, pretrain_backbone_with_mlm
from dp.utils.device import resolve_device


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
        super().__init__(name="bert_regressor", primary_metric=primary_metric)
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
        self.device = resolve_device(device)
        self.early_stop_threshold = float(early_stop_threshold) if early_stop_threshold is not None else None
        self.early_stop_patience = int(early_stop_patience)
        self.init_checkpoint = init_checkpoint
        self.checkpoint_dir = checkpoint_dir or "tmp_hf_checkpoint"
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
        self._tokenizer: Optional[AutoTokenizer] = None
        self._model: Optional[torch.nn.Module] = None
        self._trainer: Optional[Trainer] = None
        self._target_mean: Optional[float] = None
        self._target_std: Optional[float] = None
        self._active_checkpoint: Optional[str] = None

    def _create_model(self):
        base_model = AutoModel.from_pretrained(self.model_name)
        checkpoint_source = self._active_checkpoint or self.init_checkpoint
        if checkpoint_source:
            try:
                checkpoint_model = AutoModel.from_pretrained(checkpoint_source)
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

        if self.use_pretraining:
            self._active_checkpoint = pretrain_backbone_with_mlm(
                model_name=self.model_name,
                texts=train_texts,
                output_dir=self.checkpoint_dir,
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
                item["labels"] = torch.tensor(self.labels[idx], dtype=torch.float)
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
            denom = float(np.sum((labels - labels.mean()) ** 2))
            r2 = float(1 - (np.sum((labels - preds) ** 2) / denom)) if denom > 0 else 0.0

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
            weight_decay=self.weight_decay,
            logging_steps=10,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="r2",
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
            optimizers=(optimizer, scheduler),
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
                encodings = {k: v.to(self._model_device()) for k, v in encodings.items()}
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
        denom = float(np.sum((y_arr - y_arr.mean()) ** 2))
        r2 = float(1 - (np.sum((y_arr - predictions) ** 2) / denom)) if denom > 0 else 0.0
        return {
            "rmse": rmse,
            "r2": r2,
        }

    def setup(self) -> None:
        pass

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
