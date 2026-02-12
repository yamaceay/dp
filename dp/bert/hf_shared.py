from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Sequence, Set, Tuple

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer, Trainer, TrainingArguments

from dp.bert.common import (
    EarlyStoppingCallback,
    build_optimizer_and_scheduler,
    mask_stopword_tokens,
    pretrain_backbone_with_mlm,
)
from dp.utils.device import resolve_device


def load_backbone_with_optional_checkpoint(model_name: str, checkpoint_source: Optional[str]) -> Any:
    base_model = AutoModel.from_pretrained(model_name)
    if checkpoint_source:
        try:
            checkpoint_model = AutoModel.from_pretrained(checkpoint_source)
            base_model.load_state_dict(checkpoint_model.state_dict(), strict=False)
        except Exception:
            pass
    return base_model


class EncodedDataset(torch.utils.data.Dataset):
    def __init__(self, encodings: Dict[str, torch.Tensor], labels: Any, label_dtype: torch.dtype):
        self.encodings = encodings
        self.labels = labels
        self.label_dtype = label_dtype

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=self.label_dtype)
        return item


@dataclass
class HFTrainSpec:
    metric_name: str
    minimize_metric: bool
    weight_decay_effective: float
    compute_metrics: Callable


class BertHFPlumbing:
    def __init__(self, device: Optional[str]):
        self.device = resolve_device(device)
        self._tokenizer: Optional[AutoTokenizer] = None
        self._trainer: Optional[Trainer] = None
        self._stopwords: Set[str] = set()
        self._active_checkpoint: Optional[str] = None

    def _model_device(self, model: torch.nn.Module) -> torch.device:
        return next(model.parameters()).device

    def _maybe_pretrain(
        self,
        *,
        use_pretraining: bool,
        model_name: str,
        texts: Sequence[str],
        output_dir: str,
        init_checkpoint: Optional[str],
        pretraining_epochs: int,
        pretraining_batch_size: int,
        pretraining_learning_rate: float,
        pretraining_mlm_probability: float,
    ) -> None:
        self._active_checkpoint = None
        if not use_pretraining:
            return
        self._active_checkpoint = pretrain_backbone_with_mlm(
            model_name=model_name,
            texts=texts,
            output_dir=output_dir,
            init_checkpoint=init_checkpoint,
            epochs=pretraining_epochs,
            batch_size=pretraining_batch_size,
            learning_rate=pretraining_learning_rate,
            mlm_probability=pretraining_mlm_probability,
        )

    def _load_tokenizer_with_fallback(self, *, model_name: str, init_checkpoint: Optional[str]) -> None:
        tokenizer_source = self._active_checkpoint or init_checkpoint or model_name
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
        except Exception:
            self._tokenizer = AutoTokenizer.from_pretrained(model_name)

    def _maybe_enable_stopwords(self, mask_stopwords: bool) -> None:
        self._stopwords = set()
        if not mask_stopwords:
            return
        from dp.utils.stopwords import DEFAULT_STOPWORDS

        self._stopwords = DEFAULT_STOPWORDS
        print(f"Stopword masking enabled: {len(self._stopwords)} stopwords will be masked")

    def _encode_texts(self, texts: Sequence[str], *, mask_stopwords: bool) -> Dict[str, torch.Tensor]:
        if self._tokenizer is None:
            raise RuntimeError("Tokenizer not initialized")
        enc = self._tokenizer(list(texts), padding=True, truncation=True, return_tensors="pt")
        if mask_stopwords:
            enc = mask_stopword_tokens(self._tokenizer, self._stopwords, enc, list(texts))
        return enc

    def _make_trainer(
        self,
        *,
        model: torch.nn.Module,
        train_dataset: torch.utils.data.Dataset,
        val_dataset: torch.utils.data.Dataset,
        spec: HFTrainSpec,
        checkpoint_dir: str,
        epochs: int,
        batch_size: int,
        warmup_steps: int,
        head_lr: float,
        gradient_clip: float,
        optimizer_type: str,
        scheduler_type: str,
        encoder_lr: float,
        weight_decay: float,
        warmup_ratio: Optional[float],
        early_stop_patience: int,
        early_stop_threshold: Optional[float],
    ) -> Tuple[Trainer, EarlyStoppingCallback]:
        training_args = TrainingArguments(
            output_dir=checkpoint_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_steps=warmup_steps,
            learning_rate=head_lr,
            weight_decay=spec.weight_decay_effective,
            logging_steps=10,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model=spec.metric_name,
            greater_is_better=(not spec.minimize_metric),
            max_grad_norm=gradient_clip,
            report_to="none",
        )

        steps_per_epoch = max(1, int(np.ceil(len(train_dataset) / batch_size)))
        num_training_steps = steps_per_epoch * epochs

        optimizer, scheduler = build_optimizer_and_scheduler(
            model,
            optimizer_type=optimizer_type,
            scheduler_type=scheduler_type,
            encoder_lr=encoder_lr,
            head_lr=head_lr,
            weight_decay=weight_decay,
            warmup_steps=warmup_steps,
            warmup_ratio=warmup_ratio,
            num_training_steps=num_training_steps,
        )

        early_stopping = EarlyStoppingCallback(
            early_stopping_patience=early_stop_patience,
            early_stopping_threshold=early_stop_threshold,
            metric_name=spec.metric_name,
            minimize=spec.minimize_metric,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=spec.compute_metrics,
            callbacks=[early_stopping],
            optimizers=(optimizer, scheduler),
        )
        return trainer, early_stopping

    def _predict_in_batches(
        self,
        *,
        model: torch.nn.Module,
        texts: Sequence[str],
        batch_size: int,
        mask_stopwords: bool,
    ) -> Sequence[Any]:
        if self._tokenizer is None:
            raise RuntimeError("Tokenizer not initialized")
        model.eval()
        outputs = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = list(texts[i : i + batch_size])
                enc = self._tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt")
                if mask_stopwords:
                    enc = mask_stopword_tokens(self._tokenizer, self._stopwords, enc, batch_texts)
                enc = {k: v.to(self._model_device(model)) for k, v in enc.items()}
                out = model(**enc)
                outputs.append(out)
        return outputs

    def cleanup_plumbing(self, model_attr_names: Sequence[str]) -> None:
        for name in model_attr_names:
            obj = getattr(self, name, None)
            if obj is not None:
                del obj
            setattr(self, name, None)
        if self._tokenizer is not None:
            del self._tokenizer
        if self._trainer is not None:
            del self._trainer
        self._tokenizer = None
        self._trainer = None
        import gc

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
