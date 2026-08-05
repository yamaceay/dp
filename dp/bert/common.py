from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, Optional, Sequence, Set, Tuple

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    get_constant_schedule,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)


class EarlyStoppingCallback(TrainerCallback):
    def __init__(self, early_stopping_patience: int, early_stopping_threshold: float | None, metric_name: str, minimize: bool):
        self.patience = early_stopping_patience
        self.threshold = early_stopping_threshold
        self.metric_name = metric_name
        self.minimize = minimize
        self.best_metric = float("inf") if minimize else -float("inf")
        self.wait = 0
        self.stopped_epoch = 0

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        current = metrics.get(f"eval_{self.metric_name}")
        if current is None:
            return
        if self.threshold is None:
            improved = (current < self.best_metric) if self.minimize else (current > self.best_metric)
            if improved:
                self.best_metric = current
                self.wait = 0
                control.should_save = True
            return
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


class ExternalEvalEarlyStoppingCallback(TrainerCallback):
    def __init__(
        self,
        *,
        early_stopping_patience: int,
        early_stopping_threshold: float | None,
        metric_name: str,
        minimize: bool,
        evaluator: Callable[[], Dict[str, float]],
        label: str = "test",
    ):
        self.patience = early_stopping_patience
        self.threshold = early_stopping_threshold
        self.metric_name = metric_name
        self.minimize = minimize
        self.evaluator = evaluator
        self.label = label
        self.best_metric = float("inf") if minimize else -float("inf")
        self.wait = 0
        self.stopped_epoch = 0
        self.last_metrics: Dict[str, float] = {}

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        values = self.evaluator()
        normalized: Dict[str, float] = {}
        for key, value in values.items():
            normalized[str(key)] = float(value)
        self.last_metrics = normalized
        current = normalized.get(self.metric_name)
        if current is None:
            raise ValueError(f"External evaluator did not return metric '{self.metric_name}'")
        payload = ", ".join(f"{k}={v:.6f}" for k, v in sorted(normalized.items()))
        print(f"{self.label} monitor: {payload}")
        if self.threshold is None:
            improved = (current < self.best_metric) if self.minimize else (current > self.best_metric)
            if improved:
                self.best_metric = current
                self.wait = 0
            else:
                self.wait += 1
                if self.wait >= self.patience:
                    control.should_training_stop = True
            return
        if (self.minimize and current <= self.threshold) or (not self.minimize and current >= self.threshold):
            control.should_training_stop = True
            return
        improved = (current < self.best_metric) if self.minimize else (current > self.best_metric)
        if improved:
            self.best_metric = current
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                control.should_training_stop = True


def mask_stopword_tokens(
    tokenizer,
    stopwords: Set[str],
    encodings: Dict[str, torch.Tensor],
    texts: Sequence[str],
) -> Dict[str, torch.Tensor]:
    for idx, text in enumerate(texts):
        token_ids = encodings["input_ids"][idx]
        attention_mask = encodings["attention_mask"][idx]
        for token_pos in range(len(token_ids)):
            token_id = token_ids[token_pos].item() if hasattr(token_ids[token_pos], "item") else token_ids[token_pos]
            token_str = tokenizer.convert_ids_to_tokens(token_id)
            normalized_token = token_str.lower().strip("#").replace("##", "")
            if normalized_token in stopwords:
                attention_mask[token_pos] = 0
    return encodings


class MaskedLMDataset(Dataset):
    def __init__(self, encodings: Dict[str, torch.Tensor]):
        self.encodings = encodings

    def __len__(self) -> int:
        return int(self.encodings["input_ids"].shape[0])

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {key: value[idx] for key, value in self.encodings.items()}


def pretrain_backbone_with_mlm(
    model_name: str,
    texts: Sequence[str],
    output_dir: str,
    *,
    init_checkpoint: Optional[str] = None,
    epochs: int = 1,
    batch_size: int = 8,
    learning_rate: float = 5e-5,
    mlm_probability: float = 0.15,
    seed: Optional[int] = None,
) -> str:
    if not texts:
        raise ValueError("texts cannot be empty for MLM pretraining")
    if epochs <= 0:
        raise ValueError(f"epochs must be positive, got {epochs}")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    if learning_rate <= 0:
        raise ValueError(f"learning_rate must be positive, got {learning_rate}")
    if mlm_probability <= 0.0 or mlm_probability >= 1.0:
        raise ValueError(f"mlm_probability must be in (0, 1), got {mlm_probability}")

    source = init_checkpoint or model_name
    try:
        tokenizer = AutoTokenizer.from_pretrained(source)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    try:
        mlm_model = AutoModelForMaskedLM.from_pretrained(source)
    except Exception:
        mlm_model = AutoModelForMaskedLM.from_pretrained(model_name)

    max_length = tokenizer.model_max_length
    if not isinstance(max_length, int) or max_length <= 0 or max_length > 4096:
        max_length = 512
    encodings = tokenizer(
        list(texts),
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )
    dataset = MaskedLMDataset(encodings)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=mlm_probability)
    pretrain_dir = Path(output_dir) / "pretraining"
    args = TrainingArguments(
        output_dir=str(pretrain_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=learning_rate,
        logging_strategy="epoch",
        save_strategy="no",
        report_to="none",
        seed=seed if seed is not None else 42,
    )
    trainer = Trainer(
        model=mlm_model,
        args=args,
        train_dataset=dataset,
        data_collator=data_collator,
    )
    trainer.train()
    backbone_dir = pretrain_dir / "backbone"
    mlm_model.save_pretrained(str(backbone_dir))
    tokenizer.save_pretrained(str(backbone_dir))
    return str(backbone_dir)


def build_optimizer_and_scheduler(
    model: torch.nn.Module,
    *,
    optimizer_type: str,
    scheduler_type: str,
    encoder_lr: float,
    head_lr: float,
    weight_decay: float,
    warmup_steps: int,
    warmup_ratio: Optional[float],
    num_training_steps: int,
) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]:
    if encoder_lr <= 0:
        raise ValueError(f"encoder_lr must be positive, got {encoder_lr}")
    if head_lr <= 0:
        raise ValueError(f"head_lr must be positive, got {head_lr}")
    if weight_decay < 0:
        raise ValueError(f"weight_decay must be >= 0, got {weight_decay}")
    if warmup_steps < 0:
        raise ValueError(f"warmup_steps must be >= 0, got {warmup_steps}")
    if warmup_ratio is not None and (warmup_ratio < 0.0 or warmup_ratio >= 1.0):
        raise ValueError(f"warmup_ratio must be in [0, 1), got {warmup_ratio}")
    if num_training_steps <= 0:
        raise ValueError(f"num_training_steps must be positive, got {num_training_steps}")

    encoder_params = []
    head_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("base_model."):
            encoder_params.append(param)
        else:
            head_params.append(param)
    param_groups = []
    if encoder_params:
        param_groups.append({"params": encoder_params, "lr": encoder_lr, "weight_decay": weight_decay})
    if head_params:
        param_groups.append({"params": head_params, "lr": head_lr, "weight_decay": weight_decay})
    if not param_groups:
        raise ValueError("No trainable parameters found for optimizer")

    optimizer_key = optimizer_type.lower().strip()
    if optimizer_key == "adamw":
        optimizer = torch.optim.AdamW(param_groups)
    elif optimizer_key == "adam":
        optimizer = torch.optim.Adam(param_groups)
    elif optimizer_key == "sgd":
        optimizer = torch.optim.SGD(param_groups, momentum=0.9)
    else:
        raise ValueError(f"Unknown optimizer_type: {optimizer_type}")

    warmup = warmup_steps if warmup_steps > 0 else int(num_training_steps * (warmup_ratio or 0.0))
    scheduler_key = scheduler_type.lower().strip()
    if scheduler_key == "constant":
        scheduler = get_constant_schedule(optimizer)
    elif scheduler_key == "linear":
        scheduler = get_linear_schedule_with_warmup(optimizer, warmup, num_training_steps)
    elif scheduler_key == "cosine":
        scheduler = get_cosine_schedule_with_warmup(optimizer, warmup, num_training_steps)
    else:
        raise ValueError(f"Unknown scheduler_type: {scheduler_type}")
    return optimizer, scheduler
