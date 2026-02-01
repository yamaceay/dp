from __future__ import annotations

from typing import Dict, Sequence, Set

import torch
from transformers import TrainerCallback


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
