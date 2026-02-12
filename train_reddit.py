from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error
from torch.optim import AdamW
from transformers import (
    AutoModel,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    EvalPrediction,
    Trainer,
    TrainingArguments,
    get_linear_schedule_with_warmup,
)

from dp.loaders.derive import NOMINAL_GROUPERS, ORDINAL_GROUPERS


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def prepare_dataset(records: List[Dict[str, Any]], grouper, feature: str) -> Dict[str, List[Any]]:
    dataset: Dict[str, List[Any]] = {"texts": [], "labels": []}
    for record in records:
        response = str(record.get("response", "")).strip()
        question = str(record.get("question_asked", "")).strip()
        text = f"{question}\n{response}".strip()  # or just response
        if not text:
            continue

        actual_feature = str(record.get("feature", "")).strip()
        if actual_feature != feature:
            continue
        label = record.get("personality", {}).get(actual_feature)
        if label is None:
            raise ValueError(f"Missing label for feature '{feature}' in record: {record}")
        label = grouper(label) if grouper else label
        dataset["texts"].append(text)
        dataset["labels"].append(label)
    return dataset


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


def compute_coral_bias_init(encoded_labels: np.ndarray, num_classes: int) -> np.ndarray:
    eps = 1e-7
    thresholds = num_classes - 1
    bias = np.zeros(thresholds, dtype=np.float32)
    for t in range(thresholds):
        p = float((encoded_labels > t).mean())
        p = min(max(p, eps), 1.0 - eps)
        bias[t] = math.log(p / (1.0 - p))
    return bias


class CoralOrdinalModel(torch.nn.Module):
    def __init__(
        self,
        model_name: str,
        num_classes: int,
        bias_init: np.ndarray,
        pos_weight: Optional[np.ndarray],
    ):
        super().__init__()
        self.base_model = AutoModel.from_pretrained(model_name)
        hidden_size = int(self.base_model.config.hidden_size)
        self.pre_classifier = torch.nn.Linear(hidden_size, hidden_size)
        self.dropout = torch.nn.Dropout(0.1)
        self.weight = torch.nn.Parameter(torch.randn(hidden_size) * 0.01)
        self.biases = torch.nn.Parameter(torch.from_numpy(bias_init).float())
        self.num_classes = int(num_classes)
        if pos_weight is None:
            self.register_buffer("pos_weight", None, persistent=False)
        else:
            self.register_buffer("pos_weight", torch.tensor(pos_weight, dtype=torch.float32), persistent=False)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state[:, 0, :]
        hidden = self.pre_classifier(hidden)
        hidden = torch.nn.functional.relu(hidden)
        hidden = self.dropout(hidden)
        logits = torch.matmul(hidden, self.weight.unsqueeze(1))
        logits = logits + self.biases.unsqueeze(0)
        loss = None
        if labels is not None:
            pw = self.pos_weight.to(logits.device) if self.pos_weight is not None else None
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels, pos_weight=pw)
        return {"loss": loss, "logits": logits}


def encode_cumulative(encoded: np.ndarray, num_classes: int) -> np.ndarray:
    thresholds = num_classes - 1
    cumulative = np.zeros((len(encoded), thresholds), dtype=np.float32)
    for i, value in enumerate(encoded):
        cumulative[i, : int(value)] = 1.0
    return cumulative


def build_optimizer_and_scheduler(
    model: torch.nn.Module,
    base_prefix: str,
    encoder_lr: float,
    head_lr: float,
    weight_decay: float,
    warmup_steps: int,
    num_training_steps: int,
) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]:
    encoder_params: List[torch.nn.Parameter] = []
    head_params: List[torch.nn.Parameter] = []
    encoder_prefix = f"{base_prefix}."
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith(encoder_prefix):
            encoder_params.append(param)
        else:
            head_params.append(param)
    param_groups: List[Dict[str, Any]] = []
    if encoder_params:
        param_groups.append({"params": encoder_params, "lr": encoder_lr, "weight_decay": weight_decay})
    if head_params:
        param_groups.append({"params": head_params, "lr": head_lr, "weight_decay": weight_decay})
    optimizer = AdamW(param_groups)
    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_training_steps,
    )
    return optimizer, scheduler


def train_ordinal(
    args: argparse.Namespace,
    train_texts: Sequence[str],
    train_labels: Sequence[str],
    eval_texts: Sequence[str],
    eval_labels: Sequence[str],
    label_order: List[str],
) -> None:
    label_to_idx = {label: i for i, label in enumerate(label_order)}
    idx_to_label = {i: label for i, label in enumerate(label_order)}
    y_train = np.array([label_to_idx[str(v)] for v in train_labels], dtype=np.int64)
    y_eval = np.array([label_to_idx[str(v)] for v in eval_labels], dtype=np.int64)
    num_classes = len(label_order)
    if args.disable_bias_init:
        bias_init = np.zeros(num_classes - 1, dtype=np.float32)
    else:
        bias_init = compute_coral_bias_init(y_train, num_classes) * float(args.bias_init_scale)
    y_train_coral = encode_cumulative(y_train, num_classes)
    y_eval_coral = encode_cumulative(y_eval, num_classes)

    pos_weight: Optional[np.ndarray] = None
    if args.use_pos_weight:
        positives = y_train_coral.sum(axis=0)
        negatives = y_train_coral.shape[0] - positives
        pos_weight = (negatives / (positives + 1e-8)).astype(np.float32)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    train_enc = tokenizer(list(train_texts), padding=True, truncation=True, return_tensors="pt")
    eval_enc = tokenizer(list(eval_texts), padding=True, truncation=True, return_tensors="pt")
    train_ds = EncodedDataset(train_enc, y_train_coral, label_dtype=torch.float32)
    eval_ds = EncodedDataset(eval_enc, y_eval_coral, label_dtype=torch.float32)

    model = CoralOrdinalModel(
        model_name=args.model_name,
        num_classes=num_classes,
        bias_init=bias_init,
        pos_weight=pos_weight,
    )

    def compute_metrics(eval_pred: EvalPrediction) -> Dict[str, float]:
        logits = eval_pred.predictions
        labels = eval_pred.label_ids
        probs = 1.0 / (1.0 + np.exp(-logits))
        pred_class = (probs > 0.5).sum(axis=1).astype(int)
        true_class = (labels > 0.5).sum(axis=1).astype(int)
        return {
            "macro_mae": float(np.mean([float(np.abs(true_class[true_class == c] - pred_class[true_class == c]).mean()) for c in np.unique(true_class)])),
            "mae": float(mean_absolute_error(true_class, pred_class)),
        }

    steps_per_epoch = max(1, int(math.ceil(len(train_ds) / args.batch_size)))

    warmup_epochs = max(0, int(args.freeze_encoder_epochs))
    warmup_trainer: Optional[Trainer] = None
    if warmup_epochs > 0:
        for p in model.base_model.parameters():
            p.requires_grad = False
        warmup_steps_total = steps_per_epoch * warmup_epochs
        warmup_opt, warmup_sched = build_optimizer_and_scheduler(
            model=model,
            base_prefix="base_model",
            encoder_lr=0.0,
            head_lr=args.head_only_lr,
            weight_decay=args.weight_decay,
            warmup_steps=args.head_warmup_steps,
            num_training_steps=max(1, warmup_steps_total),
        )
        warmup_args = TrainingArguments(
            output_dir=args.checkpoint_dir,
            num_train_epochs=warmup_epochs,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            eval_strategy="no",
            save_strategy="no",
            report_to="none",
        )
        warmup_trainer = Trainer(
            model=model,
            args=warmup_args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            compute_metrics=compute_metrics,
            optimizers=(warmup_opt, warmup_sched),
        )
        warmup_trainer.train()
        for p in model.base_model.parameters():
            p.requires_grad = True

    finetune_epochs = max(0, int(args.epochs) - warmup_epochs)
    trainer: Optional[Trainer] = None
    if finetune_epochs > 0:
        training_args = TrainingArguments(
            output_dir=args.checkpoint_dir,
            num_train_epochs=finetune_epochs,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            warmup_steps=args.warmup_steps,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="macro_mae",
            greater_is_better=False,
            report_to="none",
        )
        total_steps = steps_per_epoch * finetune_epochs
        optimizer, scheduler = build_optimizer_and_scheduler(
            model=model,
            base_prefix="base_model",
            encoder_lr=args.encoder_lr,
            head_lr=args.head_lr,
            weight_decay=args.weight_decay,
            warmup_steps=args.warmup_steps,
            num_training_steps=max(1, total_steps),
        )
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_stop_patience)],
            optimizers=(optimizer, scheduler),
        )
        trainer.train()
        eval_metrics = trainer.evaluate()
        pred_out = trainer.predict(eval_ds).predictions
        bias = trainer.model.biases.detach().cpu().numpy()
        w_norm = float(trainer.model.weight.detach().norm().cpu().item())
    else:
        if warmup_trainer is None:
            raise RuntimeError("No training phase executed")
        eval_metrics = warmup_trainer.evaluate(eval_ds)
        pred_out = warmup_trainer.predict(eval_ds).predictions
        bias = warmup_trainer.model.biases.detach().cpu().numpy()
        w_norm = float(warmup_trainer.model.weight.detach().norm().cpu().item())
    wx_matrix = pred_out - bias.reshape(1, -1)
    wx_values = wx_matrix.mean(axis=1)
    pred_probs = 1.0 / (1.0 + np.exp(-pred_out))
    pred_threshold_rates = pred_probs.mean(axis=0)
    true_threshold_rates = y_eval_coral.mean(axis=0)
    print(
        "CORAL wx diagnostics:",
        {
            "std_wx": float(wx_values.std()),
            "mean_wx": float(wx_values.mean()),
            "min_wx": float(wx_values.min()),
            "max_wx": float(wx_values.max()),
            "w_norm": w_norm,
            "biases": [float(v) for v in bias.tolist()],
            "std_per_threshold": [float(v) for v in wx_matrix.std(axis=0).tolist()],
            "mean_row_std_across_thresholds": float(wx_matrix.std(axis=1).mean()),
            "max_row_std_across_thresholds": float(wx_matrix.std(axis=1).max()),
            "true_threshold_rates": [float(v) for v in true_threshold_rates.tolist()],
            "pred_threshold_rates": [float(v) for v in pred_threshold_rates.tolist()],
        },
    )
    if args.debug_coral_outputs:
        probs = pred_probs
        n = min(args.debug_samples, len(pred_out))
        print("CORAL raw outputs (before thresholding):")
        for i in range(n):
            true_idx = int(y_eval[i])
            true_label = idx_to_label[true_idx]
            row_logits = [float(v) for v in pred_out[i].tolist()]
            row_probs = [float(v) for v in probs[i].tolist()]
            print(
                {
                    "i": i,
                    "true_idx": true_idx,
                    "true_label": true_label,
                    "logits": row_logits,
                    "probs": row_probs,
                }
            )
    pred_idx = (1.0 / (1.0 + np.exp(-pred_out)) > 0.5).sum(axis=1).astype(int)
    pred_labels = [idx_to_label[int(i)] for i in pred_idx]
    print(f"Evaluation metrics: {eval_metrics}")
    print(f"Prediction distribution: {dict(Counter(pred_labels))}")


def train_non_ordinal(
    args: argparse.Namespace,
    target_type: str,
    train_texts: Sequence[str],
    train_labels: Sequence[Any],
    eval_texts: Sequence[str],
    eval_labels: Sequence[Any],
) -> None:
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    train_enc = tokenizer(list(train_texts), padding=True, truncation=True, return_tensors="pt")
    eval_enc = tokenizer(list(eval_texts), padding=True, truncation=True, return_tensors="pt")

    if target_type == "nominal":
        classes = sorted({str(v) for v in list(train_labels) + list(eval_labels)})
        label_to_idx = {label: i for i, label in enumerate(classes)}
        idx_to_label = {i: label for i, label in enumerate(classes)}
        y_train = [label_to_idx[str(v)] for v in train_labels]
        y_eval = [label_to_idx[str(v)] for v in eval_labels]
        train_ds = EncodedDataset(train_enc, y_train, label_dtype=torch.long)
        eval_ds = EncodedDataset(eval_enc, y_eval, label_dtype=torch.long)
        model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=len(classes))

        def compute_metrics(eval_pred: EvalPrediction) -> Dict[str, float]:
            labels = eval_pred.label_ids
            preds = eval_pred.predictions.argmax(-1)
            return {
                "accuracy": float(accuracy_score(labels, preds)),
                "macro_f1": float(f1_score(labels, preds, average="macro", zero_division=0)),
            }

        metric_name = "macro_f1"
        greater = True
    elif target_type == "cardinal":
        y_train = [float(v) for v in train_labels]
        y_eval = [float(v) for v in eval_labels]
        train_ds = EncodedDataset(train_enc, y_train, label_dtype=torch.float32)
        eval_ds = EncodedDataset(eval_enc, y_eval, label_dtype=torch.float32)
        model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=1)
        model.config.problem_type = "regression"

        def compute_metrics(eval_pred: EvalPrediction) -> Dict[str, float]:
            labels = eval_pred.label_ids
            preds = eval_pred.predictions.reshape(-1)
            return {"mae": float(mean_absolute_error(labels, preds))}

        metric_name = "mae"
        greater = False
    else:
        raise ValueError(f"Unknown target type: {target_type}")

    training_args = TrainingArguments(
        output_dir=args.checkpoint_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.head_lr,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model=metric_name,
        greater_is_better=greater,
        report_to="none",
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_stop_patience)],
    )
    trainer.train()
    eval_metrics = trainer.evaluate()
    pred_out = trainer.predict(eval_ds).predictions
    if target_type == "nominal":
        pred_idx = pred_out.argmax(-1)
        pred_labels = [idx_to_label[int(i)] for i in pred_idx]
    else:
        pred_labels = [str(float(v)) for v in pred_out.reshape(-1)]
    print(f"Evaluation metrics: {eval_metrics}")
    print(f"Prediction distribution: {dict(Counter(pred_labels))}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature", type=str, required=True)
    parser.add_argument("--train_in", type=str, default="data/reddit/synthetic_dataset.jsonl")
    parser.add_argument("--eval_in", type=str, default="data/reddit/reddit.jsonl")
    parser.add_argument("--model_name", type=str, default="distilbert-base-uncased")
    parser.add_argument("--checkpoint_dir", type=str, default="tmp_hf_checkpoint")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--encoder_lr", type=float, default=1e-5)
    parser.add_argument("--head_lr", type=float, default=5e-5)
    parser.add_argument("--warmup_steps", type=int, default=10)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--early_stop_patience", type=int, default=20)
    parser.add_argument("--use_pos_weight", action="store_true")
    parser.add_argument("--disable_bias_init", action="store_true")
    parser.add_argument("--bias_init_scale", type=float, default=1.0)
    parser.add_argument("--freeze_encoder_epochs", type=int, default=0)
    parser.add_argument("--head_only_lr", type=float, default=1e-3)
    parser.add_argument("--head_warmup_steps", type=int, default=0)
    parser.add_argument("--debug_coral_outputs", action="store_true")
    parser.add_argument("--debug_samples", type=int, default=20)
    args = parser.parse_args()

    train_records = load_jsonl(args.train_in)
    eval_records = load_jsonl(args.eval_in)

    feature_type: str
    grouper = None
    label_order: Optional[List[str]] = None
    if values := ORDINAL_GROUPERS.get(args.feature):
        feature_type = "ordinal"
        grouper, label_order = values
    elif values := NOMINAL_GROUPERS.get(args.feature):
        feature_type = "nominal"
        grouper = values
    else:
        feature_type = "cardinal"

    train_dataset = prepare_dataset(train_records, grouper, args.feature)
    eval_dataset = prepare_dataset(eval_records, grouper, args.feature)
    if not train_dataset["texts"]:
        raise SystemExit("No train samples after filtering")
    if not eval_dataset["texts"]:
        raise SystemExit("No eval samples after filtering")

    lengths = [len(t) for t in train_dataset["texts"]]
    print("Train text length min/mean/max:", min(lengths), sum(lengths)/len(lengths), max(lengths))


    print(f"Prepared train dataset with {len(train_dataset['texts'])} records.")
    print(f"Prepared eval dataset with {len(eval_dataset['texts'])} records.")
    print(f"Train distribution: {dict(Counter(train_dataset['labels']))}")
    print(f"Eval distribution: {dict(Counter(eval_dataset['labels']))}")

    if feature_type == "ordinal":
        if label_order is None:
            raise ValueError("Ordinal feature requires explicit label order")
        train_ordinal(
            args=args,
            train_texts=train_dataset["texts"],
            train_labels=train_dataset["labels"],
            eval_texts=eval_dataset["texts"],
            eval_labels=eval_dataset["labels"],
            label_order=label_order,
        )
    else:
        train_non_ordinal(
            args=args,
            target_type=feature_type,
            train_texts=train_dataset["texts"],
            train_labels=train_dataset["labels"],
            eval_texts=eval_dataset["texts"],
            eval_labels=eval_dataset["labels"],
        )


if __name__ == "__main__":
    main()
