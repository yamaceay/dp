from abc import ABC, abstractmethod
import os
import json
from typing import Optional, List, Dict, Any, Union, Tuple
import torch
from pathlib import Path
from torch.utils.data import Dataset
from torch.nn import functional as F
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForMaskedLM,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    DataCollatorForLanguageModeling,
    get_constant_schedule,
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    pipeline,
)
from torch.optim import AdamW, Adam, SGD
from dp.loaders.base import DatasetRecord
from dp.utils.chunking import TokenAwareChunker, ProbabilityAggregator, process_with_chunking
from dp.utils.device import resolve_device

class TRIDataset(Dataset):
    def __init__(
        self,
        records: List[DatasetRecord],
        tokenizer,
        name_to_label: Dict[str, int],
        max_length: int,
        use_labels: bool = True,
        stride_fraction: float = 0.25,
    ):
        if not records:
            raise ValueError("records cannot be empty")
        if max_length <= 0:
            raise ValueError(f"max_length must be positive, got {max_length}")
        if stride_fraction <= 0 or stride_fraction >= 1:
            raise ValueError(f"stride_fraction must be in (0, 1), got {stride_fraction}")
        self.input_ids: List[torch.Tensor] = []
        self.attention_masks: List[torch.Tensor] = []
        self.labels: List[torch.Tensor] = []
        stride_tokens = max(1, min(max_length - 1, int(max_length * stride_fraction)))
        for record in records:
            encodings = tokenizer(
                record.text,
                truncation=True,
                padding="max_length",
                max_length=max_length,
                return_overflowing_tokens=True,
                stride=stride_tokens,
            )
            input_set = encodings["input_ids"]
            if isinstance(input_set[0], int):
                input_set = [input_set]
            mask_set = encodings["attention_mask"]
            if isinstance(mask_set[0], int):
                mask_set = [mask_set]
            for ids, mask in zip(input_set, mask_set):
                self.input_ids.append(torch.tensor(ids))
                self.attention_masks.append(torch.tensor(mask))
                if use_labels:
                    self.labels.append(torch.tensor(name_to_label[record.name], dtype=torch.long))
        if not self.input_ids:
            raise ValueError("tokenization produced no samples")
        self.use_labels = use_labels
    
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        item = {
            "input_ids": self.input_ids[idx].clone().detach(),
            "attention_mask": self.attention_masks[idx].clone().detach(),
        }
        if self.use_labels:
            item["labels"] = self.labels[idx].clone().detach()
        return item

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    logits = torch.from_numpy(logits)
    logits_dict = {}
    for logit, label in zip(logits, labels):
        current_logits = logits_dict.get(label, torch.zeros_like(logit))
        logits_dict[label] = current_logits + logit
    num_predictions = len(logits_dict)
    all_predictions = torch.zeros(num_predictions, device="cpu")
    all_labels = torch.zeros(num_predictions, device="cpu")
    for idx, (label, summed_logits) in enumerate(logits_dict.items()):
        all_labels[idx] = torch.tensor(label)
        probabilities = F.softmax(summed_logits, dim=-1)
        all_predictions[idx] = torch.argmax(probabilities)
    correct_predictions = torch.sum(all_predictions == all_labels)
    accuracy = float(correct_predictions) / num_predictions
    return {"Accuracy": accuracy}

class MetricsPrintCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if not metrics:
            return control
        for key, value in metrics.items():
            if isinstance(value, float):
                print(f"{key}: {value:.4f}")
            else:
                print(f"{key}: {value}")
        return control

class EarlyStopCallback(TrainerCallback):
    def __init__(self, min_accuracy: float = 1.0):
        self.min_accuracy = min_accuracy
    
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if not metrics:
            return control
        accuracies = [v for k, v in metrics.items() if "Accuracy" in k and isinstance(v, (int, float))]
        if not accuracies:
            return control
        min_acc = min(accuracies)
        if min_acc >= self.min_accuracy:
            print(f"Minimum accuracy {min_acc:.2f} reached threshold {self.min_accuracy:.2f} for one metric, stopping training")
            control.should_training_stop = True
        return control

class TRIDetector(ABC):
    def __init__(
        self,
        dataset_name: Optional[str] = None,
        model_name: str = "distilbert-base-uncased",
        max_length: int = 512,
        device: Optional[Union[str, int]] = None,
        use_chunking: bool = True,
    ):
        if max_length <= 0:
            raise ValueError(f"max_length must be positive, got {max_length}")
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.max_length = max_length
        self.device = torch.device(resolve_device(device))
        self.use_chunking = use_chunking
        self.chunker = None
        self.tokenizer = None
        self.model = None
        self.pipe = None
        self.label_to_name = {}
        self.name_to_label = {}
        self.num_labels = 0
        self.train_records: Optional[List[DatasetRecord]] = None

    @abstractmethod
    def setup(self, *args, **kwargs):
        raise NotImplementedError
    
    @abstractmethod
    def get_eval_dataset(self, best_metric_dataset: Optional[str] = None, per_step: Optional[int] = None) -> Tuple[Union[TRIDataset, Dict[str, TRIDataset]], Dict[str, Any]]:
        raise NotImplementedError
    
    def pretrain(self, epochs: int, batch_size: int, learning_rate: float, output_dir: str) -> None:
        if not self.model:
            raise ValueError("Model must be initialized before pretraining")
        if not self.tokenizer:
            raise ValueError("Tokenizer must be initialized before pretraining")
        mlm_model = AutoModelForMaskedLM.from_pretrained(self.model_name)
        if hasattr(self.model, 'distilbert'):
            mlm_model.distilbert = self.model.distilbert
        elif hasattr(self.model, 'bert'):
            mlm_model.bert = self.model.bert
        elif hasattr(self.model, 'roberta'):
            mlm_model.roberta = self.model.roberta
        else:
            raise ValueError(f"Unsupported model architecture: {self.model_name}")
        mlm_model.to(self.device)
        train_dataset = TRIDataset(self.train_records, self.tokenizer, self.name_to_label, self.max_length, use_labels=False)
        data_collator = DataCollatorForLanguageModeling(self.tokenizer, mlm_probability=0.15)
        training_args = TrainingArguments(
            output_dir=f"{output_dir}/pretraining",
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            learning_rate=learning_rate,
            logging_strategy="epoch",
            save_strategy="epoch",
            save_steps=epochs,
            save_total_limit=1,
            report_to="none",
        )
        trainer = Trainer(
            model=mlm_model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=data_collator,
        )
        trainer.train()
        if hasattr(mlm_model, 'distilbert'):
            self.model.distilbert.load_state_dict(mlm_model.distilbert.state_dict())
        elif hasattr(mlm_model, 'roberta'):
            base_dict = mlm_model.roberta.state_dict()
            base_dict = {k: v for k, v in base_dict.items() if not k.startswith('pooler')}
            self.model.roberta.load_state_dict(base_dict, strict=False)
        elif hasattr(mlm_model, 'bert'):
            self.model.bert.load_state_dict(mlm_model.bert.state_dict())
        else:
            raise ValueError(f"Unable to transfer weights for {self.model_name}")
        del mlm_model
        torch.cuda.empty_cache()
    
    def predict(self, records: List[DatasetRecord]) -> Dict[str, Dict[str, float]]:
        if not records:
            return {}
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model not initialized. Train or load a model first.")
        if self.pipe is None:
            self.pipe = pipeline(
                "text-classification",
                model=self.model,
                tokenizer=self.tokenizer,
                device=self.device if self.device.type != "cpu" else -1,
                top_k=None,
                truncation=True,
                max_length=self.max_length,
            )
        results: Dict[str, Dict[str, float]] = {}
        if self.use_chunking and self.chunker is not None:
            aggregator = ProbabilityAggregator()
            def classify(text: str) -> Dict[str, float]:
                entries = self.pipe(text)[0]
                return self.map_prediction_entries(entries)
            for record in records:
                scores = process_with_chunking(record.text, self.chunker, classify, aggregator)
                results[record.uid] = scores
        else:
            for record in records:
                entries = self.pipe(record.text)[0]
                results[record.uid] = self.map_prediction_entries(entries)
        return results

    def evaluate(self, records: List[DatasetRecord]) -> Dict[str, Any]:
        if not records:
            raise ValueError("records cannot be empty")
        predictions = self.predict(records)
        correct = 0
        total = 0
        for record in records:
            if not record.name or record.name not in self.name_to_label:
                continue
            scores = predictions.get(record.uid)
            if not scores:
                continue
            predicted_name = max(scores.items(), key=lambda item: item[1])[0]
            if predicted_name == record.name:
                correct += 1
            total += 1
        accuracy = correct / total if total else 0.0
        return {
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
        }
    
    def load(self, model_path: Optional[str] = None) -> None:
        if model_path is None:
            if not self.dataset_name:
                raise ValueError("dataset_name must be set when model_path is not provided")
            model_path = f"models/tri_pipelines/{self.dataset_name}"
        if not Path(model_path).exists():
            raise ValueError(f"Model path does not exist: {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device)
        if self.max_length > 2:
            self.chunker = TokenAwareChunker(self.tokenizer, self.max_length - 2)
        else:
            self.chunker = TokenAwareChunker(self.tokenizer, 1)
        self.num_labels = self.model.config.num_labels
        label_mapping_path = Path(model_path) / "label_mapping.json"
        if not label_mapping_path.exists():
            raise ValueError(f"Label mapping not found: {label_mapping_path}")
        with open(label_mapping_path, "r") as f:
            self.name_to_label = json.load(f)
        self.label_to_name = {v: k for k, v in self.name_to_label.items()}

    def train(
        self,
        epochs: int = 3,
        batch_size: int = 16,
        learning_rate: float = 5e-5,
        output_dir: Optional[str] = None,
        use_pretraining: bool = False,
        pretraining_epochs: int = 3,
        best_metric_dataset: Optional[str] = None,
        early_stop_threshold: Optional[float] = None,
        per_step: Optional[int] = None,
        weight_decay: Optional[float] = None,
        warmup_ratio: Optional[float] = None,
        optimizer_type: Optional[str] = "adamw",
        scheduler_type: Optional[str] = "constant",
    ) -> None:
        if not self.train_records:
            raise ValueError("No training data set. Call set_train_dataset() first")
        if epochs <= 0:
            raise ValueError(f"epochs must be positive, got {epochs}")
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        if learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {learning_rate}")
        if use_pretraining and pretraining_epochs <= 0:
            raise ValueError(f"pretraining_epochs must be positive, got {pretraining_epochs}")
        if early_stop_threshold is not None and (early_stop_threshold <= 0.0 or early_stop_threshold > 1.0):
            raise ValueError(f"early_stop_threshold must be in (0, 1], got {early_stop_threshold}")
        if output_dir is None:
            if not self.dataset_name:
                raise ValueError("dataset_name must be set when output_dir is not provided")
            output_dir = f"models/tri_pipelines/{self.dataset_name}"
        os.makedirs(output_dir, exist_ok=True)
        self.initialize_tokenizer_and_model()
        if use_pretraining:
            self.pretrain(pretraining_epochs, batch_size, learning_rate, output_dir)
        train_dataset = TRIDataset(self.train_records, self.tokenizer, self.name_to_label, self.max_length)
        eval_dataset, eval_kwargs = self.get_eval_dataset(best_metric_dataset=best_metric_dataset, per_step=per_step)
        save_kwargs = {}
        if per_step:
            save_kwargs.update({
                "logging_strategy": "steps",
                "save_strategy": "steps",
                "save_steps": per_step,
            })
        else:
            save_kwargs.update({
                "logging_strategy": "epoch",
                "save_strategy": "epoch",
            })
        training_kwargs = {}
        if weight_decay is not None:
            training_kwargs["weight_decay"] = weight_decay
        if warmup_ratio is not None:
            training_kwargs["warmup_ratio"] = warmup_ratio
        use_cpu = str(self.device).startswith("cpu")
        ddp_backend = "gloo" if use_cpu else None
        local_rank = -1 if use_cpu else None
        if use_cpu:
            for key in ("WORLD_SIZE", "RANK", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT", "SLURM_PROCID", "SLURM_LOCALID", "SLURM_NTASKS"):
                os.environ.pop(key, None)
        training_args = TrainingArguments(
            output_dir=f"{output_dir}/finetuning",
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            save_total_limit=1,
            report_to="none",
            no_cuda=use_cpu,
            ddp_backend=ddp_backend,
            local_rank=local_rank,
            **training_kwargs,
            **save_kwargs,
            **eval_kwargs,
        )
        trainer_kwargs = {}
        if optimizer_type is not None:
            if optimizer_type == "adamw":
                optimizer = AdamW(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay or 0.0)
            elif optimizer_type == "adam":
                optimizer = Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay or 0.0)
            elif optimizer_type == "sgd":
                optimizer = SGD(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay or 0.0)
            else:
                raise ValueError(f"Unknown optimizer_type: {optimizer_type}")
            
            if scheduler_type == "constant":
                scheduler = get_constant_schedule(optimizer)
            elif scheduler_type == "linear":
                num_training_steps = len(train_dataset) * epochs // batch_size
                num_warmup_steps = int(num_training_steps * (warmup_ratio or 0.0))
                scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)
            elif scheduler_type == "cosine":
                num_training_steps = len(train_dataset) * epochs // batch_size
                num_warmup_steps = int(num_training_steps * (warmup_ratio or 0.0))
                scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)
            else:
                raise ValueError(f"Unknown scheduler_type: {scheduler_type}")
            
            trainer_kwargs["optimizers"] = [optimizer, scheduler]
        callbacks = [MetricsPrintCallback()]
        if early_stop_threshold is not None:
            callbacks.append(EarlyStopCallback(min_accuracy=early_stop_threshold))
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            **trainer_kwargs,
        )
        trainer.train()
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        label_mapping_path = Path(output_dir) / "label_mapping.json"
        with open(label_mapping_path, "w") as f:
            json.dump(self.name_to_label, f, indent=2)

    def parse_label_id(self, label: str) -> Optional[int]:
        if not label:
            return None
        if "_" not in label:
            return None
        _, value = label.rsplit("_", 1)
        if not value.isdigit():
            return None
        return int(value)

    def map_prediction_entries(self, entries: List[Dict[str, Any]]) -> Dict[str, float]:
        mapped: Dict[str, float] = {}
        for entry in entries:
            label = entry.get("label")
            score = entry.get("score")
            if label is None or score is None:
                continue
            label_id = self.parse_label_id(str(label))
            if label_id is None:
                continue
            name = self.label_to_name.get(label_id)
            if not name:
                continue
            mapped[name] = float(score)
        return mapped
    
    def initialize_tokenizer_and_model(self):
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            if self.max_length > 2:
                self.chunker = TokenAwareChunker(self.tokenizer, self.max_length - 2)
            else:
                self.chunker = TokenAwareChunker(self.tokenizer, 1)
        if self.model is None and self.num_labels > 0:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name, 
                num_labels=self.num_labels,
                problem_type="single_label_classification"
            )
            self.model.to(self.device)

    def build_label_mappings(self):
        if not self.train_records:
            return
        names = sorted(set(r.name for r in self.train_records))
        if self.name_to_label:
            existing = set(self.name_to_label.keys())
            incoming = set(names)
            if incoming != existing:
                missing = incoming.difference(existing)
                extra = existing.difference(incoming)
                raise ValueError(
                    "Existing TRI label mapping incompatible with training data. "
                    f"Missing: {sorted(missing)} Extra: {sorted(extra)}"
                )
            self.num_labels = len(self.name_to_label)
            return
        self.label_to_name = {idx: name for idx, name in enumerate(names)}
        self.name_to_label = {name: idx for idx, name in self.label_to_name.items()}
        self.num_labels = len(names)
