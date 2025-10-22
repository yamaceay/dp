from typing import List, Optional, Dict, Any
import os
import json
import torch
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset
from torch.nn import functional as F
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForSequenceClassification,
    AutoModelForMaskedLM,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    DataCollatorForLanguageModeling,
    get_constant_schedule,
    pipeline,
)
from torch.optim import AdamW
from collections import OrderedDict
from dp.loaders.base import DatasetRecord
from dp.utils.chunking import TokenAwareChunker, ProbabilityAggregator, process_with_chunking


def compute_metrics(eval_pred):
    """Compute accuracy by aggregating logits per individual."""
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

class TRIDetector:
    def __init__(
        self,
        dataset_name: Optional[str] = None,
        model_name: str = "distilbert-base-uncased",
        max_length: int = 512,
        device: str = "auto",
        use_chunking: bool = True,
    ):
        if max_length <= 0:
            raise ValueError(f"max_length must be positive, got {max_length}")
        
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.max_length = max_length
        self.device = self.resolve_device(device)
        self.use_chunking = use_chunking
        self.chunker = None
        
        self.tokenizer = None
        self.model = None
        self.label_to_name = {}
        self.name_to_label = {}
        self.num_labels = 0
        
        self.train_records: Optional[List[DatasetRecord]] = None
        self.eval_records_dict: Optional[Dict[str, List[DatasetRecord]]] = None
    
    def resolve_device(self, device: str) -> torch.device:
        if device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        return torch.device(device)
    
    def set_train_dataset(self, records: List[DatasetRecord]) -> None:
        if not records:
            raise ValueError("Training records cannot be empty")
        self.train_records = records
        self._build_label_mappings()
    
    def set_eval_datasets(self, records_dict: Dict[str, List[DatasetRecord]]) -> None:
        if not records_dict:
            raise ValueError("Evaluation records dict cannot be empty")
        for name, records in records_dict.items():
            if not records:
                raise ValueError(f"Evaluation dataset '{name}' has no records")
        self.eval_records_dict = records_dict
    
    def _build_label_mappings(self):
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
    
    def _initialize_tokenizer_and_model(self):
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
        **kwargs
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
        if early_stop_threshold is not None and (early_stop_threshold <= 0 or early_stop_threshold > 100):
            raise ValueError(f"early_stop_threshold must be in (0, 100], got {early_stop_threshold}")
        
        if output_dir is None:
            if not self.dataset_name:
                raise ValueError("dataset_name must be set when output_dir is not provided")
            output_dir = f"models/tri_pipelines/{self.dataset_name}"
        
        os.makedirs(output_dir, exist_ok=True)
        
        self._initialize_tokenizer_and_model()
        
        if use_pretraining:
            self._pretrain(pretraining_epochs, batch_size, learning_rate, output_dir)
        
        train_dataset = TRIDataset(self.train_records, self.tokenizer, self.name_to_label, self.max_length)
        
        eval_datasets_dict = None
        if self.eval_records_dict:
            eval_datasets_dict = {
                name: TRIDataset(records, self.tokenizer, self.name_to_label, self.max_length)
                for name, records in self.eval_records_dict.items()
            }
        
        eval_kwargs = {}
        if eval_datasets_dict:
            if best_metric_dataset:
                if best_metric_dataset not in eval_datasets_dict:
                    available = list(eval_datasets_dict.keys())
                    raise ValueError(f"best_metric_dataset '{best_metric_dataset}' not found in {available}")
                metric_for_best = f"eval_{best_metric_dataset}_Accuracy"
            else:
                first_eval_name = list(eval_datasets_dict.keys())[0]
                metric_for_best = f"eval_{first_eval_name}_Accuracy"
            
            eval_kwargs = {
                "eval_strategy": "epoch",
                "load_best_model_at_end": True,
                "metric_for_best_model": metric_for_best,
                "greater_is_better": True,
            }
        
        training_args = TrainingArguments(
            output_dir=f"{output_dir}/finetuning",
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            logging_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=1,
            report_to="none",
            no_cuda=True,
            **eval_kwargs,
        )
        
        optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        scheduler = get_constant_schedule(optimizer)
        
        callbacks = [MetricsPrintCallback()]
        if early_stop_threshold is not None:
            callbacks.append(EarlyStopCallback(min_accuracy=early_stop_threshold))
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_datasets_dict,
            optimizers=[optimizer, scheduler],
            compute_metrics=compute_metrics,
            callbacks=callbacks,
        )
        
        trainer.train()
        
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        label_mapping_path = Path(output_dir) / "label_mapping.json"
        with open(label_mapping_path, "w") as f:
            json.dump(self.name_to_label, f, indent=2)
    
    def _pretrain(self, epochs: int, batch_size: int, learning_rate: float, output_dir: str) -> None:
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
            save_strategy="no",
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
    
    def predict(self, records: List[DatasetRecord]) -> Dict[str, float]:
        if self.model is None:
            raise ValueError("Model not initialized. Train or load a model first.")
        
        pipe = pipeline("text-classification", model=self.model, tokenizer=self.tokenizer, device=self.device if self.device.type != "cpu" else -1, top_k=None, truncation=True, max_length=self.max_length)
        
        results = {}
        if self.use_chunking and self.chunker is not None:
            aggregator = ProbabilityAggregator()
            def classify(text: str) -> Dict[str, float]:
                preds = pipe(text)[0]
                return {pred["label"]: pred["score"] for pred in preds}
            for record in records:
                pred_dict = process_with_chunking(record.text, self.chunker, classify, aggregator)
                results[record.uid] = pred_dict
        else:
            for record in records:
                predictions = pipe(record.text)[0]
                pred_dict = {pred["label"]: pred["score"] for pred in predictions}
                results[record.uid] = pred_dict
        
        return results
    
    def evaluate(self, records: List[DatasetRecord]) -> Dict[str, Any]:
        if not records:
            raise ValueError("records cannot be empty")
        if not self.model:
            raise ValueError("Model not initialized")
        if not self.tokenizer:
            raise ValueError("Tokenizer not initialized")
        
        pipe = pipeline("text-classification", model=self.model, tokenizer=self.tokenizer, device=self.device if self.device.type != "cpu" else -1, truncation=True, max_length=self.max_length)
        
        correct = 0
        total = 0
        
        if self.use_chunking and self.chunker is not None:
            from dp.utils.chunking import ProbabilityAggregator, process_with_chunking
            aggregator = ProbabilityAggregator()
            def classify(text: str) -> Dict[str, float]:
                preds = pipe(text)[0]
                return {pred["label"]: pred["score"] for pred in preds}
            for record in records:
                pred_dict = process_with_chunking(record.text, self.chunker, classify, aggregator)
                predicted_label = max(pred_dict.items(), key=lambda item: item[1])[0]
                true_label = f"LABEL_{self.name_to_label[record.name]}"
                if predicted_label == true_label:
                    correct += 1
                total += 1
        else:
            for record in records:
                predictions = pipe(record.text)
                predicted_label = predictions[0]["label"]
                true_label = f"LABEL_{self.name_to_label[record.name]}"
                if predicted_label == true_label:
                    correct += 1
                total += 1
        
        accuracy = correct / total if total > 0 else 0
        
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


class TRIDataset(Dataset):
    def __init__(self, records: List[DatasetRecord], tokenizer, name_to_label: Dict[str, int], max_length: int, use_labels: bool = True, stride_fraction: float = 0.25):
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
