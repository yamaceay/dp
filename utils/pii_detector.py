from typing import List, Optional, Dict, Any
import os
import torch
import numpy as np
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
    pipeline,
)
from torch.utils.data import Dataset
from sklearn.metrics import precision_recall_fscore_support, classification_report

try:
    from nervaluate import Evaluator, collect_named_entities
    NERVALUATE_AVAILABLE = True
except ImportError:
    NERVALUATE_AVAILABLE = False

from dp.loaders.base import DatasetRecord, TextAnnotation
from dp.utils.chunking import SpanMergeAggregator, process_with_chunking

class PIIDetector:
    def __init__(
        self,
        model_name: str = "roberta-base",
        labels: Optional[List[str]] = None,
        max_length: int = 512,
        device: str = "auto",
        use_chunking: bool = False,
    ):
        self.model_name = model_name
        self.max_length = max_length
        self.use_chunking = use_chunking
        self.chunker = None
        
        self.device = self.resolve_device(device)
        
        self.labels = []
        self.label_to_id = {}
        self.id_to_label = {}
        self.tokenizer = None
        self.model = None
        self.pipeline = None

        if os.path.isdir(model_name) and os.path.exists(os.path.join(model_name, "config.json")):
            self._load_pretrained_model()
        elif labels is not None:
            self.labels = labels
            self._initialize_model_and_tokenizer(labels)
        
        self.train_records: Optional[List[DatasetRecord]] = None
        self.val_records: Optional[List[DatasetRecord]] = None
        self.test_records: Optional[List[DatasetRecord]] = None

        if self.labels:
            print(f"✓ PIIDetector initialized with {len(self.labels)} labels on {self.device}")
        else:
            print(f"✓ PIIDetector initialized without labels on {self.device}")
    
    def set_train_dataset(self, records: List[DatasetRecord]) -> None:
        self.train_records = records
        print(f"✓ Training dataset set: {len(records)} records")
    
    def set_val_dataset(self, records: List[DatasetRecord]) -> None:
        self.val_records = records
        print(f"✓ Validation dataset set: {len(records)} records")
    
    def set_test_dataset(self, records: List[DatasetRecord]) -> None:
        self.test_records = records
        print(f"✓ Test dataset set: {len(records)} records")
    
    def _get_labels(self, records: List[DatasetRecord]) -> List[str]:
        labels = set()
        for record in records:
            spans = record.spans or []
            for span in spans:
                if span.label:
                    label = span.label
                    if label.startswith("B-") or label.startswith("I-"):
                        label = label[2:]
                    labels.add(label)
        
        labels = sorted(list(labels))
        b_labels = [f"B-{label}" for label in labels]
        i_labels = [f"I-{label}" for label in labels]
        return ["O"] + b_labels + i_labels

    def resolve_device(self, device: str) -> torch.device:
        if device == "auto":
            if torch.backends.mps.is_available():
                device = torch.device("mps")
            elif torch.cuda.is_available():
                device = torch.device("cuda")
            else:
                device = torch.device("cpu")
        else:
            device = torch.device(device)
        return device

    def _load_pretrained_model(self) -> None:
        import os
        
        model_path = self.model_name
        if os.path.isdir(model_path):
            checkpoints = [d for d in os.listdir(model_path) if d.startswith('checkpoint-')]
            if checkpoints:
                latest_checkpoint = sorted(checkpoints, key=lambda x: int(x.split('-')[1]))[-1]
                model_path = os.path.join(model_path, latest_checkpoint)
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForTokenClassification.from_pretrained(model_path)
        self.model.to(self.device)
        
        config = self.model.config
        if hasattr(config, 'id2label') and config.id2label:
            self.id_to_label = config.id2label
            self.label_to_id = {v: k for k, v in self.id_to_label.items()}
            self.labels = [self.id_to_label[i] for i in sorted(self.id_to_label.keys())]

    def _late_initialize(self) -> None:
        if not self.labels or self.model is None:
            all_records = (
                (self.train_records if self.train_records else []) +
                (self.val_records if self.val_records else []) + 
                (self.test_records if self.test_records else [])
            )
            self._initialize_labels(self._get_labels(all_records))
            self._initialize_model_and_tokenizer(self.labels)

    def _initialize_labels(self, labels: Optional[List[str]] = None) -> None:
        if labels is not None:
            self.labels = labels
            self.label_to_id = {label: idx for idx, label in enumerate(labels)}
            self.id_to_label = {idx: label for label, idx in self.label_to_id.items()}

        elif self.model is not None:
            config = self.model.config
            if hasattr(config, 'id2label') and config.id2label:
                self.id_to_label = config.id2label
                self.label_to_id = {v: k for k, v in self.id_to_label.items()}
                self.labels = [self.id_to_label[i] for i in sorted(self.id_to_label.keys())]

        else:
            raise ValueError("Labels must be provided or model must be loaded to initialize labels")

    def _initialize_model_and_tokenizer(self, labels: List[str]) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForTokenClassification.from_pretrained(
            self.model_name,
            num_labels=len(labels),
            id2label=self.id_to_label,
            label2id=self.label_to_id,
            ignore_mismatched_sizes=True,
        ).to(self.device)

    def train(
        self,
        epochs: int = 5,
        output_dir: str = "./pii_model_output",
        batch_size: int = 8,
        learning_rate: float = 2e-5,
        use_nervaluate: bool = True,
        nervaluate_mode: str = "partial",
        metric_mode: str = "recall",
        **kwargs
    ) -> None:
        if self.train_records is None:
            raise ValueError("Training dataset not set. Use set_train_dataset() first.")

        self._late_initialize()

        train_dataset = PIIDataset(
            self.train_records, self.tokenizer, self.label_to_id, self.max_length
        )
        
        eval_dataset = None
        if self.val_records:
            eval_dataset = PIIDataset(
                self.val_records, self.tokenizer, self.label_to_id, self.max_length
            )
        
        data_collator = DataCollatorForTokenClassification(
            tokenizer=self.tokenizer,
            padding=True,
        )
        
        metric_for_best_model = None
        if eval_dataset:
            if use_nervaluate and NERVALUATE_AVAILABLE:
                metric_for_best_model = metric_mode
            else:
                metric_for_best_model = metric_mode

        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size * 2,
            learning_rate=learning_rate,
            warmup_steps=100,
            weight_decay=0.01,
            logging_dir=f"{output_dir}/logs",
            logging_steps=50,
            eval_strategy="steps" if eval_dataset else "no",
            eval_steps=250 if eval_dataset else None,
            save_strategy="steps",
            save_steps=250,
            load_best_model_at_end=True if eval_dataset else False,
            metric_for_best_model=metric_for_best_model,
            greater_is_better=True,
            report_to="none",
            fp16=torch.cuda.is_available(),
            **kwargs,
        )
        
        compute_metrics_fn = None
        if eval_dataset:
            if use_nervaluate and NERVALUATE_AVAILABLE:
                compute_metrics_fn = lambda eval_pred: self._compute_metrics_ner(eval_pred, mode=nervaluate_mode)
            else:
                compute_metrics_fn = self._compute_metrics
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics_fn,
        )
        
        print(f"Starting training for {epochs} epochs...")
        trainer.train()
        
        trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        print(f"Training complete! Model saved to {output_dir}")
    
    def predict(self, records: List[DatasetRecord]) -> List[DatasetRecord]:
        if not records:
            return []
        
        self._late_initialize()
        
        if self.pipeline is None:
            self.pipeline = pipeline(
                "token-classification",
                model=self.model,
                tokenizer=self.tokenizer,
                device=self.device,
                aggregation_strategy="simple",
            )
        
        results = []
        
        for record in records:
            if self.use_chunking and self.chunker is not None:
                aggregator = SpanMergeAggregator()
                
                def detect(text):
                    preds = self.pipeline(text)
                    return [
                        {
                            "start": p["start"],
                            "end": p["end"],
                            "label": p["entity_group"],
                            "score": p["score"]
                        }
                        for p in preds
                    ]
                
                span_dicts = process_with_chunking(record.text, self.chunker, detect, aggregator)
                spans = [
                    TextAnnotation(
                        start=s["start"],
                        end=s["end"],
                        label=s["label"],
                        text=record.text[s["start"]:s["end"]],
                        confidence=s["score"],
                    )
                    for s in span_dicts
                ]
            else:
                predictions = self.pipeline(record.text)
                spans = []
                for pred in predictions:
                    spans.append(
                        TextAnnotation(
                            start=pred["start"],
                            end=pred["end"],
                            label=pred["entity_group"],
                            text=record.text[pred["start"]:pred["end"]],
                            confidence=pred["score"],
                        )
                    )
            
            new_record = DatasetRecord(
                text=record.text,
                uid=record.uid,
                name=record.name,
                spans=spans,
                utilities=record.utilities.copy(),
                metadata={**record.metadata, "predicted": True},
            )
            results.append(new_record)
        
        print(f"✓ Predicted spans for {len(records)} records")
        return results
    
    def evaluate(self, records: List[DatasetRecord], use_nervaluate: bool = True, modes: Optional[List[str]] = None) -> Dict[str, Any]:
        if not records:
            raise ValueError("No records provided for evaluation")

        self._late_initialize()

        predicted_records = self.predict(records)

        all_true_labels = []
        all_pred_labels = []
        
        for true_record, pred_record in zip(records, predicted_records):
            true_labels = self._spans_to_char_labels(
                true_record.text, true_record.spans or []
            )
            pred_labels = self._spans_to_char_labels(
                pred_record.text, pred_record.spans or []
            )
            
            all_true_labels.extend(true_labels)
            all_pred_labels.extend(pred_labels)
        
        unique_entity_types = sorted(list(set([
            label[2:] for label in self.labels 
            if label.startswith('B-')
        ])))
        
        unique_bio_labels = [l for l in self.labels if l != "O"]
        
        metrics = {}

        if use_nervaluate and NERVALUATE_AVAILABLE:
            if modes is None:
                modes = ["strict", "partial", "exact"]
            
            true_labels_seq = []
            pred_labels_seq = []
            
            for true_record, pred_record in zip(records, predicted_records):
                true_bio = self._spans_to_token_labels(true_record.text, true_record.spans or [])
                pred_bio = self._spans_to_token_labels(pred_record.text, pred_record.spans or [])
                true_labels_seq.append(true_bio)
                pred_labels_seq.append(pred_bio)
            
            true_collected = [collect_named_entities(msg) for msg in true_labels_seq]
            pred_collected = [collect_named_entities(msg) for msg in pred_labels_seq]
            
            evaluator = Evaluator(true_collected, pred_collected, unique_entity_types)
            results_dict = evaluator.evaluate()
            
            overall_results = results_dict.get("overall", {})
            entity_results = results_dict.get("entities", {})
            
            for mode in modes:
                if mode in overall_results:
                    result = overall_results[mode]
                    metrics[f"{mode}_precision"] = float(result.precision)
                    metrics[f"{mode}_recall"] = float(result.recall)
                    metrics[f"{mode}_f1"] = float(result.f1)
            
            if entity_results:
                metrics["per_category"] = {}
                for mode in modes:
                    metrics["per_category"][mode] = {}
                    for tag in unique_entity_types:
                        if tag in entity_results and mode in entity_results[tag]:
                            result = entity_results[tag][mode]
                            metrics["per_category"][mode][tag] = {
                                "precision": float(result.precision),
                                "recall": float(result.recall),
                                "f1": float(result.f1),
                                "support": int(result.possible)
                            }
            
            print(f"\nNER Evaluation Results (nervaluate):")
            for mode in modes:
                if f"{mode}_f1" in metrics:
                    print(f"\n  {mode.upper()} mode:")
                    print(f"    Precision: {metrics[f'{mode}_precision']:.4f}")
                    print(f"    Recall:    {metrics[f'{mode}_recall']:.4f}")
                    print(f"    F1 Score:  {metrics[f'{mode}_f1']:.4f}")
                else:
                    print(f"\n  {mode.upper()} mode: No metrics available")
            
            if "per_category" in metrics and metrics["per_category"]:
                print(f"\n  Per-category results (partial mode):")
                if "partial" in metrics["per_category"]:
                    for tag, tag_metrics in metrics["per_category"]["partial"].items():
                        print(f"    {tag}: P={tag_metrics['precision']:.4f}, R={tag_metrics['recall']:.4f}, F1={tag_metrics['f1']:.4f}, Support={tag_metrics['support']}")
                else:
                    print(f"    No partial mode results")
            else:
                print(f"\n  No per-category results available")
        
        else:
            precision, recall, f1, support = precision_recall_fscore_support(
                all_true_labels,
                all_pred_labels,
                labels=unique_bio_labels,
                average="micro",
                zero_division=0,
            )
            
            report = classification_report(
                all_true_labels,
                all_pred_labels,
                labels=unique_bio_labels,
                zero_division=0,
                output_dict=True,
            )
            
            metrics = {
                "precision": float(precision) if precision is not None else 0.0,
                "recall": float(recall) if recall is not None else 0.0,
                "f1": float(f1) if f1 is not None else 0.0,
                "support": int(support) if support is not None else 0,
                "detailed_report": report,
            }
            
            print(f"\nEvaluation Results:")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall:    {recall:.4f}")
            print(f"  F1 Score:  {f1:.4f}")
            print(f"  Support:   {support}")
        
        return metrics
    
    def _compute_metrics(self, eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=2)
        
        true_predictions = []
        true_labels = []
        
        for pred_seq, label_seq in zip(predictions, labels):
            for pred, label in zip(pred_seq, label_seq):
                if label != -100:
                    true_predictions.append(self.id_to_label[pred])
                    true_labels.append(self.id_to_label[label])
        
        unique_bio_labels = [l for l in self.labels if l != "O"]
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels,
            true_predictions,
            labels=unique_bio_labels,
            average="micro",
            zero_division=0,
        )
        
        return {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        }
    
    def _get_preds_and_labels_for_nervaluate(self, predictions, labels):
        predictions = np.argmax(predictions, axis=2)
        
        true_preds = []
        true_labels = []
        
        for pred_batch, label_batch in zip(predictions, labels):
            pred_seq = []
            label_seq = []
            for pred, label in zip(pred_batch, label_batch):
                if label != -100:
                    pred_seq.append(self.id_to_label[pred])
                    label_seq.append(self.id_to_label[label])
            true_preds.append(pred_seq)
            true_labels.append(label_seq)
        
        return true_preds, true_labels
    
    def _compute_metrics_ner(self, eval_pred, mode="partial"):
        if not NERVALUATE_AVAILABLE:
            return self._compute_metrics(eval_pred)
        
        from nervaluate import collect_named_entities
        
        predictions, labels = eval_pred
        true_preds, true_labels = self._get_preds_and_labels_for_nervaluate(predictions, labels)
        
        unique_entity_types = sorted(list(set([
            label[2:] for label in self.labels 
            if label.startswith('B-')
        ])))
        
        true_collected = [collect_named_entities(msg) for msg in true_labels]
        pred_collected = [collect_named_entities(msg) for msg in true_preds]
        
        evaluator = Evaluator(true_collected, pred_collected, unique_entity_types)
        results_dict = evaluator.evaluate()
        
        overall_results = results_dict.get("overall", {})
        
        metrics = {}
        if mode in overall_results:
            result = overall_results[mode]
            metrics["precision"] = float(result.precision)
            metrics["recall"] = float(result.recall)
            metrics["f1"] = float(result.f1)
            metrics[f"{mode}_precision"] = metrics["precision"]
            metrics[f"{mode}_recall"] = metrics["recall"]
            metrics[f"{mode}_f1"] = metrics["f1"]
        
        return metrics
    
    def _spans_to_char_labels(
        self, text: str, spans: List[TextAnnotation]
    ) -> List[str]:
        labels = ["O"] * len(text)
        
        for span in sorted(spans, key=lambda s: s.start):
            if span.start >= len(text):
                continue
            
            label = span.label or "ENTITY"
            if label.startswith("B-") or label.startswith("I-"):
                label = label[2:]
            
            labels[span.start] = f"B-{label}"
            for i in range(span.start + 1, min(span.end, len(text))):
                labels[i] = f"I-{label}"
        
        return labels

    def _spans_to_token_labels(
        self, text: str, spans: List[TextAnnotation]
    ) -> List[str]:
        tokens = text.split()
        labels = ["O"] * len(tokens)
        
        char_to_token = []
        current_pos = 0
        for token_idx, token in enumerate(tokens):
            token_start = text.find(token, current_pos)
            if token_start == -1:
                continue
            token_end = token_start + len(token)
            for char_idx in range(token_start, token_end):
                char_to_token.append(token_idx)
            current_pos = token_end
        
        for span in sorted(spans, key=lambda s: s.start):
            if span.start >= len(char_to_token):
                continue
            
            label = span.label or "ENTITY"
            if label.startswith("B-") or label.startswith("I-"):
                label = label[2:]
            
            token_start = char_to_token[span.start] if span.start < len(char_to_token) else None
            token_end = char_to_token[span.end - 1] if span.end > 0 and span.end - 1 < len(char_to_token) else None
            
            if token_start is not None:
                labels[token_start] = f"B-{label}"
                if token_end is not None and token_end > token_start:
                    for token_idx in range(token_start + 1, token_end + 1):
                        if token_idx < len(labels):
                            labels[token_idx] = f"I-{label}"
        
        return labels


class PIIDataset(Dataset):
    def __init__(
        self,
        records: List[DatasetRecord],
        tokenizer,
        label_to_id: Dict[str, int],
        max_length: int = 512,
    ):
        self.records = records
        self.tokenizer = tokenizer
        self.label_to_id = label_to_id
        self.max_length = max_length
    
    def __len__(self):
        return len(self.records)
    
    def __getitem__(self, idx):
        record = self.records[idx]
        text = record.text
        spans = record.spans or []
        
        tokenized = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_offsets_mapping=True,
        )
        
        char_labels = self._spans_to_char_labels(text, spans)
        
        token_labels = []
        for start_char, end_char in tokenized["offset_mapping"]:
            if start_char == 0 and end_char == 0:
                token_labels.append(-100)
            else:
                if start_char < len(char_labels):
                    label = char_labels[start_char]
                    token_labels.append(self.label_to_id.get(label, self.label_to_id["O"]))
                else:
                    token_labels.append(self.label_to_id["O"])
        
        return {
            "input_ids": torch.tensor(tokenized["input_ids"]),
            "attention_mask": torch.tensor(tokenized["attention_mask"]),
            "labels": torch.tensor(token_labels),
        }
    
    def _spans_to_char_labels(
        self, text: str, spans: List[TextAnnotation]
    ) -> List[str]:
        labels = ["O"] * len(text)
        
        for span in sorted(spans, key=lambda s: s.start):
            if span.start >= len(text):
                continue
            
            label = span.label or "ENTITY"
            if label.startswith("B-") or label.startswith("I-"):
                label = label[2:]
            
            labels[span.start] = f"B-{label}"
            for i in range(span.start + 1, min(span.end, len(text))):
                labels[i] = f"I-{label}"
        
        return labels