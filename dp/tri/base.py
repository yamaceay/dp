from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Sequence, Tuple

import torch
import numpy as np

from dp.bert.classifier import BertClassifierHead
from dp.loaders.base import DatasetRecord
from dp.tri.loaders.base import AttackerDatasetRecord
from dp.utils.chunking import ProbabilityAggregator, TokenAwareChunker, process_with_chunking
from dp.utils.device import resolve_device
from dp.utils.stopwords import strip_stopwords


class TRIDetector:
    def __init__(
        self,
        dataset_name: Optional[str] = None,
        model_name: str = "distilbert-base-uncased",
        max_length: int = 512,
        device: Optional[Union[str, int]] = None,
        use_chunking: bool = True,
        p_agg: str = "avg",
    ):
        if max_length <= 0:
            raise ValueError(f"max_length must be positive, got {max_length}")
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.max_length = int(max_length)
        self.device = torch.device(resolve_device(device))
        self.use_chunking = bool(use_chunking)
        self.p_agg = p_agg
        self.label_to_name: Dict[int, str] = {}
        self.name_to_label: Dict[str, int] = {}
        self.num_labels = 0
        self.classifier: Optional[BertClassifierHead] = None
        self.chunker: Optional[TokenAwareChunker] = None
        self.train_records: Optional[List[DatasetRecord]] = None
        self.eval_records: Optional[List[DatasetRecord]] = None

    def setup(
        self,
        records: List[AttackerDatasetRecord],
        exclude_stopwords: bool = False,
    ) -> None:
        self._set_dataset(records, exclude_stopwords=exclude_stopwords)
        self.build_label_mappings()

    def train(
        self,
        epochs: int = 3,
        batch_size: int = 16,
        learning_rate: float = 5e-5,
        output_dir: Optional[str] = None,
        use_pretraining: bool = False,
        pretraining_epochs: int = 3,
        early_stop_threshold: Optional[float] = None,
        early_stop_patience: Optional[int] = None,
        loss_type: str = "cross_entropy",
        focal_gamma: float = 2.0,
        focal_alpha: Optional[float] = None,
        focal_ignore_pt: Optional[float] = None,
        weight_decay: float = 0.01,
        scheduler_type: str = "linear",
        optimizer_type: str = "adamw",
        warmup_steps: int = 10,
        warmup_ratio: Optional[float] = None,
        init_checkpoint: Optional[str] = None,
    ) -> None:
        if not self.train_records:
            raise ValueError("No training data set. Call setup() first")
        if self.eval_records is None:
            raise ValueError("No evaluation data set. Call setup() first")
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
        if early_stop_patience is not None and early_stop_patience <= 0:
            raise ValueError(f"early_stop_patience must be positive, got {early_stop_patience}")
        if loss_type not in {"cross_entropy", "focal"}:
            raise ValueError(f"Unknown loss_type: {loss_type}")
        if focal_gamma < 0:
            raise ValueError(f"focal_gamma must be >= 0, got {focal_gamma}")
        if focal_ignore_pt is not None and (focal_ignore_pt <= 0.0 or focal_ignore_pt >= 1.0):
            raise ValueError(f"focal_ignore_pt must be in (0, 1), got {focal_ignore_pt}")
        if weight_decay < 0:
            raise ValueError(f"weight_decay must be >= 0, got {weight_decay}")
        if warmup_steps < 0:
            raise ValueError(f"warmup_steps must be >= 0, got {warmup_steps}")
        if warmup_ratio is not None and (warmup_ratio < 0.0 or warmup_ratio >= 1.0):
            raise ValueError(f"warmup_ratio must be in [0, 1), got {warmup_ratio}")

        resolved_output_dir = output_dir
        if resolved_output_dir is None:
            if not self.dataset_name:
                raise ValueError("dataset_name must be set when output_dir is not provided")
            resolved_output_dir = f"models/tri_pipelines/{self.dataset_name}"
        Path(resolved_output_dir).mkdir(parents=True, exist_ok=True)

        self.classifier = BertClassifierHead(
            model_name=self.model_name,
            batch_size=batch_size,
            epochs=epochs,
            encoder_lr=learning_rate,
            device=str(self.device),
            early_stop_threshold=early_stop_threshold,
            early_stop_patience=(early_stop_patience if early_stop_patience is not None else 2),
            loss_type=loss_type,
            focal_gamma=focal_gamma,
            focal_alpha=focal_alpha,
            focal_ignore_pt=focal_ignore_pt,
            use_pretraining=use_pretraining,
            pretraining_epochs=pretraining_epochs,
            pretraining_batch_size=batch_size,
            pretraining_learning_rate=learning_rate,
            optimizer_type=optimizer_type,
            scheduler_type=scheduler_type,
            weight_decay=weight_decay,
            warmup_steps=warmup_steps,
            warmup_ratio=warmup_ratio,
            checkpoint_dir=f"{resolved_output_dir}/finetuning",
            pretraining_output_dir=f"{resolved_output_dir}/pretraining",
            init_checkpoint=init_checkpoint,
        )
        train_texts = [record.text for record in self.train_records]
        train_labels = [record.name for record in self.train_records]
        eval_texts = [record.text for record in self.eval_records]
        eval_labels = [record.name for record in self.eval_records]
        self.classifier.fit(train_texts, train_labels, eval_texts, eval_labels)
        self.classifier.save(resolved_output_dir)
        if self.classifier._label_to_id is None:
            raise ValueError("BertClassifierHead did not produce label mapping")
        self.name_to_label = dict(self.classifier._label_to_id)
        self.label_to_name = {v: k for k, v in self.name_to_label.items()}
        self.num_labels = len(self.name_to_label)
        self._sync_chunker_from_classifier()
        label_mapping_path = Path(resolved_output_dir) / "label_mapping.json"
        with label_mapping_path.open("w", encoding="utf-8") as f:
            json.dump(self.name_to_label, f, indent=2)

    def load(self, model_path: Optional[str] = None) -> None:
        resolved_model_path = model_path
        if resolved_model_path is None:
            if not self.dataset_name:
                raise ValueError("dataset_name must be set when model_path is not provided")
            resolved_model_path = f"models/tri_pipelines/{self.dataset_name}"
        model_dir = Path(resolved_model_path)
        if not model_dir.exists():
            raise ValueError(f"Model path does not exist: {resolved_model_path}")
        checkpoint_path = model_dir / "bert_classifier.pt"
        if not checkpoint_path.exists():
            raise ValueError(
                f"Unsupported TRI checkpoint format at {resolved_model_path}. "
                "Expected BertClassifierHead checkpoint 'bert_classifier.pt'."
            )
        self.classifier = BertClassifierHead(
            model_name=self.model_name,
            batch_size=16,
            device=str(self.device),
            encoder_lr=1e-5,
        )
        self.classifier.load(str(model_dir))
        label_mapping_path = model_dir / "label_mapping.json"
        if not label_mapping_path.exists():
            raise ValueError(f"Label mapping not found: {label_mapping_path}")
        with label_mapping_path.open("r", encoding="utf-8") as f:
            self.name_to_label = json.load(f)
        self.label_to_name = {v: k for k, v in self.name_to_label.items()}
        self.num_labels = len(self.name_to_label)
        self._sync_chunker_from_classifier()

    def predict(self, records: List[DatasetRecord]) -> Dict[str, Dict[str, float]]:
        if not records:
            return {}
        if self.classifier is None:
            raise ValueError("Model not initialized. Train or load a model first.")
        if self.use_chunking and self.chunker is not None:
            aggregator = ProbabilityAggregator(self.p_agg)

            def classify(text: str) -> Dict[str, float]:
                return self.classifier.predict_proba([text])[0]

            predictions: Dict[str, Dict[str, float]] = {}
            for idx, record in enumerate(records):
                key = record.uid or str(idx)
                predictions[key] = process_with_chunking(record.text, self.chunker, classify, aggregator)
            return predictions
        probabilities = self.classifier.predict_proba([record.text for record in records])
        return {
            (record.uid or str(idx)): scores
            for idx, (record, scores) in enumerate(zip(records, probabilities))
        }

    def labels_by_id(self) -> List[Tuple[int, str]]:
        if not self.label_to_name:
            return []
        return sorted(self.label_to_name.items(), key=lambda item: item[0])

    def predict_proba_batch(self, texts: Sequence[str]) -> List[Dict[str, float]]:
        if self.classifier is None:
            raise ValueError("Model not initialized. Train or load a model first.")
        batch = [str(t) for t in texts]
        if self.use_chunking and self.chunker is not None:
            aggregator = ProbabilityAggregator(self.p_agg)

            def classify(text: str) -> Dict[str, float]:
                return self.classifier.predict_proba([text])[0]

            return [process_with_chunking(text, self.chunker, classify, aggregator) for text in batch]
        return self.classifier.predict_proba(batch)

    def predict_proba_matrix(self, texts: Sequence[str]) -> np.ndarray:
        rows = self.predict_proba_batch(texts)
        labels = self.labels_by_id()
        matrix = np.zeros((len(rows), len(labels)), dtype=np.float64)
        for row_idx, scores in enumerate(rows):
            for col_idx, (_, name) in enumerate(labels):
                matrix[row_idx, col_idx] = float(scores.get(name, 0.0))
        return matrix

    def evaluate(self, records: List[DatasetRecord]) -> Dict[str, Any]:
        if not records:
            raise ValueError("records cannot be empty")
        predictions = self.predict(records)
        correct = 0
        total = 0
        for idx, record in enumerate(records):
            if not record.name or record.name not in self.name_to_label:
                continue
            key = record.uid or str(idx)
            scores = predictions.get(key, {})
            if not scores:
                continue
            predicted_name = max(scores.items(), key=lambda item: item[1])[0]
            if predicted_name == record.name:
                correct += 1
            total += 1
        accuracy = correct / total if total else 0.0
        return {"accuracy": accuracy, "correct": correct, "total": total}

    def build_label_mappings(self) -> None:
        if not self.train_records:
            return
        names = sorted(set(record.name for record in self.train_records))
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
            self.label_to_name = {idx: name for name, idx in self.name_to_label.items()}
            return
        self.label_to_name = {idx: name for idx, name in enumerate(names)}
        self.name_to_label = {name: idx for idx, name in self.label_to_name.items()}
        self.num_labels = len(names)

    def _set_dataset(
        self,
        records: List[AttackerDatasetRecord],
        exclude_stopwords: bool,
    ) -> None:
        if not records:
            raise ValueError("Training records cannot be empty")
        self.train_records = []
        self.eval_records = []
        for record in records:
            name_raw = str(record.name).strip()
            train_texts = list(record.train_texts)
            eval_texts = list(record.eval_texts)
            if exclude_stopwords:
                train_texts = [strip_stopwords(text) for text in train_texts]
                eval_texts = [strip_stopwords(text) for text in eval_texts]
            self.train_records.extend(DatasetRecord(text=text, name=name_raw) for text in train_texts)
            self.eval_records.extend(DatasetRecord(text=text, name=name_raw) for text in eval_texts)
        if not self.train_records:
            raise ValueError("No training records built from attacker records")
        if not self.eval_records:
            raise ValueError("No evaluation records built from attacker records")

    def _sync_chunker_from_classifier(self) -> None:
        if not self.use_chunking or self.classifier is None:
            self.chunker = None
            return
        tokenizer = self.classifier._tokenizer
        if tokenizer is None:
            self.chunker = None
            return
        max_tokens = self.max_length - 2 if self.max_length > 2 else 1
        self.chunker = TokenAwareChunker(tokenizer, max_tokens)
