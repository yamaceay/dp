from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedShuffleSplit

from experiments import Experiment, ExperimentResult
from dp.loaders.base import DatasetRecord

def split_indices(size: int, test_size: float, random_state: int, labels: Sequence[str]) -> Tuple[np.ndarray, np.ndarray]:
    if size == 0:
        raise ValueError("Empty dataset")
    if size < 2:
        raise ValueError("At least two records required")
    if test_size <= 0 or test_size >= 1:
        raise ValueError("test_size must be in (0, 1)")
    indices = np.arange(size)
    unique_labels = np.unique(labels)
    if unique_labels.size == 1:
        rng = np.random.default_rng(random_state)
        rng.shuffle(indices)
        split = int(size * (1 - test_size))
        return indices[:split], indices[split:]

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    try:
        train_idx, test_idx = next(splitter.split(indices, labels))
    except ValueError:
        rng = np.random.default_rng(random_state)
        rng.shuffle(indices)
        split = int(size * (1 - test_size))
        if split == 0 or split == size:
            split = max(1, size - 1)
        train_idx = indices[:split]
        test_idx = indices[split:]
    return train_idx, test_idx

class DownstreamModel:
    @abstractmethod
    def clone(self) -> DownstreamModel:
        raise NotImplementedError()

    @abstractmethod
    def fit(self, x: Any, y: Any) -> None:
        raise NotImplementedError()

    @abstractmethod
    def predict(self, x: Any) -> Any:
        raise NotImplementedError()
    
    @abstractmethod
    def score(self, x: Any, y: Any) -> Dict[str, float]:
        raise NotImplementedError()
    
    @abstractmethod
    def cleanup(self) -> None:
        raise NotImplementedError()

class UtilityTarget:
    class Mode(Enum):
        BINARY = "binary"
        NOMINAL = "nominal"
        ORDINAL = "ordinal"
        CARDINAL = "cardinal"
    
    def __init__(self,
                 mode: Optional[str] = None,
                 getter: Optional[Callable[[Any], Any]] = None,
                 source: Optional[str] = None):
        self.mode = None
        if mode is not None and self._mode_valid(mode):
            self.mode = UtilityTarget.Mode(mode)
        self.getter = getter
        self.source = source

    def _mode_valid(self, mode: str) -> bool:
        return mode in UtilityTarget.Mode._value2member_map_

class TextUtilityExperiment(Experiment):
    def __init__(
        self,
        vectorizer: Optional[TfidfVectorizer] = None,
        test_size: float = 0.2,
        random_state: int = 42,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.evaluation_texts: Dict[str, Dict[str, str]] = {}
        self.test_size = test_size
        self.random_state = random_state

        self.target: Optional[UtilityTarget] = None
        self.records: List[DatasetRecord] = []
        self.max_records: Optional[int] = None
        self.record_info: Dict[str, Dict[str, Any]] = {}

        self._texts: List[str] = []
        self._labels: List[str] = []
        self._keys: List[str] = []
        self._train_idx: Optional[np.ndarray] = None
        self._test_idx: Optional[np.ndarray] = None
        self._baseline_score_dict: Optional[float] = None
        self._selected_records: List[DatasetRecord] = []
        self._vectorizer = vectorizer or TfidfVectorizer(ngram_range=(1, 2), max_features=20000, min_df=2)
        self._model: Optional[DownstreamModel] = None
        self._indexed = True

    def set_data(self, records: List[DatasetRecord]) -> None:
        self.records = list(records)
        assert len(self.records) > 0, "records cannot be empty"
        uids_valid = all(record.uid is not None for record in self.records)
        uids_not_valid = all(record.uid is None for record in self.records)
        assert uids_valid or uids_not_valid, "Either all records must have uid or none of them should have uid"
        if uids_not_valid:
            self._indexed = False
            for idx, record in enumerate(self.records):
                record.uid = f"record_{idx+1}"
        exists_empty_text = any(not record.text for record in self.records)
        if exists_empty_text:
            print("Warning: Some records have empty text and will be ignored")
        self._selected_records = [record for record in self.records if record.text]
        self._keys = [record.uid for record in self._selected_records]
        self._texts = [record.text for record in self._selected_records]

    def split_data(self) -> None:
        self._train_idx, self._test_idx = split_indices(len(self._texts), self.test_size, self.random_state, self._labels)
        self.train_texts = [self._texts[i] for i in self._train_idx]
        self.test_texts = [self._texts[i] for i in self._test_idx]
        self.train_labels = [self._labels[i] for i in self._train_idx]
        self.test_labels = [self._labels[i] for i in self._test_idx]

    def get_label(self, record: DatasetRecord) -> Any:
        assert self.target is not None, "target must be set before getting labels"
        assert self.target.getter is not None, "target getter function must be specified"
        value = self.target.getter(record)
        assert value is not None, f"Transformed label is None for record {record.uid}"
        return value

    def preprocess_data(self) -> None:
        self.target = target
        self._labels = [self.get_label(record) for record in self._selected_records]
        self.record_info = {key: {"index": idx + 1, "label": self._labels[idx]} for idx, key in enumerate(self._keys)}

    def _clone_vectorizer(self) -> TfidfVectorizer:
        assert self._vectorizer is not None, "Vectorizer must be specified"
        return TfidfVectorizer(
            ngram_range=self._vectorizer.ngram_range,
            max_features=self._vectorizer.max_features,
            min_df=self._vectorizer.min_df,
        )
    
    def _clone_model(self) -> DownstreamModel:
        assert self._model is not None, "Model must be specified"
        return self._model.clone()  # type: ignore[return-value]        

    def fit(self, texts: Sequence[str], labels: Sequence[str]) -> None:
        vectorizer = self._clone_vectorizer()
        model = self._clone_model()
        x = vectorizer.fit_transform(texts)
        model.fit(x, labels)
        self._vectorizer = vectorizer
        self._model = model
    
    def score(self, texts: Sequence[str], labels: Sequence[str]) -> Dict[str, float]:
        assert self._vectorizer is not None, "Vectorizer must be fitted before scoring"
        assert self._model is not None, "Model must be fitted before scoring"
        x = self._vectorizer.transform(texts)
        predictions = self._model.predict(x)
        return self._model.score(labels, predictions)

    def setup(self, 
              target: UtilityTarget,
              records: list[DatasetRecord], 
              model: DownstreamModel,
              **kwargs) -> None:  # type: ignore[override]
        
        self.set_data(records)

        self.target = target
        self.preprocess_data()

        self._model = model
        assert self._model is not None, "Model must be specified"
        self.fit(self.train_texts, self.train_labels)
        self._baseline_score_dict = self.score(self.test_texts, self.test_labels)
        
        super().setup(**kwargs)

    def score_difference(self, baseline: Dict[str, float], current: Dict[str, float]) -> Dict[str, float]:
        score_diffs = {}
        for metric_name in baseline:
            if metric_name in current:
                score_diffs[f"{metric_name}_drop"] = baseline[metric_name] - current[metric_name]
        if not score_diffs:
            raise ValueError("Unsupported score difference computation")
        return score_diffs

    def run(self, evaluation_texts: Dict[str, Dict[str, str]], **kwargs: Any) -> ExperimentResult:  # type: ignore[override]
        if not self._model or not self.target:
            raise RuntimeError("setup must be completed before run")
        if not self._indexed:
            raise RuntimeError("Records must have valid uid for evaluation")
        evaluations: Dict[str, Dict[str, Any]] = {}
        evaluation_texts_selected = {}
        for name, texts_dict in evaluation_texts.items():
            filtered_texts = {key: text for key, text in texts_dict.items() if any(record.uid == key for record in self._selected_records)}
            evaluation_texts_selected[name] = filtered_texts

        losses: List[float] = []
        train_total = len(self._train_idx)
        test_total = len(self._test_idx)

        for name, texts_map in sorted(self.evaluation_texts_selected.items()):
            eval_keys = list(texts_map.keys())
            eval_texts = [texts_map[key] for key in eval_keys]
            eval_labels = [
                self.get_label(next(record for record in self._selected_records if record.uid == key))
                for key in eval_keys
            ]
            
            evaluation: Dict[str, Any] = {
                "train_matched": len(matched_train_idx),
                "test_matched": len(matched_test_idx),
                "train_total": train_total,
                "test_total": test_total,
                "available": len(texts_map),
                "valid": False,
            }

            if matched_train_idx and matched_test_idx:
                score_dict = self.score(eval_texts, eval_labels)
                loss_dict = self.score_difference(self._baseline_score_dict, score_dict)
                evaluation.update(**score_dict, **loss_dict)
                evaluation["valid"] = True
            evaluations[name] = evaluation
            
        metrics = {
            "metadata": {
                "train_size": train_total,
                "test_size": test_total,
            },
            "baseline": self._baseline_score_dict,
            "evaluations": evaluations,
            "records": self.record_info,
        }
        return ExperimentResult(score=None, metrics=metrics)

    def cleanup(self) -> None:  # type: ignore[override]
        self._texts = []
        self._labels = []
        self._keys = []
        self._train_idx = None
        self._test_idx = None
        self.record_info = {}
        self._vectorizer = None
        self._model.cleanup()
        self._model = None
        super().cleanup()