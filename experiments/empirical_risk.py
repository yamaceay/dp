from typing import List, Dict, Any
import numpy as np

from dp.experiments import Experiment, ExperimentResult
from dp.loaders.base import DatasetRecord
from dp.utils.tri_detector import TRIDetector
from dp.methods.registry import MODEL_REGISTRY
from dp.methods.constants import get_capabilities


class ReidentificationRiskExperiment(Experiment):
    def __init__(
        self,
        dataset_records: List[DatasetRecord],
        train_records: List[DatasetRecord],
        model_name: str,
        model_config: Dict[str, Any],
        tri_model_path: str = None,
        tri_max_length: int = 512,
        tri_device: str = "auto",
        dataset_name: str = None,
    ):
        super().__init__()
        self.dataset_records = dataset_records
        self.train_records = train_records
        self.model_name = model_name
        self.model_config = model_config
        self.tri_model_path = tri_model_path
        self.tri_max_length = tri_max_length
        self.tri_device = tri_device
        self.dataset_name = dataset_name
        
        self.tri_detector = None
        self.anonymizer = None
        self.original_ranks = None
        self.capabilities = get_capabilities(model_name)
        self.name_to_label = None
    
    def setup(self):
        self._setup_tri_detector()
        self._setup_anonymizer()
        self._compute_original_ranks()
        super().setup()
    
    def _setup_tri_detector(self):
        self.tri_detector = TRIDetector(
            dataset_name=self.dataset_name,
            max_length=self.tri_max_length,
            device=self.tri_device
        )
        
        if self.tri_model_path:
            self.tri_detector.load(self.tri_model_path)
        else:
            self.tri_detector.set_train_dataset(self.train_records)
            self.tri_detector.train()
        
        self.name_to_label = self.tri_detector.name_to_label
    
    def _setup_anonymizer(self):
        model_class = MODEL_REGISTRY[self.model_name]
        
        if self.capabilities.must_use_dataset:
            self.anonymizer = model_class(
                dataset_records=self.dataset_records,
                **self.model_config
            )
        else:
            self.anonymizer = model_class(**self.model_config)
        
        if self.capabilities.requires_k:
            import tempfile
            import os
            from dp.utils.explainer import GreedyExplainer
            
            temp_dir = tempfile.mkdtemp()
            tri_model_path = os.path.join(temp_dir, "tri_model")
            self.tri_detector.model.save_pretrained(tri_model_path)
            self.tri_detector.tokenizer.save_pretrained(tri_model_path)
            
            explainer = GreedyExplainer(model_name=tri_model_path, device=str(self.tri_detector.device))
            self.anonymizer.set_scoring_strategy(explainer)
    
    def _compute_original_ranks(self):
        predictions = self.tri_detector.predict(self.dataset_records)
        self.original_ranks = self._extract_ranks(predictions)
    
    def _extract_ranks(self, predictions: Dict[str, Dict[str, float]]) -> Dict[str, int]:
        uid_to_name = {r.uid: r.name for r in self.dataset_records}
        ranks = {}
        
        for uid, probs in predictions.items():
            name = uid_to_name[uid]
            if name not in self.name_to_label:
                continue
            
            true_label = self.name_to_label[name]
            sorted_labels = sorted(probs.items(), key=lambda x: x[1], reverse=True)
            
            for rank, (label_str, _) in enumerate(sorted_labels, start=1):
                label_id = int(label_str.split("_")[1])
                if label_id == true_label:
                    ranks[uid] = rank
                    break
        
        return ranks
    
    def run(self, parameter_values: List[Any]) -> ExperimentResult:
        if not self.setup_complete:
            raise RuntimeError("Must call setup() before run()")
        
        results = []
        
        for param_value in parameter_values:
            anonymized_records = self._anonymize_with_parameter(param_value)
            predictions = self.tri_detector.predict(anonymized_records)
            anonymized_ranks = self._extract_ranks(predictions)
            
            rank_changes = self._compute_rank_changes(anonymized_ranks)
            privacy_score = self._compute_privacy_score(rank_changes)
            
            results.append({
                "parameter": param_value,
                "privacy_score": privacy_score,
                "mean_rank_change": np.mean(list(rank_changes.values())),
                "median_rank_change": np.median(list(rank_changes.values())),
            })
        
        return ExperimentResult(
            score=np.mean([r["privacy_score"] for r in results]),
            metrics={
                "results": results,
                "original_ranks": self.original_ranks,
            },
            metadata={
                "model": self.model_name,
                "num_records": len(self.dataset_records),
                "parameters": parameter_values,
            }
        )
    
    def _anonymize_with_parameter(self, param_value: Any) -> List[DatasetRecord]:
        builder = self.anonymizer.builder()
        
        if self.capabilities.requires_k:
            indices = list(range(len(self.dataset_records)))
            builder.with_indices(indices).with_ks([param_value])
            results = builder.anonymize()
            return self._flatten_k_results(results)
        
        elif self.capabilities.requires_epsilon:
            texts = [r.text for r in self.dataset_records]
            builder.with_texts(texts).with_epsilons([param_value])
            results = builder.anonymize()
            return self._flatten_epsilon_results(results)
        
        elif self.capabilities.must_use_dataset:
            indices = list(range(len(self.dataset_records)))
            builder.with_indices(indices)
            results = builder.anonymize()
            return self._flatten_dataset_results(results)
        
        else:
            texts = [r.text for r in self.dataset_records]
            builder.with_texts(texts)
            results = builder.anonymize()
            return self._flatten_simple_results(results)
    
    def _flatten_k_results(self, results: List[List[Any]]) -> List[DatasetRecord]:
        anonymized = []
        for idx, result_list in enumerate(results):
            record = self.dataset_records[idx]
            anonymized.append(DatasetRecord(
                uid=record.uid,
                text=result_list[0].text,
                name=record.name,
            ))
        return anonymized
    
    def _flatten_epsilon_results(self, results: List[List[Any]]) -> List[DatasetRecord]:
        anonymized = []
        for idx, result_list in enumerate(results):
            record = self.dataset_records[idx]
            anonymized.append(DatasetRecord(
                uid=record.uid,
                text=result_list[0].text,
                name=record.name,
            ))
        return anonymized
    
    def _flatten_dataset_results(self, results: List[Any]) -> List[DatasetRecord]:
        anonymized = []
        for idx, result in enumerate(results):
            record = self.dataset_records[idx]
            anonymized.append(DatasetRecord(
                uid=record.uid,
                text=result.text,
                name=record.name,
            ))
        return anonymized
    
    def _flatten_simple_results(self, results: List[Any]) -> List[DatasetRecord]:
        anonymized = []
        for idx, result in enumerate(results):
            record = self.dataset_records[idx]
            anonymized.append(DatasetRecord(
                uid=record.uid,
                text=result.text,
                name=record.name,
            ))
        return anonymized
    
    def _compute_rank_changes(self, anonymized_ranks: Dict[str, int]) -> Dict[str, int]:
        changes = {}
        for uid in self.original_ranks:
            if uid in anonymized_ranks:
                changes[uid] = anonymized_ranks[uid] - self.original_ranks[uid]
        return changes
    
    def _compute_privacy_score(self, rank_changes: Dict[str, int]) -> float:
        if not rank_changes:
            return 0.0
        
        changes = list(rank_changes.values())
        positive_changes = [c for c in changes if c > 0]
        
        if not positive_changes:
            return 0.0
        
        return float(np.mean(positive_changes) * len(positive_changes) / len(changes))
    
    def cleanup(self):
        if self.tri_detector and hasattr(self.tri_detector, 'model'):
            del self.tri_detector.model
        self.tri_detector = None
        self.anonymizer = None
        super().cleanup()
