from typing import Dict, List, Any, Optional
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
        model_class_name: str,
        model_config: Dict[str, Any],
        tri_model_path: Optional[str] = None,
        tri_max_length: int = 512,
        tri_device: str = "auto",
        dataset_name: str = None,
    ):
        super().__init__()
        self.dataset_records = dataset_records
        self.train_records = train_records
        self.model_class_name = model_class_name
        self.model_config = model_config
        self.tri_model_path = tri_model_path
        self.tri_max_length = tri_max_length
        self.tri_device = tri_device
        self.dataset_name = dataset_name
        
        self.tri_detector = None
        self.models: Dict[Any, Any] = {}
        self.original_ranks: Optional[Dict[str, int]] = None
        self.capabilities = get_capabilities(model_class_name)
    
    def setup(self):
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
        
        predictions = self.tri_detector.predict(self.dataset_records)
        self.original_ranks = self._compute_ranks(predictions)
        
        super().setup()
    
    def _create_model(self, parameter_value: Any):
        if parameter_value in self.models:
            return self.models[parameter_value]
        
        model_class = MODEL_REGISTRY[self.model_class_name]
        
        if self.capabilities.requires_dataset:
            model_instance = model_class(
                dataset_records=self.dataset_records,
                **self.model_config
            )
        else:
            model_instance = model_class(**self.model_config)
        
        self.models[parameter_value] = model_instance
        return model_instance
    
    def run(self, parameter_values: List[Any]) -> ExperimentResult:
        if not self.setup_complete:
            raise RuntimeError("Must call setup() before run()")
        
        results = []
        
        for param_value in parameter_values:
            model = self._create_model(param_value)
            
            if self.capabilities.requires_dataset:
                anonymized_records = self._anonymize_requires_idx(model, param_value)
            else:
                anonymized_records = self._anonymize_text_based(model, param_value)
            
            predictions = self.tri_detector.predict(anonymized_records)
            anonymized_ranks = self._compute_ranks(predictions)
            
            rank_changes = self._compute_rank_changes(self.original_ranks, anonymized_ranks)
            privacy_score = self._compute_privacy_score(rank_changes)
            
            results.append({
                "parameter": param_value,
                "privacy_score": privacy_score,
                "rank_changes": rank_changes,
                "anonymized_ranks": anonymized_ranks,
            })
        
        avg_privacy = np.mean([r["privacy_score"] for r in results])
        
        return ExperimentResult(
            score=avg_privacy,
            metrics={
                "average_privacy_score": avg_privacy,
                "results_per_parameter": results,
            },
            metadata={
                "model": self.model_class_name,
                "num_records": len(self.dataset_records),
                "parameters": parameter_values,
            }
        )
    
    def cleanup(self):
        if self.tri_detector is not None:
            del self.tri_detector.model
            self.tri_detector = None
        for model in self.models.values():
            del model
        self.models.clear()
        super().cleanup()
    
    def _anonymize_requires_idx(self, model, param_value: Any) -> List[DatasetRecord]:
        idx_to_record = {i: record for i, record in enumerate(self.dataset_records)}
        indices = list(idx_to_record.keys())
        
        from dp.methods.registry import is_k_anonymizer
        model_class = MODEL_REGISTRY[self.model_class_name]
        
        anonymized = []
        for idx, result in zip(indices, results[0]):
            record = idx_to_record[idx]
            anonymized.append(DatasetRecord(
                uid=record.uid,
                text=result[0],
                name=record.name,
            ))
        return anonymized
    
    def _anonymize_text_based(self, model, param_value: Any) -> List[DatasetRecord]:
        texts = [record.text for record in self.dataset_records]
        
        from dp.methods.registry import is_dp_anonymizer
        model_class = MODEL_REGISTRY[self.model_class_name]
        
        anonymized = []
        for record, result in zip(self.dataset_records, results[0]):
            anonymized.append(DatasetRecord(
                uid=record.uid,
                text=result[0],
                name=record.name,
            ))
        return anonymized
    
    def _compute_ranks(self, predictions: Dict[str, Dict[str, float]]) -> Dict[str, int]:
        ranks = {}
        for uid, prob_dict in predictions.items():
            sorted_labels = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)
            true_label = None
            for record in self.dataset_records:
                if record.uid == uid:
                    true_label = f"LABEL_{self.tri_detector.name_to_label[record.name]}"
                    break
            
            if true_label:
                for rank, (label, _) in enumerate(sorted_labels, start=1):
                    if label == true_label:
                        ranks[uid] = rank
                        break
        
        return ranks
    
    def _compute_rank_changes(
        self,
        original_ranks: Dict[str, int],
        anonymized_ranks: Dict[str, int]
    ) -> Dict[str, int]:
        changes = {}
        for uid in original_ranks:
            if uid in anonymized_ranks:
                changes[uid] = anonymized_ranks[uid] - original_ranks[uid]
        return changes
    
    def _compute_privacy_score(self, rank_changes: Dict[str, int]) -> float:
        if not rank_changes:
            return 0.0
        
        changes_list = list(rank_changes.values())
        positive_changes = [c for c in changes_list if c > 0]
        
        if not positive_changes:
            return 0.0
        
        avg_positive_change = np.mean(positive_changes)
        pct_improved = len(positive_changes) / len(changes_list)
        
        privacy_score = avg_positive_change * pct_improved
        
        return float(privacy_score)
