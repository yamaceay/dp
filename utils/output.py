from typing import Optional, Dict, Any
from pathlib import Path
from datetime import datetime
import json
from dataclasses import asdict
import numpy as np

from dp.methods.anonymizer import AnonymizationResult
from dp.loaders.base import TextAnnotation


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles NumPy types"""
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


OUTPUT_STRUCTURE = {
    model: "outputs/{dataset}" + f"/{model}" for model in [
        "spacy", "presidio", "manual", "baroud",
        "dpmlm", "dpbart", "dpprompt", "dpparaphrase",
        "petre",
    ]
}


class OutputHandler:
    def output(self, result: AnonymizationResult, dataset: str, model: str, **kwargs):
        raise NotImplementedError


class PrintOutputHandler(OutputHandler):
    def output(self, result: AnonymizationResult, dataset: str, model: str, **kwargs):
        print("Anonymized Text:", result.text)
        print("Annotations:", result.spans)
        print("Metadata:", result.metadata)


class JsonlOutputHandler(OutputHandler):
    def __init__(self, base_path: str = "outputs", timestamp: Optional[str] = None):
        self.base_path = base_path
        self.timestamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.jsonl_file = None
        self.jsonl_path = None
    
    def output(self, result: AnonymizationResult, dataset: str, model: str, **kwargs):
        output_dir = self._get_output_dir(dataset, model)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if self.jsonl_file is None:
            self.jsonl_path = output_dir / f"{self.timestamp}.jsonl"
            self.jsonl_file = open(self.jsonl_path, 'w', encoding='utf-8')
        
        idx = kwargs.get("idx", None)
        
        record = {
            "idx": idx,
            "text": result.text,
        }
        
        if result.spans:
            record["spans"] = [self._serialize_annotation_minimal(ann) for ann in result.spans]
        
        if result.metadata:
            record["metadata"] = result.metadata
        
        self.jsonl_file.write(json.dumps(record, ensure_ascii=False, cls=NumpyEncoder) + '\n')
        self.jsonl_file.flush()
    
    def close(self):
        if self.jsonl_file is not None:
            self.jsonl_file.close()
            print(f"Output written to: {self.jsonl_path}")
            self.jsonl_file = None
    
    def _get_output_dir(self, dataset: str, model: str) -> Path:
        pattern = OUTPUT_STRUCTURE.get(model, f"outputs/{{dataset}}/{model}")
        path_str = pattern.format(dataset=dataset)
        return Path(path_str)
    
    def _serialize_annotation_minimal(self, ann: TextAnnotation) -> Dict[str, Any]:
        """Serialize annotation with omitempty pattern, converting NumPy types to native Python"""
        data = {}
        if ann.start is not None:
            data["start"] = int(ann.start) if isinstance(ann.start, (np.integer, np.int32, np.int64)) else ann.start
        if ann.end is not None:
            data["end"] = int(ann.end) if isinstance(ann.end, (np.integer, np.int32, np.int64)) else ann.end
        if ann.label:
            data["label"] = ann.label
        if ann.text:
            data["text"] = ann.text
        if ann.replacement:
            data["replacement"] = ann.replacement
        if ann.confidence is not None:
            # Convert NumPy float types to native Python float
            data["confidence"] = float(ann.confidence) if isinstance(ann.confidence, (np.floating, np.float32, np.float64)) else ann.confidence
        if ann.annotator:
            data["annotator"] = ann.annotator
        if ann.metadata:
            data["metadata"] = self._convert_numpy_types(ann.metadata)
        return data
    
    def _convert_numpy_types(self, obj):
        """Recursively convert NumPy types to native Python types"""
        if isinstance(obj, dict):
            return {k: self._convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(item) for item in obj]
        elif isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return obj


OUTPUT_HANDLER_REGISTRY = {
    "print": PrintOutputHandler,
    "jsonl": JsonlOutputHandler,
}
