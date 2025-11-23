from typing import Optional, Dict, Any
from pathlib import Path
from datetime import datetime
import json
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
        "spacy", "presidio", "baroud", "risk",  "manual",
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
        print("# Annotations:", len(result.spans or []))
        print("Metadata:", result.metadata)


class JsonlOutputHandler(OutputHandler):
    def __init__(self, base_path: str = "outputs", timestamp: Optional[str] = None):
        self.base_path = base_path
        self.timestamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
        self._streams: Dict[str, Any] = {}
        self._paths: Dict[str, Path] = {}

    def output(self, result: AnonymizationResult, dataset: str, model: str, task_id: int, **kwargs):
        idx = kwargs.get("idx", None)
        variant_key = self._derive_variant_key(result)
        stream = self._ensure_stream(dataset, model, variant_key, task_id)

        record = {
            "idx": idx,
            "text": result.text,
        }
        
        if result.spans:
            record["spans"] = [self._serialize_annotation_minimal(ann) for ann in result.spans]
        
        if result.metadata:
            record["metadata"] = result.metadata
        
        stream.write(json.dumps(record, ensure_ascii=False, cls=NumpyEncoder) + '\n')
        stream.flush()
    
    def close(self):
        for key, stream in self._streams.items():
            stream.close()
            path = self._paths.get(key)
            if path is not None:
                print(f"Output written to: {path}")
        self._streams.clear()
        self._paths.clear()

    def _get_output_dir(self, dataset: str, model: str) -> Path:
        pattern = OUTPUT_STRUCTURE.get(model, f"outputs/{{dataset}}/{model}")
        path_str = pattern.format(dataset=dataset)
        return Path(path_str)

    def _ensure_stream(self, dataset: str, model: str, variant_key: str, task_id: int):
        if variant_key in self._streams:
            return self._streams[variant_key]
        output_dir = self._get_output_dir(dataset, model)
        output_dir.mkdir(parents=True, exist_ok=True)
        suffix = f"_{variant_key}" if variant_key else ""
        sanitized_suffix = suffix.replace(" ", "_")
        path = output_dir / f"{self.timestamp}{sanitized_suffix}_{task_id}.jsonl"
        handle = open(path, 'w', encoding='utf-8')
        self._streams[variant_key] = handle
        self._paths[variant_key] = path
        return handle

    def _derive_variant_key(self, result: AnonymizationResult) -> str:
        metadata = result.metadata or {}
        param = metadata.get("_grid_param")
        value = metadata.get("_grid_value") if param else None
        if param is None:
            for key in ("pii_confidence", "risk_tolerance", "epsilon", "k"):
                if key in metadata and metadata[key] is not None:
                    param = key
                    value = metadata[key]
                    break
        if param is None or value is None:
            return ""
        formatted_value = self._format_variant_value(value)
        return f"{param}-{formatted_value}"

    def _format_variant_value(self, value: Any) -> str:
        if isinstance(value, float):
            return f"{value:.4f}".rstrip("0").rstrip(".")
        return str(value)
    
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
