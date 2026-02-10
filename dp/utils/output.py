from typing import Optional, Dict, Any, Mapping, List
from pathlib import Path
from datetime import datetime
import json
import numpy as np

from dp.methods.anonymizer import AnonymizationResult
from dp.loaders.base import TextAnnotation

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles NumPy types"""
    def default(self, o):
        if isinstance(o, (np.integer, np.int32, np.int64)):
            return int(o)
        elif isinstance(o, (np.floating, np.float32, np.float64)):
            return float(o)
        elif isinstance(o, np.ndarray):
            return o.tolist()
        elif isinstance(o, np.bool_):
            return bool(o)
        return super().default(o)


class OutputHandler:
    def output(self, result: AnonymizationResult, dataset: str, model: str, **kwargs):
        raise NotImplementedError


class PrintOutputHandler(OutputHandler):
    def output(self, result: AnonymizationResult, dataset: str, model: str, **kwargs):
        print("Anonymized Text:", result.text)
        print("# Annotations:", len(result.annotations.spans))
        print("Metadata:", result.metadata)


class JsonlOutputHandler(OutputHandler):
    def __init__(self, base_path: str = "outputs", timestamp: Optional[str] = None):
        self.base_path = base_path
        self.timestamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
        self._streams: Dict[str, Any] = {}
        self._paths: Dict[str, Path] = {}

    def output(self, result: AnonymizationResult, dataset: str, model: str, task_id: Optional[int] = None, **kwargs):
        idx = kwargs.get("idx", None)
        unique_name = kwargs.get("unique_name")
        hyperparams = kwargs.get("hyperparams")
        variant_key = self._derive_variant_key_from_hyperparams(hyperparams) or self._derive_variant_key(result)
        stream = self._ensure_stream(dataset, model, variant_key, task_id, unique_name)

        record = {
            "idx": idx,
            "text": result.text,
        }

        annotations: Dict[str, Any] = {}
        if result.annotations.spans:
            annotations["spans"] = [self._serialize_annotation_minimal(ann) for ann in result.annotations.spans]
        if result.annotations.token_edits:
            annotations["token_edits"] = [te.to_dict() for te in result.annotations.token_edits]
        if annotations:
            record["annotations"] = annotations
        
        metadata = result.metadata
        if unique_name is not None:
            metadata = dict(metadata)
            metadata["unique_name"] = unique_name
        if hyperparams and isinstance(hyperparams, Mapping):
            if not isinstance(metadata, dict):
                metadata = dict(metadata)
            metadata["hyperparams"] = self._convert_numpy_types(dict(hyperparams))
        record["metadata"] = metadata
        
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
        pattern = f"{self.base_path}/{{dataset}}/{model}"
        path_str = pattern.format(dataset=dataset)
        return Path(path_str)

    def _ensure_stream(self, dataset: str, model: str, variant_key: str, task_id: Optional[int], unique_name: Optional[str]):
        stream_key = (variant_key, unique_name)
        if stream_key in self._streams:
            return self._streams[stream_key]
        output_dir = self._get_output_dir(dataset, model)
        output_dir.mkdir(parents=True, exist_ok=True)
        suffix = ""
        if unique_name:
            suffix += f"_{unique_name}"
        if variant_key:
            # variant_key can be URL-param style like '?rho=090&lambda=090'
            suffix += variant_key if variant_key.startswith("?") else f"_{variant_key}"
        sanitized_suffix = suffix.replace(" ", "_")
        path = output_dir / f"{self.timestamp}{sanitized_suffix}"
        if task_id is not None:
            path = path.with_name(f"{path.stem}_{task_id}")
        path = path.with_suffix(".jsonl")
        handle = open(path, 'w', encoding='utf-8')
        self._streams[stream_key] = handle
        self._paths[stream_key] = path
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

    def _derive_variant_key_from_hyperparams(self, hyperparams: Any) -> str:
        if not hyperparams or not isinstance(hyperparams, Mapping):
            return ""

        normalized: Dict[str, Any] = {}
        for key, value in hyperparams.items():
            if value is None:
                continue
            if not isinstance(key, str):
                key = str(key)
            normalized[key] = value

        if not normalized:
            return ""

        def format_value(key: str, value: Any) -> str:
            if isinstance(value, str):
                stripped = value.strip()
                if stripped.isdigit():
                    return stripped
                try:
                    value = float(stripped)
                except ValueError:
                    return stripped

            if isinstance(value, bool):
                return "1" if value else "0"

            if isinstance(value, int):
                if key in {"k"}:
                    return str(value).zfill(3)
                return str(value)

            if isinstance(value, float):
                if key in {"rho", "lambda"} and 0.0 <= value <= 1.0:
                    return str(int(round(value * 100.0))).zfill(3)
                return f"{value:.4f}".rstrip("0").rstrip(".")

            return str(value)

        preferred_order = ["rho", "lambda", "k", "epsilon"]
        keys: List[str] = []
        for key in preferred_order:
            if key in normalized:
                keys.append(key)
        for key in sorted(k for k in normalized.keys() if k not in set(preferred_order)):
            keys.append(key)

        pairs: List[str] = []
        for key in keys:
            pairs.append(f"{key}={format_value(key, normalized[key])}")

        return "?" + "&".join(pairs)

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
