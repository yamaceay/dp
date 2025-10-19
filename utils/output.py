from typing import Optional, Dict, Any
from pathlib import Path
from datetime import datetime
import json
from dataclasses import asdict

from dp.methods.anonymizer import AnonymizationResult
from dp.loaders.base import TextAnnotation


OUTPUT_STRUCTURE = {
    "simple": "outputs/{dataset}/simple",
    "k-anon": "outputs/{dataset}/k-anon",
    "dpmlm": "outputs/{dataset}/dp/dpmlm",
    "dpbart": "outputs/{dataset}/dp/dpbart",
    "dpprompt": "outputs/{dataset}/dp/dpprompt",
    "dpparaphrase": "outputs/{dataset}/dp/dpparaphrase",
    "petre": "outputs/{dataset}/k-anon/petre",
}


class OutputHandler:
    def output(self, result: AnonymizationResult, dataset: str, model: str, **kwargs):
        raise NotImplementedError


class PrintOutputHandler(OutputHandler):
    def output(self, result: AnonymizationResult, dataset: str, model: str, **kwargs):
        print("Anonymized Text:", result.text)
        print("Annotations:", result.spans)
        print("Metadata:", result.metadata)


class FileOutputHandler(OutputHandler):
    def __init__(self, base_path: str = "outputs"):
        self.base_path = base_path
    
    def output(self, result: AnonymizationResult, dataset: str, model: str, **kwargs):
        output_dir = self._get_output_dir(dataset, model)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        idx = kwargs.get("idx", None)
        if idx is not None:
            filename = f"{timestamp}_idx{idx}.json"
        else:
            filename = f"{timestamp}.json"
        
        output_path = output_dir / filename
        
        data = {
            "text": result.text,
            "spans": [self._serialize_annotation(ann) for ann in (result.spans or [])],
            "metadata": result.metadata
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"Output written to: {output_path}")
    
    def _get_output_dir(self, dataset: str, model: str) -> Path:
        pattern = OUTPUT_STRUCTURE.get(model, f"outputs/{{dataset}}/{model}")
        path_str = pattern.format(dataset=dataset)
        return Path(path_str)
    
    def _serialize_annotation(self, ann: TextAnnotation) -> Dict[str, Any]:
        return asdict(ann)


OUTPUT_HANDLER_REGISTRY = {
    "print": PrintOutputHandler,
    "file": FileOutputHandler,
}
