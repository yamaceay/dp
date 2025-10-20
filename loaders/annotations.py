from typing import List, Dict, Optional
import json
from pathlib import Path

from dp.loaders.base import TextAnnotation


def read_annotations(path: str) -> Dict[str, List[TextAnnotation]]:
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    annotations = {}
    for uid, items in data.items():
        if not items:
            annotations[uid] = []
            continue
        
        if isinstance(items[0], dict):
            annotations[uid] = [TextAnnotation(**item) for item in items]
        elif isinstance(items[0], (list, tuple)) and len(items[0]) >= 2:
            annotations[uid] = [
                TextAnnotation(start=item[0], end=item[1])
                for item in items
            ]
        else:
            annotations[uid] = []
    
    return annotations


def write_annotations(annotations: Dict[str, List[TextAnnotation]], path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    
    data = {}
    for uid, annots in annotations.items():
        data[uid] = [
            {
                "start": ann.start,
                "end": ann.end,
                "label": ann.label,
                "text": ann.text,
                "replacement": ann.replacement,
                "confidence": ann.confidence,
                "annotator": ann.annotator,
                "metadata": ann.metadata
            }
            for ann in annots
        ]
    
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def apply_annotations(text: str, annotations: List[TextAnnotation], replacement: str = "[MASK]") -> str:
    sorted_annots = sorted(annotations, key=lambda x: x.start, reverse=True)
    masked_text = text
    for ann in sorted_annots:
        if 0 <= ann.start < ann.end <= len(masked_text):
            repl = ann.replacement if ann.replacement else replacement
            masked_text = masked_text[:ann.start] + repl + masked_text[ann.end:]
    return masked_text


def annotations_to_spans(annotations: List[TextAnnotation]) -> List[List[int]]:
    return [[ann.start, ann.end] for ann in annotations]


def spans_to_annotations(spans: List[List[int]], text: str = "", **kwargs) -> List[TextAnnotation]:
    return [
        TextAnnotation(
            start=span[0],
            end=span[1],
            text=text[span[0]:span[1]] if text and span[1] <= len(text) else None,
            **kwargs
        )
        for span in spans
    ]


def read_batch_annotations_from_path(path: str) -> List[List[TextAnnotation]]:
    jsonl_path = Path(path)
    
    if not jsonl_path.exists():
        raise ValueError(f"Annotation file not found: {path}")
    
    if jsonl_path.suffix == '.jsonl':
        return _read_jsonl_annotations(jsonl_path)
    
    raise ValueError(f"Unsupported annotation file format: {jsonl_path.suffix}")


def read_batch_annotations(
    dataset: str,
    model: str,
    timestamp: str,
    base_path: str = "outputs"
) -> List[List[TextAnnotation]]:
    """
    Read batch annotations from file. Automatically detects format based on file extension.
    Supports:
    - .jsonl format (one JSON object per line)
    - .json format (separate files per record)
    """
    from dp.utils.output import OUTPUT_STRUCTURE
    
    pattern = OUTPUT_STRUCTURE.get(model, f"outputs/{{dataset}}/{model}")
    output_dir = Path(pattern.format(dataset=dataset))
    
    jsonl_path = output_dir / f"{timestamp}.jsonl"
    
    if jsonl_path.exists():
        return _read_jsonl_annotations(jsonl_path)
    
    raise ValueError(f"No annotation files found for {dataset}/{model}/{timestamp}")


def _read_jsonl_annotations(jsonl_path: Path) -> List[List[TextAnnotation]]:
    """Read annotations from a JSONL file."""
    annotations_by_idx = {}
    
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            
            record = json.loads(line)
            idx = record.get("idx")
            if idx is None:
                continue
            
            spans = record.get("spans", [])
            if isinstance(spans, list) and spans:
                if isinstance(spans[0], dict):
                    annotations = [TextAnnotation(**span) for span in spans]
                else:
                    annotations = []
            else:
                annotations = []
            
            annotations_by_idx[idx] = annotations
    
    if not annotations_by_idx:
        return []
    
    max_idx = max(annotations_by_idx.keys())
    result = []
    for idx in range(max_idx + 1):
        result.append(annotations_by_idx.get(idx, []))
    
    return result

def list_batch_timestamps(
    dataset: str,
    model: str,
    base_path: str = "outputs"
) -> List[str]:
    from dp.utils.output import OUTPUT_STRUCTURE
    
    pattern = OUTPUT_STRUCTURE.get(model, f"outputs/{{dataset}}/{model}")
    output_dir = Path(pattern.format(dataset=dataset))
    
    if not output_dir.exists():
        return []
    
    files = output_dir.glob("*.jsonl")
    timestamps = [file_path.stem for file_path in files]
    
    return sorted(timestamps)

