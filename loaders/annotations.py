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
