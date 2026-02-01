from typing import List, Dict, Optional
import json
from pathlib import Path

from dp.loaders.base import TextAnnotation, TextAnnotations, TokenEdit


_ANNOTATION_KEYS = {
    "start",
    "end",
    "label",
    "text",
    "replacement",
    "confidence",
    "annotator",
    "metadata",
}


def _parse_text_annotation(obj: object, *, path: Path, line_num: int, span_idx: int) -> TextAnnotation:
    if not isinstance(obj, dict):
        raise ValueError(f"Invalid annotation span in '{path}' at line {line_num} (span {span_idx})")
    unknown = set(obj.keys()) - _ANNOTATION_KEYS
    if unknown:
        raise ValueError(
            f"Unknown annotation keys {sorted(unknown)} in '{path}' at line {line_num} (span {span_idx})"
        )

    if "start" not in obj or "end" not in obj:
        raise ValueError(f"Missing start/end in '{path}' at line {line_num} (span {span_idx})")
    start = obj["start"]
    end = obj["end"]
    if not isinstance(start, int) or not isinstance(end, int):
        raise ValueError(f"start/end must be ints in '{path}' at line {line_num} (span {span_idx})")
    if end < start:
        raise ValueError(f"Invalid span (end < start) in '{path}' at line {line_num} (span {span_idx})")

    label = obj.get("label")
    if label is not None and not isinstance(label, str):
        raise ValueError(f"label must be a string or null in '{path}' at line {line_num} (span {span_idx})")
    text = obj.get("text")
    if text is not None and not isinstance(text, str):
        raise ValueError(f"text must be a string or null in '{path}' at line {line_num} (span {span_idx})")
    replacement = obj.get("replacement")
    if replacement is not None and not isinstance(replacement, str):
        raise ValueError(
            f"replacement must be a string or null in '{path}' at line {line_num} (span {span_idx})"
        )
    confidence = obj.get("confidence")
    if confidence is not None and not isinstance(confidence, (int, float)):
        raise ValueError(
            f"confidence must be a number or null in '{path}' at line {line_num} (span {span_idx})"
        )
    annotator = obj.get("annotator")
    if annotator is not None and not isinstance(annotator, str):
        raise ValueError(
            f"annotator must be a string or null in '{path}' at line {line_num} (span {span_idx})"
        )
    metadata = obj.get("metadata")
    if metadata is None:
        metadata = {}
    if not isinstance(metadata, dict):
        raise ValueError(f"metadata must be an object in '{path}' at line {line_num} (span {span_idx})")

    return TextAnnotation(
        start=start,
        end=end,
        label=label,
        text=text,
        replacement=replacement,
        confidence=float(confidence) if isinstance(confidence, (int, float)) else None,
        annotator=annotator,
        metadata=metadata,
    )


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


def apply_annotations(text: str, annotations: List[TextAnnotation], mask_text: Optional[str] = None) -> str:
    if not annotations:
        return text
    masked = text
    for ann in sorted(annotations, key=lambda a: a.start, reverse=True):
        if not isinstance(ann.start, int) or not isinstance(ann.end, int):
            raise ValueError("Starting anonymization annotation start and end must be integers")
        if ann.start < 0 or ann.end < 0 or ann.start >= ann.end:
            raise ValueError("Starting anonymization annotation has invalid start/end span")
        if ann.end > len(masked):
            raise ValueError("Starting anonymization annotation end exceeds text length")
        repl: Optional[str] = mask_text
        if isinstance(ann.replacement, str) and ann.replacement:
            repl = ann.replacement
        elif isinstance(ann.label, str) and ann.label:
            repl = f"[{ann.label}]"
        if repl is None:
            raise ValueError("Starting anonymization span has no replacement and no label; cannot mask")
        masked = masked[: ann.start] + repl + masked[ann.end :]
    return masked


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


def read_batch_textannotations_from_path(path: str) -> List[TextAnnotations]:
    jsonl_path = Path(path)
    if not jsonl_path.exists():
        raise ValueError(f"Annotation file not found: {path}")
    if jsonl_path.suffix == ".jsonl":
        return _read_jsonl_textannotations(jsonl_path)
    raise ValueError(f"Unsupported annotation file format: {jsonl_path.suffix}")


def read_batch_annotations(
    dataset: str,
    model: str,
    timestamp: str,
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


def read_batch_textannotations(
    dataset: str,
    model: str,
    timestamp: str,
) -> List[TextAnnotations]:
    from dp.utils.output import OUTPUT_STRUCTURE

    pattern = OUTPUT_STRUCTURE.get(model, f"outputs/{{dataset}}/{model}")
    output_dir = Path(pattern.format(dataset=dataset))
    jsonl_path = output_dir / f"{timestamp}.jsonl"

    if jsonl_path.exists():
        return _read_jsonl_textannotations(jsonl_path)

    raise ValueError(f"No annotation files found for {dataset}/{model}/{timestamp}")


def _read_jsonl_annotations(jsonl_path: Path) -> List[List[TextAnnotation]]:
    annotations_by_idx: Dict[int, List[TextAnnotation]] = {}

    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, start=1):
            if not line.strip():
                continue

            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"Invalid JSONL record in '{jsonl_path}' at line {line_num}")
            idx = record.get("idx")
            if not isinstance(idx, int) or idx < 0:
                raise ValueError(f"Invalid idx in '{jsonl_path}' at line {line_num}")
            annotations_obj = record.get("annotations")
            if not isinstance(annotations_obj, dict):
                raise ValueError(f"Missing annotations object in '{jsonl_path}' at line {line_num}")
            spans_raw = annotations_obj.get("spans")
            if not isinstance(spans_raw, list):
                raise ValueError(f"annotations.spans must be a list in '{jsonl_path}' at line {line_num}")

            annotations: List[TextAnnotation] = [
                _parse_text_annotation(span, path=jsonl_path, line_num=line_num, span_idx=span_idx)
                for span_idx, span in enumerate(spans_raw)
            ]

            annotations_by_idx[idx] = annotations
    
    if not annotations_by_idx:
        return []
    
    max_idx = max(annotations_by_idx.keys())
    result = []
    for idx in range(max_idx + 1):
        result.append(annotations_by_idx.get(idx, []))
    
    return result


def _read_jsonl_textannotations(jsonl_path: Path) -> List[TextAnnotations]:
    annotations_by_idx: Dict[int, TextAnnotations] = {}

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            if not line.strip():
                continue

            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"Invalid JSONL record in '{jsonl_path}' at line {line_num}")
            idx = record.get("idx")
            if not isinstance(idx, int) or idx < 0:
                raise ValueError(f"Invalid idx in '{jsonl_path}' at line {line_num}")

            annotations_obj = record.get("annotations")
            if annotations_obj is None:
                annotations_obj = {}
            if not isinstance(annotations_obj, dict):
                raise ValueError(f"annotations must be an object in '{jsonl_path}' at line {line_num}")

            spans_raw = annotations_obj.get("spans")
            if spans_raw is None:
                spans_raw = []
            if not isinstance(spans_raw, list):
                raise ValueError(f"annotations.spans must be a list in '{jsonl_path}' at line {line_num}")
            spans: list[TextAnnotation] = [
                _parse_text_annotation(span, path=jsonl_path, line_num=line_num, span_idx=span_idx)
                for span_idx, span in enumerate(spans_raw)
            ]

            token_edits_raw = annotations_obj.get("token_edits")
            if token_edits_raw is None:
                metadata_obj = record.get("metadata")
                if metadata_obj is None:
                    metadata_obj = {}
                if not isinstance(metadata_obj, dict):
                    raise ValueError(f"metadata must be an object in '{jsonl_path}' at line {line_num}")
                token_edits_raw = metadata_obj.get("token_edits")

            token_edits: list[TokenEdit] = []
            if token_edits_raw is None:
                token_edits_raw = []
            if not isinstance(token_edits_raw, list):
                raise ValueError(f"token_edits must be a list in '{jsonl_path}' at line {line_num}")
            for item_idx, item in enumerate(token_edits_raw):
                try:
                    token_edits.append(TokenEdit.from_mapping(item))
                except Exception as e:
                    raise ValueError(
                        f"Invalid token_edit in '{jsonl_path}' at line {line_num} (item {item_idx})"
                    ) from e

            annotations_by_idx[idx] = TextAnnotations(spans=spans, token_edits=token_edits)

    if not annotations_by_idx:
        return []

    max_idx = max(annotations_by_idx.keys())
    result: List[TextAnnotations] = []
    for idx in range(max_idx + 1):
        result.append(annotations_by_idx.get(idx, TextAnnotations()))

    return result

def list_batch_timestamps(
    dataset: str,
    model: str,
) -> List[str]:
    from dp.utils.output import OUTPUT_STRUCTURE
    
    pattern = OUTPUT_STRUCTURE.get(model, f"outputs/{{dataset}}/{model}")
    output_dir = Path(pattern.format(dataset=dataset))
    
    if not output_dir.exists():
        return []
    
    files = output_dir.glob("*.jsonl")
    timestamps = [file_path.stem for file_path in files]
    
    return sorted(timestamps)

