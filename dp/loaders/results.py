from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import json

from dp.loaders.base import DatasetRecord, TextAnnotation, TextAnnotations, TokenEdit


@dataclass(frozen=True)
class ResultRecord:
    idx: Optional[int]
    text: str
    annotations: TextAnnotations
    metadata: Dict[str, Any]


Replacement = Tuple[int, int, str]


def load_result_records(path: str) -> List[ResultRecord]:
    records: List[ResultRecord] = []
    with open(path, "r", encoding="utf-8") as reader:
        for line_num, line in enumerate(reader, start=1):
            entry = line.strip()
            if not entry:
                continue
            payload = json.loads(entry)
            if not isinstance(payload, dict):
                raise ValueError(f"Invalid JSONL record at line {line_num}")
            idx = payload.get("idx")
            if idx is not None and not isinstance(idx, int):
                raise ValueError(f"Invalid idx at line {line_num}")
            text = payload.get("text")
            if not isinstance(text, str):
                raise ValueError(f"Missing text at line {line_num}")
            metadata = payload.get("metadata")
            if metadata is None:
                metadata = {}
            if not isinstance(metadata, dict):
                raise ValueError(f"metadata must be an object at line {line_num}")
            annotations = _parse_text_annotations(payload, line_num)
            records.append(ResultRecord(idx=idx, text=text, annotations=annotations, metadata=metadata))
    return records


def build_dataset_from_results(
    result_path: str,
    original_records: Optional[Sequence[DatasetRecord]] = None,
) -> Tuple[List[DatasetRecord], List[int]]:
    result_records = load_result_records(result_path)
    dataset_records: List[DatasetRecord] = []
    source_indices: List[int] = []
    for pos, result in enumerate(result_records):
        idx = result.idx if result.idx is not None else pos
        if idx < 0:
            raise ValueError("Result record idx must be non-negative")
        original = None
        if original_records is not None:
            if idx >= len(original_records):
                raise ValueError(f"Result idx {idx} is out of range for original records")
            original = original_records[idx]
        dataset_records.append(_merge_result_record(result, original))
        source_indices.append(idx)
    return dataset_records, source_indices


def _parse_text_annotations(payload: Dict[str, Any], line_num: int) -> TextAnnotations:
    annotations_obj = payload.get("annotations")
    if annotations_obj is None:
        annotations_obj = {}
    if not isinstance(annotations_obj, dict):
        raise ValueError(f"annotations must be an object at line {line_num}")
    spans_raw = annotations_obj.get("spans")
    if spans_raw is None:
        spans_raw = payload.get("spans")
    if spans_raw is None:
        spans_raw = []
    if not isinstance(spans_raw, list):
        raise ValueError(f"annotations.spans must be a list at line {line_num}")
    spans: List[TextAnnotation] = []
    for span_idx, span in enumerate(spans_raw):
        spans.append(_parse_text_annotation(span, line_num, span_idx))
    token_edits_raw = annotations_obj.get("token_edits")
    if token_edits_raw is None:
        metadata_obj = payload.get("metadata")
        if metadata_obj is None:
            metadata_obj = {}
        if not isinstance(metadata_obj, dict):
            raise ValueError(f"metadata must be an object at line {line_num}")
        token_edits_raw = metadata_obj.get("token_edits")
    if token_edits_raw is None:
        token_edits_raw = []
    if not isinstance(token_edits_raw, list):
        raise ValueError(f"token_edits must be a list at line {line_num}")
    token_edits: List[TokenEdit] = []
    for item_idx, item in enumerate(token_edits_raw):
        try:
            token_edits.append(TokenEdit.from_mapping(item))
        except Exception as exc:
            raise ValueError(f"Invalid token_edit at line {line_num} (item {item_idx})") from exc
    return TextAnnotations(spans=spans, token_edits=token_edits)


def _parse_text_annotation(obj: object, line_num: int, span_idx: int) -> TextAnnotation:
    if not isinstance(obj, dict):
        raise ValueError(f"Invalid annotation span at line {line_num} (span {span_idx})")
    if "start" not in obj or "end" not in obj:
        raise ValueError(f"Missing start/end at line {line_num} (span {span_idx})")
    start = obj["start"]
    end = obj["end"]
    if not isinstance(start, int) or not isinstance(end, int):
        raise ValueError(f"start/end must be ints at line {line_num} (span {span_idx})")
    if end < start:
        raise ValueError(f"Invalid span at line {line_num} (span {span_idx})")
    label = obj.get("label")
    if label is not None and not isinstance(label, str):
        raise ValueError(f"label must be a string or null at line {line_num} (span {span_idx})")
    text = obj.get("text")
    if text is not None and not isinstance(text, str):
        raise ValueError(f"text must be a string or null at line {line_num} (span {span_idx})")
    replacement = obj.get("replacement")
    if replacement is not None and not isinstance(replacement, str):
        raise ValueError(f"replacement must be a string or null at line {line_num} (span {span_idx})")
    confidence = obj.get("confidence")
    if confidence is not None and not isinstance(confidence, (int, float)):
        raise ValueError(f"confidence must be a number or null at line {line_num} (span {span_idx})")
    annotator = obj.get("annotator")
    if annotator is not None and not isinstance(annotator, str):
        raise ValueError(f"annotator must be a string or null at line {line_num} (span {span_idx})")
    metadata = obj.get("metadata")
    if metadata is None:
        metadata = {}
    if not isinstance(metadata, dict):
        raise ValueError(f"metadata must be an object at line {line_num} (span {span_idx})")
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


def _merge_result_record(
    result: ResultRecord,
    original: Optional[DatasetRecord],
) -> DatasetRecord:
    original_text = original.text if original is not None else result.text
    replacements = _build_replacements(result.annotations, original_text)
    merged_spans: List[TextAnnotation] = []
    if result.annotations.spans:
        merged_spans.extend(_shift_annotations(result.annotations.spans, replacements, result.text))
    merged_spans = _dedupe_spans(sorted(merged_spans, key=lambda ann: (ann.start, ann.end)))
    spans_out = merged_spans if merged_spans else None
    metadata: Dict[str, Any] = {}
    if original is not None and isinstance(original.metadata, dict):
        metadata.update(original.metadata)
    if isinstance(result.metadata, dict):
        metadata.update(result.metadata)
    
    token_edits_list = [te.to_dict() for te in result.annotations.token_edits]
    if not token_edits_list and result.annotations.spans:
        token_edits_list = _spans_to_prior_edits(result.annotations.spans, original_text)
    if token_edits_list:
        metadata["prior_token_edits"] = token_edits_list
    if original is not None:
        metadata["original_text"] = original.text
    
    uid = ""
    name = ""
    if original is not None:
        uid = original.uid
        name = original.name
    return DatasetRecord(
        text=original_text,
        uid=uid,
        name=name,
        spans=spans_out,
        metadata=metadata,
    )


def _spans_to_prior_edits(spans: List[TextAnnotation], original_text: str) -> List[Dict[str, Any]]:
    edits: List[Dict[str, Any]] = []
    cursor = 0
    for ann in sorted(spans, key=lambda a: a.start):
        original_span_text = ann.text
        if not original_span_text:
            continue
        replacement = ann.replacement
        if replacement is None and ann.label:
            replacement = f"[{ann.label}]"
        if replacement is None:
            continue
        
        pos = original_text.find(original_span_text, cursor)
        if pos < 0:
            continue
        
        start = pos
        end = pos + len(original_span_text)
        cursor = end
        
        edits.append({
            "kind": "replaced",
            "span": [start, end],
            "text": replacement,
        })
    return edits


def _build_replacements(annotations: TextAnnotations, original_text: str) -> List[Replacement]:
    replacements: List[Replacement] = []
    for ann in sorted(annotations.spans or [], key=lambda a: a.start):
        if ann.start < 0 or ann.end < ann.start or ann.end > len(original_text):
            raise ValueError("Invalid annotation span for replacements")
        replacement = ann.replacement
        if replacement is None and ann.label:
            replacement = f"[{ann.label}]"
        if replacement is None:
            replacement = original_text[ann.start:ann.end]
        replacements.append((ann.start, ann.end, replacement))
    return replacements


def _shift_annotations(
    annotations: Iterable[TextAnnotation],
    replacements: List[Replacement],
    new_text: str,
) -> List[TextAnnotation]:
    shifted: List[TextAnnotation] = []
    for ann in annotations:
        new_start, new_end = _shift_span(ann.start, ann.end, replacements)
        if new_start < 0 or new_end > len(new_text) or new_end < new_start:
            raise ValueError("Shifted annotation span is out of bounds")
        new_text_value = new_text[new_start:new_end]
        shifted.append(
            TextAnnotation(
                start=new_start,
                end=new_end,
                label=ann.label,
                text=new_text_value,
                replacement=ann.replacement,
                confidence=ann.confidence,
                annotator=ann.annotator,
                metadata=dict(ann.metadata) if isinstance(ann.metadata, dict) else {},
            )
        )
    return shifted


def _shift_span(start: int, end: int, replacements: List[Replacement]) -> Tuple[int, int]:
    if not replacements:
        return start, end
    delta = 0
    for rep_start, rep_end, rep_text in replacements:
        if end <= rep_start:
            break
        if start >= rep_end:
            delta += len(rep_text) - (rep_end - rep_start)
            continue
        new_start = rep_start + delta
        return new_start, new_start + len(rep_text)
    return start + delta, end + delta


def _dedupe_spans(spans: Sequence[TextAnnotation]) -> List[TextAnnotation]:
    seen: set[Tuple[int, int, Optional[str], Optional[str], Optional[str], Optional[float]]] = set()
    deduped: List[TextAnnotation] = []
    for ann in spans:
        key = (ann.start, ann.end, ann.label, ann.annotator, ann.replacement, ann.confidence)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(ann)
    return deduped
