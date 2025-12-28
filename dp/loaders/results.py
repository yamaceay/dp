from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple
import json

from dp.loaders.base import DatasetRecord, TextAnnotation, TextAnnotations


@dataclass(frozen=True)
class ResultRecord:
    idx: Optional[int]
    text: str
    annotations: TextAnnotations
    metadata: Dict[str, Any]


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
    metadata: Dict[str, Any] = {}
    if original is not None and isinstance(original.metadata, dict):
        metadata.update(original.metadata)
    if isinstance(result.metadata, dict):
        metadata.update(result.metadata)
    
    if original is not None:
        metadata["original_text"] = original.text
    
    spans_out: Optional[List[TextAnnotation]] = None
    if result.annotations.spans:
        spans_out = list(result.annotations.spans)
    
    uid = ""
    name = ""
    if original is not None:
        uid = original.uid
        name = original.name
    return DatasetRecord(
        text=result.text,
        uid=uid,
        name=name,
        spans=spans_out,
        metadata=metadata,
    )
