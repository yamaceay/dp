from __future__ import annotations

from typing import Dict, List, Sequence, Tuple, Optional

from dp.loaders.base import TextAnnotation


def _normalize_span(span: Sequence[int] | Sequence[float]) -> Tuple[int, int]:
    if len(span) < 2:
        raise ValueError("span must contain start and end positions")
    start = int(span[0])
    end = int(span[1])
    return start, end


def apply_token_edits(original_text: str, edits: Sequence[Dict[str, object]]) -> str:
    operations: List[Tuple[int, int, str, str]] = []
    for edit in edits:
        kind = str(edit.get("kind"))
        payload = str(edit.get("text", ""))
        span_value = edit.get("span")
        if kind in {"replaced", "deleted"}:
            if not isinstance(span_value, (list, tuple)):
                raise ValueError(f"{kind} edit requires a span")
            start, end = _normalize_span(span_value)
            operations.append((start, end, kind, payload))
        elif kind == "added":
            if not isinstance(span_value, (list, tuple)):
                raise ValueError("added edit requires a span anchor")
            start, _ = _normalize_span(span_value)
            operations.append((start, start, kind, payload))
        else:
            raise ValueError(f"unknown edit kind '{kind}'")

    operations.sort(key=lambda item: (item[0], 1 if item[2] == "added" else 0))

    cursor = 0
    output: List[str] = []

    for start, end, kind, payload in operations:
        if start < cursor:
            if kind == "added":
                start = cursor
                end = cursor
            elif end <= cursor:
                continue
            else:
                start = cursor
        output.append(original_text[cursor:start])
        if kind == "deleted":
            cursor = end
        elif kind == "replaced":
            output.append(payload)
            cursor = end
        elif kind == "added":
            output.append(payload)
            cursor = start
        else:
            raise ValueError(f"unsupported edit kind '{kind}'")
    output.append(original_text[cursor:])
    return "".join(output)


def edits_to_annotations(
    original_text: str,
    edits: Sequence[Dict[str, object]],
) -> List[TextAnnotation]:
    annotations: List[TextAnnotation] = []
    for edit in edits:
        kind = str(edit.get("kind"))
        payload = str(edit.get("text", ""))
        span_value = edit.get("span")
        if not isinstance(span_value, (list, tuple)):
            if kind == "added":
                raise ValueError("added edit requires span anchor")
            continue
        start, end = _normalize_span(span_value)
        if kind == "deleted":
            annotations.append(
                TextAnnotation(
                    start=start,
                    end=end,
                    label=None,
                    text=original_text[start:end],
                    replacement="",
                )
            )
        elif kind == "replaced":
            annotations.append(
                TextAnnotation(
                    start=start,
                    end=end,
                    label=None,
                    text=original_text[start:end],
                    replacement=payload,
                )
            )
        elif kind == "added":
            annotations.append(
                TextAnnotation(
                    start=start,
                    end=end,
                    label=None,
                    text="",
                    replacement=payload,
                )
            )
        else:
            raise ValueError(f"unknown edit kind '{kind}'")
    annotations.sort(key=lambda ann: (ann.start, ann.end))
    return annotations


def validate_offsets(
    original_text: str,
    target_text: str,
    edits: Sequence[Dict[str, object]],
) -> Dict[str, object]:
    applied = apply_token_edits(original_text, edits)
    if applied == target_text:
        return {"ok": True, "mismatch_at": None}
    i = 0
    a_len = min(len(applied), len(target_text))
    while i < a_len and applied[i] == target_text[i]:
        i += 1
    ctx_start = max(0, i - 20)
    return {
        "ok": False,
        "mismatch_at": i,
        "applied_slice": applied[ctx_start:i + 20],
        "target_slice": target_text[ctx_start:i + 20],
        "applied_len": len(applied),
        "target_len": len(target_text),
    }


def build_offset_map(edits: Sequence[Dict[str, object]]) -> List[Tuple[int, int, int]]:
    """Build a list of (original_pos, result_pos, delta) checkpoints from edits.

    Each edit changes the delta between original and result coordinates.
    Returns sorted checkpoints that can be used to map offsets.
    """
    if not edits:
        return []
    checkpoints: List[Tuple[int, int, int]] = []
    delta = 0
    ops: List[Tuple[int, int, str, str]] = []
    for edit in edits:
        kind = str(edit.get("kind", ""))
        text = str(edit.get("text", ""))
        span_val = edit.get("span")
        if not isinstance(span_val, (list, tuple)) or len(span_val) < 2:
            continue
        start, end = int(span_val[0]), int(span_val[1])
        ops.append((start, end, kind, text))
    ops.sort(key=lambda x: (x[0], 1 if x[2] == "added" else 0))
    for start, end, kind, text in ops:
        result_pos = start + delta
        if kind == "deleted":
            removed = end - start
            delta -= removed
            checkpoints.append((end, result_pos, delta))
        elif kind == "replaced":
            removed = end - start
            added = len(text)
            delta += added - removed
            checkpoints.append((end, result_pos + added, delta))
        elif kind == "added":
            added = len(text)
            delta += added
            checkpoints.append((start, result_pos + added, delta))
    return checkpoints


def map_result_offset_to_original(
    result_start: int,
    result_end: int,
    edits: Sequence[Dict[str, object]],
) -> Tuple[int, int]:
    """Map a (start, end) offset from result text back to original text coordinates.

    Uses the edits metadata to reverse the transformation. If the offset falls
    inside a replacement/insertion region, raises ValueError.
    """
    if result_start < 0:
        raise ValueError(f"result_start must be non-negative, got {result_start}")
    if result_end < result_start:
        raise ValueError(f"result_end ({result_end}) < result_start ({result_start})")
    if not edits:
        return (result_start, result_end)
    checkpoints = build_offset_map(edits)
    if not checkpoints:
        return (result_start, result_end)

    def _map_pos(result_pos: int) -> int:
        delta = 0
        for orig_end, res_pos, new_delta in checkpoints:
            if result_pos < res_pos:
                break
            delta = new_delta
        return result_pos - delta

    return (_map_pos(result_start), _map_pos(result_end))


def map_original_offset_to_result(
    orig_start: int,
    orig_end: int,
    edits: Sequence[Dict[str, object]],
) -> Optional[Tuple[int, int]]:
    """Map a (start, end) offset from original text to result text coordinates.

    Returns None if the span was deleted or modified by an edit.
    """
    if orig_start < 0:
        raise ValueError(f"orig_start must be non-negative, got {orig_start}")
    if orig_end < orig_start:
        raise ValueError(f"orig_end ({orig_end}) < orig_start ({orig_start})")
    if not edits:
        return (orig_start, orig_end)
    ops: List[Tuple[int, int, str, str]] = []
    for edit in edits:
        kind = str(edit.get("kind", ""))
        text = str(edit.get("text", ""))
        span_val = edit.get("span")
        if not isinstance(span_val, (list, tuple)) or len(span_val) < 2:
            continue
        start, end = int(span_val[0]), int(span_val[1])
        ops.append((start, end, kind, text))
    ops.sort(key=lambda x: (x[0], 1 if x[2] == "added" else 0))
    for start, end, kind, _ in ops:
        if kind in ("deleted", "replaced"):
            if not (orig_end <= start or orig_start >= end):
                return None
    delta = 0
    for start, end, kind, text in ops:
        if start >= orig_end:
            break
        if kind == "deleted":
            if end <= orig_start:
                delta -= (end - start)
        elif kind == "replaced":
            if end <= orig_start:
                delta += len(text) - (end - start)
        elif kind == "added":
            if start <= orig_start:
                delta += len(text)
    return (orig_start + delta, orig_end + delta)


def map_offsets_to_result(
    offsets: Sequence[Tuple[int, int]],
    edits: Sequence[Dict[str, object]],
) -> List[Tuple[int, int]]:
    mapped: List[Tuple[int, int]] = []
    for start, end in offsets:
        mapped_span = map_original_offset_to_result(start, end, edits)
        if mapped_span is None:
            raise ValueError(f"Unable to map offset ({start}, {end}) to result coordinates")
        mapped.append(mapped_span)
    return mapped
