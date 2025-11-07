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
