from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from dp.loaders import DatasetRecord
from dp.utils.splitter import TextSplitter
from dp.utils.token_edits import map_offsets_to_result


Span = Tuple[int, int]


@dataclass(frozen=True)
class PrecomputedRiskEntry:
    span_scores: Dict[Span, float]
    ordered_offsets: List[Span]
    ordered_scores: np.ndarray


def _record_state_by_key(records: Optional[Sequence[DatasetRecord]]) -> Dict[str, Tuple[str, List[Dict[str, object]]]]:
    if not records:
        return {}
    state: Dict[str, Tuple[str, List[Dict[str, object]]]] = {}
    for idx, record in enumerate(records):
        prior_edits: List[Dict[str, object]] = []
        if isinstance(record.metadata, dict):
            prior = record.metadata.get("prior_token_edits", [])
            if isinstance(prior, list):
                prior_edits = prior
        text = record.text or ""
        if record.uid:
            state[str(record.uid)] = (text, prior_edits)
        if record.name:
            state[str(record.name)] = (text, prior_edits)
        if not record.uid and not record.name:
            state[f"record_{idx}"] = (text, prior_edits)
    return state


def align_precomputed_risk_scores(
    risk_scores: Dict[str, Dict[str, object]],
    records: Optional[Sequence[DatasetRecord]] = None,
    absolute_risk: bool = False,
) -> Dict[str, PrecomputedRiskEntry]:
    if not risk_scores:
        return {}
    splitter = TextSplitter()
    state_by_key = _record_state_by_key(records)
    resolved: Dict[str, PrecomputedRiskEntry] = {}
    for uid, payload in risk_scores.items():
        if not isinstance(payload, dict):
            continue
        offsets = payload.get("offsets")
        scores = payload.get("scores")
        if offsets is None or scores is None:
            continue

        span_map: Dict[Span, float] = {}
        raw_spans: List[Span] = []
        raw_scores: List[float] = []
        for span, value in zip(offsets, scores):
            if not isinstance(span, (list, tuple)) or len(span) < 2:
                continue
            try:
                span_key = (int(span[0]), int(span[1]))
                score_val = float(value)
            except (TypeError, ValueError):
                continue
            score_use = abs(score_val) if absolute_risk else score_val
            span_map[span_key] = score_use
            raw_spans.append(span_key)
            raw_scores.append(score_use)

        if not span_map:
            continue

        order = sorted(range(len(raw_spans)), key=lambda i: (raw_spans[i][0], raw_spans[i][1]))
        ordered_spans = [raw_spans[i] for i in order]
        ordered_scores = [raw_scores[i] for i in order]

        state = state_by_key.get(str(uid))
        if state is not None:
            text, prior_edits = state
            splitter_spans = [(start, end) for start, end, _ in splitter.tokenize_with_spans(text)]
            if not set(ordered_spans).issubset(set(splitter_spans)):
                if not prior_edits:
                    raise ValueError(f"Risk offsets for uid {uid!r} do not align with record text")
                mapped_spans = map_offsets_to_result(ordered_spans, prior_edits)
                mapped_scores: Dict[Span, float] = {}
                for mapped_span, original_span in zip(mapped_spans, ordered_spans):
                    if mapped_span in mapped_scores:
                        raise ValueError(f"Duplicate mapped span {mapped_span} for uid {uid!r}")
                    mapped_scores[mapped_span] = span_map[original_span]
                mapped_offsets = sorted(mapped_scores.keys(), key=lambda span: (span[0], span[1]))
                mapped_ordered_scores = [mapped_scores[span] for span in mapped_offsets]
                resolved[str(uid)] = PrecomputedRiskEntry(
                    span_scores=mapped_scores,
                    ordered_offsets=mapped_offsets,
                    ordered_scores=np.asarray(mapped_ordered_scores, dtype=float),
                )
                continue

        resolved[str(uid)] = PrecomputedRiskEntry(
            span_scores=span_map,
            ordered_offsets=ordered_spans,
            ordered_scores=np.asarray(ordered_scores, dtype=float),
        )
    return resolved
