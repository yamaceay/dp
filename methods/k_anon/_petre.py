from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple
from collections import defaultdict
import re
import nltk
import numpy as np
import torch

from transformers import pipeline

from dp.utils.explainer.base import TokenExplainer
from dp.methods.anonymizer import AnonymizationResult
from dp.methods.k_anon import KAnonymizer
from dp.loaders.base import DatasetRecord, TextAnnotation
from dp.utils.splitter import TextSplitter
from dp.utils.chunking import TokenAwareChunker
from dp.methods.constants import PII_CLASSIFIER_MODEL_LIST, RISK_MASKER_MODEL_LIST

@dataclass
class RecordState:
    uid: str
    name: str
    label: int
    text: str
    sentence_spans: List[Tuple[int, int]]
    term_spans: List[Tuple[int, int]]
    term_texts: List[str]
    term_indices_by_text: Dict[str, List[int]]


class PetreAnonymizer(KAnonymizer):
    def __init__(
        self,
        mask_text: str = "[MASK]",
        device: str = "auto",
        use_chunking: bool = True,
        mask_all_instances: bool = True,
        batch_size: int = 32,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.mask_text = mask_text
        self.use_chunking = use_chunking
        self.mask_all_instances = mask_all_instances
        self.batch_size = batch_size
        self.device = self._resolve_device(device)
        self.splitter = TextSplitter()
        self.explainer = None
        self.tri_pipeline_path: Optional[str] = None
        self.tri_pipeline = None
        self._special_pattern = re.compile(r"[^\nA-Za-z0-9À-ÖØ-öø-ÿЀ-ӿ/]+")
        self._terms_to_ignore = set()
        self._k_processed: Dict[int, Set[int]] = defaultdict(set)
        self._annotation_history: Dict[int, Dict[str, List[TextAnnotation]]] = {}
        self.dataset_records: List[DatasetRecord] = []
        self._records_by_idx: List[RecordState] = []
        self._records_by_uid: Dict[str, RecordState] = {}
        self.label_to_name: Dict[int, str] = {}
        self.name_to_label: Dict[str, int] = {}
        self.num_labels: int = 0
        self._starting_annotations: Dict[str, List[TextAnnotation]] = {}
        self.annotations: Dict[str, List[TextAnnotation]] = {}
        self._annotation_name: Optional[str] = None
        self.tri_chunker: Optional[TokenAwareChunker] = None
        self._score_cache: Dict[str, np.ndarray] = {}
        self._score_order_cache: Dict[str, List[int]] = {}
        self._raw_risk_scores: Dict[str, Tuple[List[Tuple[int, int]], List[float]]] = {}
        self._prepared_risk_scores: Dict[str, np.ndarray] = {}

    def add_dataset_records(self, dataset_records: List[DatasetRecord]):
        if not dataset_records:
            raise ValueError("dataset_records cannot be empty")
        self.dataset_records = list(dataset_records)
        self._build_label_mappings(self.dataset_records)
        self._build_record_states(self.dataset_records)
        self._starting_annotations = {state.uid: [] for state in self._records_by_idx}
        self.annotations = {uid: [] for uid in self._starting_annotations}
        self._k_processed.clear()
        self._annotation_history.clear()
        self._annotation_name = None
        self.tri_chunker = None
        self._terms_to_ignore = self._build_terms_to_ignore(self.annotations, self._annotation_name)
        self._clear_score_cache()
        self._prepare_risk_scores_for_records()

    def _clear_score_cache(self) -> None:
        self._score_cache.clear()
        self._score_order_cache.clear()
        self._refresh_prepared_risk_cache()

    def _refresh_prepared_risk_cache(self) -> None:
        if not self._prepared_risk_scores:
            return
        for uid, scores in self._prepared_risk_scores.items():
            self._score_cache[uid] = scores
            self._score_order_cache[uid] = list(np.argsort(-scores, kind="mergesort"))

    def set_risk_scores(
        self,
        risk_scores: Dict[str, Dict[str, object]],
        records: Optional[Sequence[DatasetRecord]] = None,
    ) -> None:
        self._raw_risk_scores = {}
        self._prepared_risk_scores = {}
        if not risk_scores:
            self._clear_score_cache()
            return
        for uid, payload in risk_scores.items():
            if not isinstance(payload, dict):
                continue
            offsets = payload.get("offsets")
            scores = payload.get("scores")
            if offsets is None or scores is None:
                continue
            normalized_offsets: List[Tuple[int, int]] = []
            for span in offsets:
                if isinstance(span, (list, tuple)) and len(span) >= 2:
                    normalized_offsets.append((int(span[0]), int(span[1])))
            score_list: List[float] = []
            for value in scores:
                try:
                    score_list.append(float(value))
                except (TypeError, ValueError):
                    score_list.append(float("nan"))
            if not normalized_offsets or not score_list:
                continue
            self._raw_risk_scores[uid] = (normalized_offsets, score_list)
        self._clear_score_cache()
        self._prepare_risk_scores_for_records()

    def _prepare_risk_scores_for_records(self) -> None:
        if not self._raw_risk_scores or not self._records_by_uid:
            return
        self._prepared_risk_scores = {}
        for uid, state in self._records_by_uid.items():
            raw = self._raw_risk_scores.get(uid)
            if raw is None:
                continue
            offsets, scores = raw
            span_map: Dict[Tuple[int, int], float] = {}
            for span, value in zip(offsets, scores):
                span_map[span] = float(value)
            token_scores = np.full(len(state.term_spans), float("-inf"), dtype=float)
            for idx, span in enumerate(state.term_spans):
                key = (span[0], span[1])
                if key not in span_map:
                    continue
                token_scores[idx] = float(span_map[key])
            self._prepared_risk_scores[uid] = token_scores
        self._refresh_prepared_risk_cache()

    def _build_terms_to_ignore(self, annotations: Dict[str, List[TextAnnotation]], name: Optional[str]) -> Set[str]:
        stopwords = set(nltk.corpus.stopwords.words("english"))
        marks = {
            self.mask_text,
            *stopwords,
            "[CLS]",
            "[SEP]",
            "[PAD]",
            "",
            " ",
            "\t",
            "\n",
        }
        if name in PII_CLASSIFIER_MODEL_LIST or name in RISK_MASKER_MODEL_LIST or name == "manual":
            for uid, anns in annotations.items():
                for ann in anns:
                    if ann.label:
                        marks.add(ann.label)
        normalized_marks: Set[str] = set()
        for mark in marks:
            if mark is None:
                continue
            normalized_marks.add(mark)
            normalized_marks.add(mark.lower())
        return normalized_marks

    def _build_label_mappings(self, dataset_records: List[DatasetRecord]) -> None:
        names: Set[str] = set()
        for idx, record in enumerate(dataset_records):
            name = record.name or record.uid or f"record_{idx}"
            names.add(name)
        sorted_names = sorted(names)
        if not sorted_names:
            raise ValueError("no individual names found in dataset_records")
        self.label_to_name = {idx: name for idx, name in enumerate(sorted_names)}
        self.name_to_label = {name: idx for idx, name in self.label_to_name.items()}
        self.num_labels = len(self.label_to_name)

    def _build_record_states(self, dataset_records: List[DatasetRecord]) -> None:
        self._records_by_idx = []
        self._records_by_uid = {}
        for idx, record in enumerate(dataset_records):
            uid = record.uid or f"record_{idx}"
            name = record.name or uid
            if name not in self.name_to_label:
                raise ValueError(f"unknown identity '{name}' for record {idx}")
            label = self.name_to_label[name]
            text = record.text or ""
            sentence_spans = self.splitter.split_sentences(text)
            if not sentence_spans:
                sentence_spans = [(0, len(text))]
            term_spans: List[Tuple[int, int]] = []
            term_texts: List[str] = []
            for sentence_start, sentence_end in sentence_spans:
                sentence_text = text[sentence_start:sentence_end]
                tokens = self.splitter.tokenize_with_spans(sentence_text)
                for token_start, token_end, _ in tokens:
                    absolute_start = sentence_start + token_start
                    absolute_end = sentence_start + token_end
                    term_spans.append((absolute_start, absolute_end))
                    term_texts.append(text[absolute_start:absolute_end])
            term_indices_by_text: Dict[str, List[int]] = defaultdict(list)
            for term_idx, term_text in enumerate(term_texts):
                term_indices_by_text[term_text].append(term_idx)
            state = RecordState(
                uid=uid,
                name=name,
                label=label,
                text=text,
                sentence_spans=sentence_spans,
                term_spans=term_spans,
                term_texts=term_texts,
                term_indices_by_text=term_indices_by_text,
            )
            self._records_by_idx.append(state)
            self._records_by_uid[uid] = state

    def _tokenize_sentence(self, text: str) -> List[Tuple[int, int]]:
        return [(start, end) for start, end, _ in self.splitter.tokenize_with_spans(text)]

    def _expand_span_to_tokens(
        self,
        state: RecordState,
        start: int,
        end: int,
    ) -> Tuple[int, int]:
        expanded_start = start
        expanded_end = end
        matched = False
        for token_start, token_end in state.term_spans:
            if token_end <= start or token_start >= end:
                continue
            matched = True
            if token_start < expanded_start:
                expanded_start = token_start
            if token_end > expanded_end:
                expanded_end = token_end
        if not matched:
            for token_start, token_end in self._tokenize_sentence(state.text):
                if token_end <= start or token_start >= end:
                    continue
                matched = True
                if token_start < expanded_start:
                    expanded_start = token_start
                if token_end > expanded_end:
                    expanded_end = token_end
        if not matched or expanded_start >= expanded_end:
            return start, end
        return expanded_start, expanded_end

    def _align_annotations(
        self,
        state: RecordState,
        annotations: List[TextAnnotation],
    ) -> List[TextAnnotation]:
        if not annotations:
            return []
        aligned: List[TextAnnotation] = []
        seen: Set[Tuple[int, int]] = set()
        for annotation in annotations:
            span_start, span_end = self._expand_span_to_tokens(state, annotation.start, annotation.end)
            key = (span_start, span_end)
            if key in seen:
                continue
            seen.add(key)
            aligned.append(
                TextAnnotation(
                    start=span_start,
                    end=span_end,
                    label=annotation.label,
                    text=state.text[span_start:span_end],
                    replacement=annotation.replacement or self.mask_text,
                    confidence=annotation.confidence,
                    annotator=annotation.annotator,
                    metadata=dict(annotation.metadata or {}),
                )
            )
        aligned.sort(key=lambda ann: (ann.start, ann.end))
        return aligned

    def _normalize_annotation_list(
        self,
        items: Iterable,
    ) -> List[TextAnnotation]:
        normalized: List[TextAnnotation] = []
        for item in items:
            if isinstance(item, TextAnnotation):
                normalized.append(
                    TextAnnotation(
                        start=int(item.start),
                        end=int(item.end),
                        label=item.label,
                        text=item.text,
                        replacement=item.replacement,
                        confidence=item.confidence,
                        annotator=item.annotator,
                        metadata=dict(item.metadata or {}),
                    )
                )
            elif isinstance(item, dict):
                if "start" not in item or "end" not in item:
                    continue
                normalized.append(
                    TextAnnotation(
                        start=int(item["start"]),
                        end=int(item["end"]),
                        label=item.get("label"),
                        text=item.get("text"),
                        replacement=item.get("replacement"),
                        confidence=item.get("confidence"),
                        annotator=item.get("annotator"),
                        metadata=dict(item.get("metadata") or {}),
                    )
                )
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                normalized.append(
                    TextAnnotation(start=int(item[0]), end=int(item[1]))
                )
        normalized.sort(key=lambda ann: (ann.start, ann.end))
        return normalized

    def _clone_annotation_dict(
        self,
        data: Dict[str, List[TextAnnotation]],
    ) -> Dict[str, List[TextAnnotation]]:
        cloned: Dict[str, List[TextAnnotation]] = {}
        for uid, annotations in data.items():
            cloned[uid] = [
                TextAnnotation(
                    start=ann.start,
                    end=ann.end,
                    label=ann.label,
                    text=ann.text,
                    replacement=ann.replacement,
                    confidence=ann.confidence,
                    annotator=ann.annotator,
                    metadata=dict(ann.metadata or {}),
                )
                for ann in annotations
            ]
        return cloned

    def _resolve_device(self, device: str) -> torch.device:
        if device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        return torch.device(device)

    def _pipeline_device(self):
        if self.device.type == "cpu":
            return -1
        if self.device.type == "cuda":
            return self.device.index or 0
        return self.device

    def _parse_label(self, label: str) -> int:
        if "_" in label:
            return int(label.split("_")[-1])
        return int(label)

    def _token_scores_for_state(self, state: RecordState) -> np.ndarray:
        if self.explainer is None:
            raise RuntimeError("Scoring strategy must be set before running PETRE")
        cached = self._score_cache.get(state.uid)
        if cached is not None:
            return cached
        precomputed = self._prepared_risk_scores.get(state.uid)
        if precomputed is not None:
            self._score_cache[state.uid] = precomputed
            self._score_order_cache[state.uid] = list(np.argsort(-precomputed, kind="mergesort"))
            return precomputed
        tokens = list(state.term_texts)
        if not tokens:
            empty = np.zeros(0, dtype=float)
            self._score_cache[state.uid] = empty
            return empty
        raw_scores = self.explainer.explain(state.text, state.term_spans)
        array = np.asarray(raw_scores, dtype=float).ravel()
        length = len(tokens)
        if array.size < length:
            padded = np.full(length, float("-inf"), dtype=float)
            if array.size > 0:
                padded[: array.size] = array
            array = padded
        elif array.size > length:
            array = array[:length]
        invalid_mask = ~np.isfinite(array)
        if invalid_mask.any():
            array = array.copy()
            array[invalid_mask] = float("-inf")
        self._score_cache[state.uid] = array
        return array

    def _ordered_token_indices_for_state(self, state: RecordState) -> List[int]:
        cached = self._score_order_cache.get(state.uid)
        if cached is not None:
            return cached
        scores = self._token_scores_for_state(state)
        if scores.size == 0:
            ordered: List[int] = []
        else:
            ordered = list(np.argsort(-scores, kind="mergesort"))
        self._score_order_cache[state.uid] = ordered
        return ordered

    def set_scoring_strategy(self, explainer: TokenExplainer) -> None:
        if explainer is None:
            raise ValueError("explainer cannot be None")
        if not hasattr(explainer, "tri_detector"):
            raise ValueError("explainer must expose tri_detector")
        tri_model_name = getattr(explainer.tri_detector, "model_name", None)
        if tri_model_name is None:
            raise ValueError("explainer.tri_detector must define model_name")
        self.explainer = explainer
        self.tri_pipeline_path = tri_model_name
        self.batch_size = int(getattr(explainer, "batch_size", self.batch_size))
        self._clear_score_cache()
        self._load_tri_pipeline()

    def _load_tri_pipeline(self) -> None:
        if self.tri_pipeline_path is None:
            raise ValueError("tri_pipeline_path must be set before loading pipeline")
        device_arg = self._pipeline_device()
        self.tri_pipeline = pipeline(
            "text-classification",
            model=self.tri_pipeline_path,
            tokenizer=self.tri_pipeline_path,
            device=device_arg,
            top_k=self.num_labels,
            truncation=False,
        )
        tokenizer = getattr(self.tri_pipeline, "tokenizer", None)
        if tokenizer is not None:
            max_tokens = getattr(tokenizer, "model_max_length", 512)
            if max_tokens is None or max_tokens <= 0:
                max_tokens = 512
            self.tri_chunker = TokenAwareChunker(tokenizer, max(max_tokens - 2, 1))
        else:
            self.tri_chunker = None

    def set_annotations(self, annotations: Dict[str, List[TextAnnotation]], name: Optional[str] = None) -> None:
        if not self._records_by_idx:
            raise RuntimeError("Dataset records must be added before setting annotations")
        aligned: Dict[str, List[TextAnnotation]] = {}
        for state in self._records_by_idx:
            uid = state.uid
            raw_items = annotations.get(uid, []) if annotations and uid in annotations else []
            normalized = self._normalize_annotation_list(raw_items)
            aligned[uid] = self._align_annotations(state, normalized)
        self._starting_annotations = self._clone_annotation_dict(aligned)
        self.annotations = self._clone_annotation_dict(aligned)
        self._k_processed.clear()
        self._annotation_history.clear()
        if name:
            self._annotation_name = name
        self._terms_to_ignore = self._build_terms_to_ignore(self.annotations, self._annotation_name)

    def anonymize(self, text: str, *args, **kwargs) -> AnonymizationResult:
        raise NotImplementedError("Use anonymize_from_dataset for PetreAnonymizer.")

    def _apply_spans_to_sentence(
        self,
        text: str,
        sentence_span: Tuple[int, int],
        spans: List[Tuple[int, int]],
    ) -> str:
        relevant: List[Tuple[int, int]] = []
        sent_start, sent_end = sentence_span
        for start, end in spans:
            if end <= sent_start or start >= sent_end:
                continue
            local_start = max(start, sent_start) - sent_start
            local_end = min(end, sent_end) - sent_start
            if local_end > local_start:
                relevant.append((local_start, local_end))
        if not relevant:
            return text[sent_start:sent_end]
        relevant.sort(key=lambda span: span[0], reverse=True)
        segment = text[sent_start:sent_end]
        for start, end in relevant:
            segment = segment[:start] + self.mask_text + segment[end:]
        return segment

    def _apply_spans_to_text(
        self,
        text: str,
        spans: List[Tuple[int, int]],
    ) -> str:
        if not spans:
            return text
        sorted_spans = sorted(spans, key=lambda span: span[0], reverse=True)
        masked = text
        for start, end in sorted_spans:
            if start < 0 or end > len(text) or start >= end:
                continue
            masked = masked[:start] + self.mask_text + masked[end:]
        return masked

    def _evaluate_state(
        self,
        state: RecordState,
        spans: List[Tuple[int, int]],
    ) -> np.ndarray:
        unique_spans = sorted({(start, end) for start, end in spans})
        pipeline_inputs: List[str] = []
        for sentence_span in state.sentence_spans:
            rendered = self._apply_spans_to_sentence(state.text, sentence_span, unique_spans)
            if not rendered.strip():
                rendered = self.mask_text
            if self.tri_chunker is not None:
                chunks = self.tri_chunker.chunk(rendered)
                if not chunks:
                    pipeline_inputs.append(rendered)
                else:
                    for chunk in chunks:
                        chunk_text = chunk.text
                        pipeline_inputs.append(chunk_text if chunk_text.strip() else self.mask_text)
            else:
                pipeline_inputs.append(rendered)
        if not pipeline_inputs:
            pipeline_inputs = [self.mask_text]
        results = self.tri_pipeline(pipeline_inputs, batch_size=self.batch_size)
        probs = np.zeros(self.num_labels, dtype=float)
        for split_result in results:
            for pred in split_result:
                label_idx = self._parse_label(pred["label"])
                if 0 <= label_idx < self.num_labels:
                    probs[label_idx] += float(pred["score"])
        num_inputs = max(len(pipeline_inputs), 1)
        probs /= float(num_inputs)
        return probs

    def _rank_from_probs(self, probs: np.ndarray, label_idx: int) -> int:
        sorted_indices = np.argsort(probs)[::-1]
        positions = np.where(sorted_indices == label_idx)[0]
        if positions.size == 0:
            return len(sorted_indices) + 1
        return int(positions[0]) + 1

    def _span_overlaps_existing(
        self,
        span: Tuple[int, int],
        existing: Set[Tuple[int, int]],
    ) -> bool:
        for other in existing:
            if not (span[1] <= other[0] or span[0] >= other[1]):
                return True
        return False

    def _should_ignore(self, text: str) -> bool:
        if not text.strip():
            return True
        clean = self._special_pattern.sub("", text).strip()
        if not clean:
            return True
        return clean.lower() in self._terms_to_ignore

    def _expand_candidate_spans(
        self,
        state: RecordState,
        base_span: Tuple[int, int],
        term_text: str,
        span_set: Set[Tuple[int, int]],
    ) -> List[Tuple[int, int]]:
        if not self.mask_all_instances:
            return [base_span]
        expanded: List[Tuple[int, int]] = []
        for idx in state.term_indices_by_text.get(term_text, []):
            candidate_span = state.term_spans[idx]
            span_tuple = (candidate_span[0], candidate_span[1])
            if span_tuple in span_set:
                continue
            if self._span_overlaps_existing(span_tuple, span_set):
                continue
            expanded.append(span_tuple)
        if not expanded:
            expanded.append(base_span)
        return expanded

    def _base_annotations_for(self, target_k: int) -> Dict[str, List[TextAnnotation]]:
        if not self._annotation_history:
            return self._starting_annotations
        eligible = [k for k in self._annotation_history if k <= target_k]
        if not eligible:
            return self._starting_annotations
        best_k = max(eligible)
        return self._annotation_history[best_k]

    def _ensure_annotations_for_k(
        self,
        target_k: int,
        indices: List[int],
    ) -> None:
        if not indices:
            return
        if self.tri_pipeline is None:
            raise RuntimeError("Scoring strategy must be set before running PETRE")

        if target_k not in self._annotation_history:
            base = self._clone_annotation_dict(self._base_annotations_for(target_k))
            aligned_base: Dict[str, List[TextAnnotation]] = {}
            for state in self._records_by_idx:
                aligned_base[state.uid] = self._align_annotations(state, base.get(state.uid, []))
            self._annotation_history[target_k] = aligned_base
            self._k_processed[target_k] = set()

        pending = [idx for idx in indices if idx not in self._k_processed[target_k]]
        if not pending:
            return

        self.annotations = self._clone_annotation_dict(self._annotation_history[target_k])
        for idx in pending:
            state = self._records_by_idx[idx]
            annotations = self.annotations[state.uid]
            spans: List[Tuple[int, int]] = [(ann.start, ann.end) for ann in annotations]
            span_set: Set[Tuple[int, int]] = set(spans)
            span_to_indices: Dict[Tuple[int, int], List[int]] = defaultdict(list)
            for token_idx, span in enumerate(state.term_spans):
                span_tuple = (span[0], span[1])
                span_to_indices[span_tuple].append(token_idx)
            used_indices: Set[int] = set()
            for existing_span in span_set:
                for mapped_idx in span_to_indices.get(existing_span, []):
                    used_indices.add(mapped_idx)
            scores = self._token_scores_for_state(state)
            ordered_indices = self._ordered_token_indices_for_state(state)
            current_probs = self._evaluate_state(state, spans)
            while True:
                rank = self._rank_from_probs(current_probs, state.label)
                if rank >= target_k:
                    break
                next_token_idx: Optional[int] = None
                candidate_spans: List[Tuple[int, int]] = []
                candidate_text: Optional[str] = None
                for token_idx in ordered_indices:
                    if token_idx in used_indices:
                        continue
                    score = scores[token_idx] if token_idx < scores.size else float("-inf")
                    if not np.isfinite(score):
                        used_indices.add(token_idx)
                        continue
                    token_text = state.term_texts[token_idx]
                    if self._should_ignore(token_text):
                        used_indices.add(token_idx)
                        if self.mask_all_instances:
                            for related_idx in state.term_indices_by_text.get(token_text, []):
                                used_indices.add(related_idx)
                        continue
                    base_span = state.term_spans[token_idx]
                    expanded = self._expand_candidate_spans(state, base_span, token_text, span_set)
                    if not expanded:
                        used_indices.add(token_idx)
                        continue
                    next_token_idx = token_idx
                    candidate_spans = expanded
                    candidate_text = token_text
                    break
                if next_token_idx is None or not candidate_spans:
                    break
                new_spans: List[Tuple[int, int]] = []
                for start, end in candidate_spans:
                    span_tuple = (start, end)
                    if span_tuple in span_set:
                        continue
                    new_spans.append(span_tuple)
                if not new_spans:
                    used_indices.add(next_token_idx)
                    continue
                for start, end in new_spans:
                    annotations.append(
                        TextAnnotation(
                            start=start,
                            end=end,
                            text=state.text[start:end],
                            replacement=self.mask_text,
                        )
                    )
                    span_tuple = (start, end)
                    span_set.add(span_tuple)
                    for mapped_idx in span_to_indices.get(span_tuple, []):
                        used_indices.add(mapped_idx)
                if candidate_text is not None:
                    for related_idx in state.term_indices_by_text.get(candidate_text, []):
                        used_indices.add(related_idx)
                used_indices.add(next_token_idx)
                spans.extend(new_spans)
                current_probs = self._evaluate_state(state, spans)
        aligned_history: Dict[str, List[TextAnnotation]] = {}
        for idx in pending:
            state = self._records_by_idx[idx]
            aligned_history[state.uid] = self._align_annotations(state, self.annotations[state.uid])
        self._annotation_history[target_k] = self._clone_annotation_dict({
            uid: self._align_annotations(self._records_by_uid[uid], anns)
            for uid, anns in self.annotations.items()
        })
        self._k_processed[target_k].update(pending)

    def _grid_anonymize_stream_from_dataset(
        self,
        idx: int,
        k_values: List[int],
        **kwargs,
    ) -> Iterator[Tuple[int, List[AnonymizationResult]]]:
        if idx < 0 or idx >= len(self._records_by_idx):
            raise IndexError(f"Index {idx} is out of bounds")
        ordered_k = [int(value) for value in dict.fromkeys(k_values)]
        for k_value in ordered_k:
            self._ensure_annotations_for_k(k_value, [idx])
        state = self._records_by_idx[idx]
        record = self.dataset_records[idx]
        text = record.text or state.text
        for k_value in ordered_k:
            raw_annotations = self._annotation_history.get(k_value, {}).get(state.uid, [])
            aligned_annotations = self._align_annotations(state, raw_annotations)
            spans = [(ann.start, ann.end) for ann in aligned_annotations]
            masked_text = self._apply_spans_to_text(text, spans)
            cloned_annotations = self._clone_annotation_dict({state.uid: aligned_annotations}).get(state.uid, [])
            yield k_value, [
                AnonymizationResult(
                    text=masked_text,
                    spans=cloned_annotations,
                    metadata={
                        "k": k_value,
                        "perturbed_tokens": len(spans),
                        "method": "petre",
                        "uid": state.uid,
                    },
                )
            ]

    def batch_grid_anonymize_from_dataset(
        self,
        indices: List[int],
        k_values: List[int],
        *,
        progress: bool = False,
        **kwargs,
    ) -> Dict[int, List[List[AnonymizationResult]]]:
        if isinstance(indices, int):
            indices = [indices]
        else:
            indices = list(indices)
        if not indices:
            raise ValueError("indices cannot be empty")
        if isinstance(k_values, int):
            ordered_k = [k_values]
        else:
            ordered_k = list(dict.fromkeys(k_values))
        for k_value in ordered_k:
            self._ensure_annotations_for_k(k_value, indices)
        results: Dict[int, List[List[AnonymizationResult]]] = {k_value: [] for k_value in ordered_k}
        iterator = indices
        if progress:
            from tqdm import tqdm
            iterator = tqdm(iterator, desc="PETRE batch anonymization")
        for idx in iterator:
            per_idx = self._grid_anonymize_from_dataset(idx, ordered_k, **kwargs)
            for k_value in ordered_k:
                results[k_value].append(per_idx[k_value])
        return results

    def batch_anonymize_from_dataset(
        self,
        indices: List[int],
        k: List[int],
        *,
        progress: bool = False,
        **kwargs,
    ) -> List[List[AnonymizationResult]] | Dict[int, List[List[AnonymizationResult]]]:
        if isinstance(indices, int):
            indices = [indices]
        else:
            indices = list(indices)
        if isinstance(k, int):
            ordered_k = [k]
        else:
            ordered_k = list(dict.fromkeys(k))
        if len(ordered_k) > 1:
            return self.batch_grid_anonymize_from_dataset(
                indices,
                ordered_k,
                progress=progress,
                **kwargs,
            )
        single_k = ordered_k[0]
        aggregated: List[List[AnonymizationResult]] = []
        iterator = indices
        if progress:
            from tqdm import tqdm
            iterator = tqdm(iterator, desc="PETRE batch anonymization")
        for idx in iterator:
            per_idx = self._grid_anonymize_from_dataset(idx, [single_k], **kwargs)
            aggregated.append(per_idx[single_k])
        return aggregated

    def anonymize_from_dataset(
        self,
        idx: int,
        k: int,
        *,
        progress: bool = False,
        **kwargs,
    ) -> AnonymizationResult:
        per_idx = self._grid_anonymize_from_dataset(idx, [k], **kwargs)
        results = per_idx.get(k)
        if not results:
            raise ValueError(f"No anonymization result produced for k={k}")
        return results[0]
