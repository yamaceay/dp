from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Set, Tuple
from collections import defaultdict
import re
import numpy as np
import torch
from tqdm import tqdm

from transformers import pipeline

from dp.methods.anonymizer import AnonymizationResult
from dp.methods.k_anon import KAnonymizer
from dp.loaders.base import DatasetRecord, TextAnnotation
from dp.utils.splitter import TextSplitter
from dp.methods.constants import SIMPLE_MODEL_LIST

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
        self._k_cache: Dict[int, bool] = {}
        self._annotation_history: Dict[int, Dict[str, List[TextAnnotation]]] = {}
        self.dataset_records: List[DatasetRecord] = []
        self._records_by_idx: List[RecordState] = []
        self._records_by_uid: Dict[str, RecordState] = {}
        self.label_to_name: Dict[int, str] = {}
        self.name_to_label: Dict[str, int] = {}
        self.num_labels: int = 0
        self._starting_annotations: Dict[str, List[TextAnnotation]] = {}
        self.annotations: Dict[str, List[TextAnnotation]] = {}

    def add_dataset_records(self, dataset_records: List[DatasetRecord]):
        if not dataset_records:
            raise ValueError("dataset_records cannot be empty")
        self.dataset_records = list(dataset_records)
        self._build_label_mappings(self.dataset_records)
        self._build_record_states(self.dataset_records)
        self._starting_annotations = {state.uid: [] for state in self._records_by_idx}
        self.annotations = {uid: [] for uid in self._starting_annotations}
        self._k_cache.clear()
        self._annotation_history.clear()

    def _build_terms_to_ignore(self, annotations: Dict[str, List[TextAnnotation]], name: Optional[str]) -> Set[str]:
        marks = {
            self.mask_text,
            "[CLS]",
            "[SEP]",
            "[PAD]",
            "",
            " ",
            "\t",
            "\n",
        }
        if name is not None and name in SIMPLE_MODEL_LIST:
            for uid, anns in annotations.items():
                for ann in anns:
                    if ann.label:
                        marks.add(ann.label)
        return marks

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

    def set_scoring_strategy(self, explainer) -> None:
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
            max_length=512,
            truncation=True,
        )

    def set_annotations(self, annotations: Dict[str, List[TextAnnotation]], name: Optional[str]) -> None:
        if not self._records_by_idx:
            raise RuntimeError("Dataset records must be added before setting annotations")
        normalized: Dict[str, List[TextAnnotation]] = {}
        for state in self._records_by_idx:
            uid = state.uid
            if annotations and uid in annotations:
                normalized[uid] = self._normalize_annotation_list(annotations[uid])
            else:
                normalized[uid] = []
        self._starting_annotations = self._clone_annotation_dict(normalized)
        self.annotations = self._clone_annotation_dict(normalized)
        self._k_cache.clear()
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
        split_texts: List[str] = []
        for sentence_span in state.sentence_spans:
            rendered = self._apply_spans_to_sentence(state.text, sentence_span, unique_spans)
            if rendered.strip():
                split_texts.append(rendered)
            else:
                split_texts.append(self.mask_text)
        if not split_texts:
            split_texts = [self.mask_text]
        results = self.tri_pipeline(split_texts, batch_size=self.batch_size)
        probs = np.zeros(self.num_labels, dtype=float)
        for split_result in results:
            for pred in split_result:
                label_idx = self._parse_label(pred["label"])
                if 0 <= label_idx < self.num_labels:
                    probs[label_idx] += float(pred["score"])
        probs /= float(len(split_texts))
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

    def _collect_candidates(
        self,
        state: RecordState,
        span_set: Set[Tuple[int, int]],
    ) -> List[Tuple[int, Tuple[int, int], str]]:
        candidates: List[Tuple[int, Tuple[int, int], str]] = []
        for idx, span in enumerate(state.term_spans):
            span_tuple = (span[0], span[1])
            if span_tuple in span_set:
                continue
            if self._span_overlaps_existing(span_tuple, span_set):
                continue
            term_text = state.term_texts[idx]
            if self._should_ignore(term_text):
                continue
            candidates.append((idx, span_tuple, term_text))
        return candidates

    def _select_candidate(
        self,
        state: RecordState,
        current_spans: List[Tuple[int, int]],
        span_set: Set[Tuple[int, int]],
        current_probs: np.ndarray,
        candidates: List[Tuple[int, Tuple[int, int], str]],
    ) -> Optional[Tuple[List[Tuple[int, int]], np.ndarray]]:
        label_idx = state.label
        baseline_prob = current_probs[label_idx]
        best_spans: Optional[List[Tuple[int, int]]] = None
        best_probs: Optional[np.ndarray] = None
        best_drop = float("-inf")
        for _, base_span, term_text in candidates:
            candidate_spans = self._expand_candidate_spans(state, base_span, term_text, span_set)
            test_spans = current_spans + candidate_spans
            probs = self._evaluate_state(state, test_spans)
            drop = baseline_prob - probs[label_idx]
            if drop > best_drop + 1e-12 or (abs(drop - best_drop) <= 1e-12 and best_spans is None):
                best_drop = drop
                best_spans = candidate_spans
                best_probs = probs
        if best_spans is None or best_probs is None:
            return None
        return best_spans, best_probs

    def _base_annotations_for(self, target_k: int) -> Dict[str, List[TextAnnotation]]:
        if not self._annotation_history:
            return self._starting_annotations
        eligible = [k for k in self._annotation_history if k <= target_k]
        if not eligible:
            return self._starting_annotations
        best_k = max(eligible)
        return self._annotation_history[best_k]

    def _run_petre_for_k(self, target_k: int, progress: bool = False) -> None:
        if self.tri_pipeline is None:
            raise RuntimeError("Scoring strategy must be set before running PETRE")
        base = self._clone_annotation_dict(self._base_annotations_for(target_k))
        self.annotations = self._clone_annotation_dict(base)
        record_iter = self._records_by_idx
        if progress:
            record_iter = tqdm(record_iter, desc=f"Running PETRE for k={target_k}")
        for state in record_iter:
            annotations = self.annotations[state.uid]
            spans = [(ann.start, ann.end) for ann in annotations]
            span_set = set(spans)
            current_probs = self._evaluate_state(state, spans)
            while True:
                rank = self._rank_from_probs(current_probs, state.label)
                if rank >= target_k:
                    break
                candidates = self._collect_candidates(state, span_set)
                if not candidates:
                    break
                selected = self._select_candidate(state, spans, span_set, current_probs, candidates)
                if selected is None:
                    break
                candidate_spans, candidate_probs = selected
                new_spans = []
                for candidate_span in candidate_spans:
                    if candidate_span in span_set:
                        continue
                    new_spans.append(candidate_span)
                if not new_spans:
                    break
                for start, end in new_spans:
                    annotations.append(
                        TextAnnotation(
                            start=start,
                            end=end,
                            text=state.text[start:end],
                            replacement=self.mask_text,
                        )
                    )
                    span_set.add((start, end))
                spans.extend(new_spans)
                current_probs = candidate_probs
        self._annotation_history[target_k] = self._clone_annotation_dict(self.annotations)

    def grid_anonymize_from_dataset(
        self,
        idx: int,
        k: List[int],
        *args,
        progress: bool = False,
        **kwargs,
    ) -> List[AnonymizationResult]:
        if idx < 0 or idx >= len(self._records_by_idx):
            raise IndexError(f"Index {idx} is out of bounds")
        requested_order = list(k)
        unique_sorted_k = sorted(set(requested_order))
        for current_k in unique_sorted_k:
            if current_k not in self._k_cache:
                self._run_petre_for_k(current_k, progress=progress)
                self._k_cache[current_k] = True
        state = self._records_by_idx[idx]
        record = self.dataset_records[idx]
        uid = state.uid
        text = record.text or state.text
        results: List[AnonymizationResult] = []
        for k_value in requested_order:
            annotations = self._annotation_history.get(k_value, {}).get(uid, [])
            spans = [(ann.start, ann.end) for ann in annotations]
            masked_text = self._apply_spans_to_text(text, spans)
            cloned_annotations = self._clone_annotation_dict({uid: annotations}).get(uid, [])
            results.append(
                AnonymizationResult(
                    text=masked_text,
                    spans=cloned_annotations,
                    metadata={
                        "k": k_value,
                        "perturbed_tokens": len(spans),
                        "method": "petre",
                        "uid": uid,
                    },
                )
            )
        return results

    def anonymize_from_dataset(
        self,
        idx: int,
        k: int,
        *args,
        progress: bool = False,
        **kwargs,
    ) -> AnonymizationResult:
        return self.grid_anonymize_from_dataset(idx, k=[k], *args, progress=progress, **kwargs)[0]
