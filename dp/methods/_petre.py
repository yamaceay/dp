from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union
from collections import defaultdict
import re
import numpy as np
import torch

from transformers import pipeline

from dp.utils.explainer.base import TokenExplainer
from dp.methods.anonymizer import Anonymizer, AnonymizationResult
from dp.methods.constants import Buckets, BucketDict, KParams
from dp.loaders.base import DatasetRecord, TextAnnotation, TextAnnotations, TokenEdit
from dp.utils.splitter import TextSplitter
from dp.utils.chunking import TokenAwareChunker
from dp.utils.token_ledger import TokenLedger
from dp.utils.selector.base import AnonymizerUnit, ApplyFn
from dp.utils.selector.until_k_selector import UntilKUnit, RankEvaluator

DEFAULT_STOPWORDS: Set[str] = {
    'a', 'about', 'above', 'after', 'again', 'against', 'ain', 'all', 'am', 'an', 'and', 'any', 'are', 'aren', "aren't", 'as', 'at', 
    'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both', 'but', 'by', 
    'can', 'couldn', "couldn't", 
    'd', 'did', 'didn', "didn't", 'do', 'does', 'doesn', "doesn't", 'doing', 'don', "don't", 'down', 'during', 
    'each', 
    'few', 'for', 'from', 'further', 
    'had', 'hadn', "hadn't", 'has', 'hasn', "hasn't", 'have', 'haven', "haven't", 'having', 'he', "he'd", "he'll", 'her', 'here', 'hers', 'herself', "he's", 'him', 'himself', 'his', 'how', 
    'i', "i'd", 'if', "i'll", "i'm", 'in', 'into', 'is', 'isn', "isn't", 'it', "it'd", "it'll", "it's", 'its', 'itself', "i've", 
    'just', 
    'll', 
    'm', 'ma', 'me', 'mightn', "mightn't", 'more', 'most', 'mustn', "mustn't", 'my', 'myself', 
    'needn', "needn't", 'no', 'nor', 'not', 'now', 
    'o', 'of', 'off', 'on', 'once', 'only', 'or', 'other', 'our', 'ours', 'ourselves', 'out', 'over', 'own', 
    're', 
    's', 'same', 'shan', "shan't", 'she', "she'd", "she'll", "she's", 'should', 'shouldn', "shouldn't", "should've", 'so', 'some', 'such', 
    't', 'than', 'that', "that'll", 'the', 'their', 'theirs', 'them', 'themselves', 'then', 'there', 'these', 'they', "they'd", "they'll", "they're", "they've", 'this', 'those', 'through', 'to', 'too', 
    'under', 'until', 'up', 
    've', 'very', 
    'was', 'wasn', "wasn't", 'we', "we'd", "we'll", "we're", 'were', 'weren', "weren't", "we've", 'what', 'when', 'where', 'which', 'while', 'who', 'whom', 'why', 'will', 'with', 'won', "won't", 'wouldn', "wouldn't", 
    'y', 'you', "you'd", "you'll", 'your', "you're", 'yours', 'yourself', 'yourselves', "you've"
}

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


class PetreAnonymizer(Anonymizer):
    MODEL_NAME = "petre"
    def __init__(
        self,
        mask_text: str = "[MASK]",
        device: str = "auto",
        use_chunking: bool = True,
        mask_all_instances: bool = False,
        batch_size: int = 32,
        *args,
        **kwargs,
    ):
        super().__init__(*args, model=self.MODEL_NAME, **kwargs)
        self.mask_text = mask_text
        self.use_chunking = use_chunking
        self.mask_all_instances = mask_all_instances
        self.batch_size = batch_size
        self.device = self._resolve_device(device)
        self.splitter = TextSplitter()
        self._explainer = None
        self._unit: Optional[UntilKUnit] = None
        self.tri_pipeline_path: Optional[str] = None
        self.tri_pipeline = None
        self._special_pattern = re.compile(r"[^\nA-Za-z0-9À-ÖØ-öø-ÿЀ-ӿ/]+")
        self._terms_to_ignore = set()
        self.dataset_records: List[DatasetRecord] = []
        self._records_by_idx: List[RecordState] = []
        self._records_by_uid: Dict[str, RecordState] = {}
        self.label_to_name: Dict[int, str] = {}
        self.name_to_label: Dict[str, int] = {}
        self.num_labels: int = 0
        self.tri_chunker: Optional[TokenAwareChunker] = None
        self._score_cache: Dict[str, np.ndarray] = {}
        self._score_order_cache: Dict[str, List[int]] = {}
        self._raw_risk_scores: Dict[str, Tuple[List[Tuple[int, int]], List[float]]] = {}
        self._prepared_risk_scores: Dict[str, np.ndarray] = {}

    def set_unit(self, unit: AnonymizerUnit) -> None:
        self._unit = unit

    def add_dataset_records(self, dataset_records: List[DatasetRecord]):
        if not dataset_records:
            raise ValueError("dataset_records cannot be empty")
        self.dataset_records = list(dataset_records)
        self._build_label_mappings(self.dataset_records)
        self._build_record_states(self.dataset_records)
        self.tri_chunker = None
        self._terms_to_ignore = self._build_terms_to_ignore({}, None)
        self._clear_score_cache()
        self._prepare_risk_scores_for_records()

    def pre_stream_anonymize(self, *args, risk_scores: Optional[Dict[str, Dict[str, object]]] = None, **kwargs) -> None:
        if risk_scores is not None:
            self.set_risk_scores(risk_scores, records=self.dataset_records or None)

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

    def _get_stopwords(self) -> Set[str]:
        return set(DEFAULT_STOPWORDS)

    def _build_terms_to_ignore(self, annotations: Dict[str, List[TextAnnotation]], name: Optional[str]) -> Set[str]:
        stopwords = self._get_stopwords()
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

    def _resolve_device(self, device: Optional[Union[str, int, torch.device]]) -> torch.device:
        return super()._resolve_device(device)

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
        if self._explainer is None:
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
        raw_scores = self._explainer.explain(state.text, state.term_spans)
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

    def set_explainer(self, explainer: TokenExplainer) -> None:
        self._clear_score_cache()
        super().set_explainer(explainer)
        self._load_tri_pipeline()

    def _load_tri_pipeline(self) -> None:
        if not hasattr(self._explainer, "tri_detector"):
            raise ValueError("explainer must expose tri_detector")
        tri_model_name = getattr(self._explainer.tri_detector, "model_name", None)
        if tri_model_name is None:
            raise ValueError("explainer.tri_detector must define model_name")
        self.tri_pipeline_path = tri_model_name
        self.batch_size = int(getattr(self._explainer, "batch_size", self.batch_size))
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

    def _apply_spans_to_text(
        self,
        text: str,
        spans: List[Tuple[int, int]],
    ) -> str:
        if not spans:
            return text
        ledger = TokenLedger(text, tuple(sorted(set(spans))))
        span_set = set(spans)
        for idx, entry in enumerate(ledger.iter_entries()):
            if (entry.start, entry.end) in span_set:
                ledger.replace(idx, self.mask_text)
        return ledger.render_offsets(text)

    def _evaluate_state(
        self,
        state: RecordState,
        spans: List[Tuple[int, int]],
    ) -> np.ndarray:
        unique_spans = sorted({(start, end) for start, end in spans})
        pipeline_inputs: List[str] = []
        for sentence_span in state.sentence_spans:
            sent_start, sent_end = sentence_span
            sentence_text = state.text[sent_start:sent_end]
            relevant_spans = [(max(s[0], sent_start) - sent_start, min(s[1], sent_end) - sent_start) 
                             for s in unique_spans 
                             if not (s[1] <= sent_start or s[0] >= sent_end)]
            if relevant_spans:
                ledger = TokenLedger(sentence_text, tuple(sorted(set(s for s in state.term_spans if sent_start <= s[0] and s[1] <= sent_end))))
                for idx, entry in enumerate(ledger.iter_entries()):
                    if (entry.start - sent_start, entry.end - sent_start) in relevant_spans:
                        ledger.replace(idx, self.mask_text)
                rendered = ledger.render_offsets(sentence_text)
            else:
                rendered = sentence_text
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
        clean = self._special_pattern.sub("", text).strip()
        return not clean or clean.lower() in self._terms_to_ignore

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


    def _ensure_annotations_for_k(self, target_k: int, state: RecordState, starting_spans: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        if self.tri_pipeline is None:
            raise RuntimeError("Scoring strategy must be set before running PETRE")
        
        spans: List[Tuple[int, int]] = list(starting_spans)
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
            
            spans.extend(new_spans)
            for start, end in new_spans:
                span_tuple = (start, end)
                span_set.add(span_tuple)
                for mapped_idx in span_to_indices.get(span_tuple, []):
                    used_indices.add(mapped_idx)
            
            if candidate_text is not None:
                for related_idx in state.term_indices_by_text.get(candidate_text, []):
                    used_indices.add(related_idx)
            
            used_indices.add(next_token_idx)
            current_probs = self._evaluate_state(state, spans)
        
        return spans

    def _make_rank_evaluator(self, state: RecordState) -> RankEvaluator:
        def rank_evaluator(current_text: str, target_label: int) -> int:
            spans = self._extract_masked_spans(state, current_text)
            probs = self._evaluate_state(state, spans)
            return self._rank_from_probs(probs, target_label)
        return rank_evaluator

    def _extract_masked_spans(self, state: RecordState, masked_text: str) -> List[Tuple[int, int]]:
        spans: List[Tuple[int, int]] = []
        for idx, span in enumerate(state.term_spans):
            original_token = state.text[span[0]:span[1]]
            if self.mask_text in masked_text:
                spans.append(span)
        return spans

    def _make_apply_fn(
        self,
        state: RecordState,
        runtime_stats: Dict[str, int],
    ) -> ApplyFn:
        def apply_fn(idx: int, ledger: TokenLedger) -> None:
            if idx >= len(state.term_spans):
                return
            
            token_text = state.term_texts[idx]
            if self._should_ignore(token_text):
                return
            
            ledger.replace(idx, self.mask_text)
            runtime_stats["masked"] += 1
            
            if self.mask_all_instances:
                for related_idx in state.term_indices_by_text.get(token_text, []):
                    if related_idx != idx:
                        ledger.replace(related_idx, self.mask_text)
                        runtime_stats["masked"] += 1

        return apply_fn

    def anonymize_from_dataset(
        self,
        idx: int,
        *args,
        buckets: Buckets = [],
        **kwargs,
    ) -> List[Tuple[BucketDict, AnonymizationResult]]:
        if idx < 0 or idx >= len(self._records_by_idx):
            raise IndexError(f"Index {idx} is out of bounds")
        
        if len(buckets) != 1 or not isinstance(buckets[0], KParams):
            raise ValueError("PetreAnonymizer only supports KParams for grid anonymization.")
        k_params: KParams = buckets[0]
        
        state = self._records_by_idx[idx]
        record = self.dataset_records[idx]
        text = record.text or state.text
        
        if self._unit is None:
            self._unit = UntilKUnit()
        
        self._unit.set_thresholds(k_params.values(), name="k")
        
        scores = self._token_scores_for_state(state)
        self._unit.set_risk_scores(scores)
        self._unit.set_target_label(state.label)
        self._unit.set_rank_evaluator(self._make_rank_evaluator(state))
        
        runtime_stats: Dict[str, int] = {"masked": 0}

        apply_fn = self._make_apply_fn(state, runtime_stats)

        outputs: List[Tuple[BucketDict, AnonymizationResult]] = []
        
        ledger = TokenLedger(text, state.term_spans)
        
        prior_edits = record.metadata.get("prior_token_edits", []) if record.metadata else []
        if prior_edits:
            ledger.apply_prior_edits(prior_edits)

        for step in self._unit.anonymize(
            text,
            state.term_spans,
            apply_fn,
            ledger=ledger,
            prior_edits=prior_edits,
        ):
            k_value = step.threshold
            private_text = step.text
            ledger = step.ledger
            
            result_spans: List[TextAnnotation] = []

            for term_idx in step.new_indices:
                result_spans.append(
                    TextAnnotation(
                        start=state.term_spans[term_idx][0],
                        end=state.term_spans[term_idx][1],
                        label="petre",
                        text=state.term_texts[term_idx],
                        replacement=self.mask_text,
                    )
                )
            
            metadata: Dict[str, Any] = {
                "k": k_value,
                "perturbed_tokens": runtime_stats["masked"],
                "method": "petre",
                "uid": state.uid,
                "rank": step.metadata.get("rank"),
                **step.metadata,
            }
            token_edits = [TokenEdit.from_mapping(e) for e in ledger.edits_metadata()]

            hp = BucketDict({"k": int(k_value) if k_value is not None else None})
            outputs.append(
                (
                    hp,
                    AnonymizationResult(
                        text=private_text,
                        annotations=TextAnnotations(spans=result_spans, token_edits=token_edits),
                        metadata=metadata,
                    ),
                )
            )

        return outputs
