from __future__ import annotations

from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple
from collections import defaultdict
import string
import numpy as np
import torch

from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForMaskedLM
from nltk.tokenize.treebank import TreebankWordDetokenizer

from dp.methods.k_anon import KAnonymizer
from dp.methods.k_anon._petre import RecordState
from dp.methods.constants import PII_CLASSIFIER_MODEL_LIST, RISK_MASKER_MODEL_LIST
from dp.methods.anonymizer import AnonymizationResult
from dp.utils.explainer.base import TokenExplainer
from dp.loaders.base import DatasetRecord, TextAnnotation
from dp.utils.splitter import TextSplitter
from dp.utils.chunking import TokenAwareChunker
from dp.utils.memory import clear_memory


class KDPMLMAnonymizer(KAnonymizer):
    def __init__(
        self,
        *args,
        epsilon: float = 1.0,
        model_checkpoint: str = "roberta-base",
        clip_min: float = -3.2093127,
        clip_max: float = 16.304797887802124,
        k_candidates: int = 5,
        use_temperature: bool = True,
        sort_tokens_by_risk: bool = True,
        risk_temperature: Optional[float] = None,
        batch_size: int = 32,
        mask_text: str = "[MASK]",
        device: str = "auto",
        use_chunking: bool = True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.epsilon = float(epsilon)
        self.model_checkpoint = model_checkpoint
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.sensitivity = abs(clip_max - clip_min)
        self.k_candidates = k_candidates
        self.use_temperature = use_temperature
        self.sort_tokens_by_risk = sort_tokens_by_risk
        self.risk_temperature = risk_temperature
        self.explainer: Optional[TokenExplainer] = None
        self.batch_size = batch_size
        self.mask_text = mask_text
        self.device = self._resolve_device(device)
        self.use_chunking = use_chunking
        self.splitter = TextSplitter()
        self.detokenizer = TreebankWordDetokenizer()
        self._risk_scores: Dict[str, Tuple[List[Tuple[int, int]], List[float]]] = {}
        self._prepared_risk_scores: Dict[str, np.ndarray] = {}
        self._score_cache: Dict[str, np.ndarray] = {}
        self._score_order_cache: Dict[str, List[int]] = {}
        self.dataset_records: List[DatasetRecord] = []
        self._records_by_idx: List[RecordState] = []
        self._records_by_uid: Dict[str, RecordState] = {}
        self.label_to_name: Dict[int, str] = {}
        self.name_to_label: Dict[str, int] = {}
        self.num_labels: int = 0
        self._starting_annotations: Dict[str, List[TextAnnotation]] = {}
        self.annotations: Dict[str, List[TextAnnotation]] = {}
        self._annotation_name: Optional[str] = None
        self._terms_to_ignore = set()
        self._special_tokens = {self.mask_text, "[CLS]", "[SEP]", "[PAD]", ""}
        self.tri_pipeline_path: Optional[str] = None
        self.tri_pipeline = None
        self.tri_chunker: Optional[TokenAwareChunker] = None
        self._mlm_tokenizer = AutoTokenizer.from_pretrained(self.model_checkpoint)
        self._mlm_model = AutoModelForMaskedLM.from_pretrained(self.model_checkpoint).to(self.device)

    def add_dataset_records(self, dataset_records: List[DatasetRecord]):
        if not dataset_records:
            raise ValueError("dataset_records cannot be empty")
        self.dataset_records = list(dataset_records)
        self._build_label_mappings(self.dataset_records)
        self._build_record_states(self.dataset_records)
        self._starting_annotations = {state.uid: [] for state in self._records_by_idx}
        self.annotations = {uid: [] for uid in self._starting_annotations}
        self._terms_to_ignore = self._build_terms_to_ignore(self.annotations, self._annotation_name)
        self._clear_score_cache()
        self._prepare_risk_scores_for_records()

    def set_risk_scores(
        self,
        risk_scores: Dict[str, Dict[str, object]],
        records: Optional[Sequence[DatasetRecord]] = None,
    ) -> None:
        self._risk_scores = {}
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
            self._risk_scores[uid] = (normalized_offsets, score_list)
        self._clear_score_cache()
        self._prepare_risk_scores_for_records()

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
        aligned: Dict[str, List[TextAnnotation]] = {}
        for state in self._records_by_idx:
            uid = state.uid
            raw_items = annotations.get(uid, []) if annotations and uid in annotations else []
            normalized = self._normalize_annotation_list(raw_items)
            aligned[uid] = self._align_annotations(state, normalized)
        self._starting_annotations = self._clone_annotation_dict(aligned)
        self.annotations = self._clone_annotation_dict(aligned)
        if name:
            self._annotation_name = name
        self._terms_to_ignore = self._build_terms_to_ignore(self.annotations, self._annotation_name)

    def _grid_anonymize_stream_from_dataset(
        self,
        idx: int,
        k_values: List[int],
        **kwargs,
    ) -> Iterator[Tuple[int, List[AnonymizationResult]]]:
        if idx < 0 or idx >= len(self._records_by_idx):
            raise IndexError(f"Index {idx} is out of bounds")
        ordered_k = [int(value) for value in dict.fromkeys(k_values)]
        state = self._records_by_idx[idx]
        for k_value in ordered_k:
            result = self._rewrite_until_k(state, k_value)
            yield k_value, [result]

    def _rewrite_until_k(self, state: RecordState, target_k: int) -> AnonymizationResult:
        text = state.text
        scores = self._token_scores_for_state(state)
        ordered = self._ordered_token_indices_for_state(state)
        if not ordered:
            ordered = list(range(len(state.term_spans)))
        processed = set()
        perturbed = 0
        while True:
            probs = self._evaluate_text(state, text)
            rank = self._rank_from_probs(probs, state.label)
            if rank >= target_k:
                break
            next_idx = self._next_token_index(ordered, processed, state)
            if next_idx is None:
                break
            processed.add(next_idx)
            tokens, offsets = self._tokenize(text)
            if next_idx >= len(tokens):
                break
            token = tokens[next_idx]
            if token in string.punctuation:
                continue
            new_token = self._privatize_token(text, token, offsets[next_idx], self.epsilon)
            original = text[offsets[next_idx][0]:offsets[next_idx][1]]
            if len(new_token) == len(original):
                new_token = "".join(
                    p.upper() if o.isupper() else p.lower()
                    for p, o in zip(new_token, original)
                )
            elif original and original[0].isupper():
                new_token = new_token.capitalize()
            text = text[:offsets[next_idx][0]] + new_token + text[offsets[next_idx][1]:]
            perturbed += int(new_token != token)
        metadata = {
            "k": target_k,
            "perturbed_tokens": perturbed,
            "method": "kdpmlm",
            "uid": state.uid,
            "epsilon": self.epsilon,
            "tri_model": self.tri_pipeline_path,
            "model": self.model_checkpoint,
        }
        return AnonymizationResult(text=text, metadata=metadata)

    def _next_token_index(self, ordered: List[int], processed: set, state: RecordState) -> Optional[int]:
        for idx in ordered:
            if idx in processed:
                continue
            token_text = state.term_texts[idx] if idx < len(state.term_texts) else ""
            if self._should_ignore(token_text):
                processed.add(idx)
                continue
            return idx
        return None

    def _tokenize(self, text: str) -> Tuple[List[str], List[Tuple[int, int]]]:
        tokens: List[str] = []
        offsets: List[Tuple[int, int]] = []
        for start, end, token in self.splitter.tokenize_with_spans(text):
            tokens.append(token)
            offsets.append((start, end))
        return tokens, offsets

    def _privatize_token(
        self,
        sentence: str,
        token: str,
        offset: Tuple[int, int],
        epsilon: float,
    ) -> str:
        masked_sentence = sentence[: offset[0]] + self._mlm_tokenizer.mask_token + sentence[offset[1]:]
        if masked_sentence == sentence:
            return token
        input_ids = self._mlm_tokenizer.encode(masked_sentence, add_special_tokens=True, truncation=True, max_length=512)
        try:
            mask_pos = input_ids.index(self._mlm_tokenizer.mask_token_id)
        except ValueError:
            return token
        model_input = torch.tensor(input_ids).reshape(1, -1).to(self.device)
        with torch.no_grad():
            output = self._mlm_model(model_input)
        logits = output[0].squeeze().detach().cpu().numpy()
        mask_logits = logits[mask_pos]
        if self.use_temperature:
            temperature = 2 * self.sensitivity / epsilon
            mask_logits = np.clip(mask_logits, self.clip_min, self.clip_max)
            mask_logits = mask_logits / temperature
            scores = torch.softmax(torch.from_numpy(mask_logits), dim=0)
            scores = scores / scores.sum()
            chosen_idx = np.random.choice(len(mask_logits), p=scores.numpy())
            return self._mlm_tokenizer.decode(chosen_idx).strip()
        top_tokens = torch.topk(torch.from_numpy(mask_logits), k=self.k_candidates, dim=0)[1]
        return self._mlm_tokenizer.decode(top_tokens[0].item()).strip()

    def _weights_to_probs(self, weights: np.ndarray, temperature: float) -> np.ndarray:
        if temperature is None:
            temperature = 1.0
        scores = np.asarray(weights, dtype=float)
        if scores.size == 0:
            return scores
        positive_scores = np.exp(scores / temperature)
        probs = positive_scores / positive_scores.sum()
        return probs

    def _token_scores_for_state(self, state: RecordState) -> np.ndarray:
        cached = self._score_cache.get(state.uid)
        if cached is not None:
            return cached
        precomputed = self._prepared_risk_scores.get(state.uid)
        if precomputed is not None:
            self._score_cache[state.uid] = precomputed
            self._score_order_cache[state.uid] = list(np.argsort(-precomputed, kind="mergesort"))
            return precomputed
        if self.explainer is None:
            empty = np.zeros(len(state.term_spans), dtype=float)
            self._score_cache[state.uid] = empty
            return empty
        scores = self.explainer.explain(state.text, state.term_spans)
        array = np.asarray(scores, dtype=float).ravel()
        length = len(state.term_spans)
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
            weights = scores
            if self.risk_temperature is not None:
                weights = self._weights_to_probs(scores, self.risk_temperature)
            ordered = list(np.argsort(-weights, kind="mergesort"))
        self._score_order_cache[state.uid] = ordered
        if self.sort_tokens_by_risk:
            return ordered
        return list(range(len(state.term_spans)))

    def _evaluate_text(self, state: RecordState, text: str) -> np.ndarray:
        if self.tri_pipeline is None:
            raise RuntimeError("Scoring strategy must be set before running KDPMLM")
        inputs: List[str] = []
        for sentence_span in state.sentence_spans:
            segment = text[sentence_span[0]:sentence_span[1]]
            if not segment.strip():
                segment = self.mask_text
            if self.tri_chunker is not None:
                chunks = self.tri_chunker.chunk(segment)
                if not chunks:
                    inputs.append(segment)
                else:
                    for chunk in chunks:
                        chunk_text = chunk.text
                        inputs.append(chunk_text if chunk_text.strip() else self.mask_text)
            else:
                inputs.append(segment)
        if not inputs:
            inputs = [self.mask_text]
        results = self.tri_pipeline(inputs, batch_size=self.batch_size)
        probs = np.zeros(self.num_labels, dtype=float)
        for split_result in results:
            for pred in split_result:
                label_idx = self._parse_label(pred["label"])
                if 0 <= label_idx < self.num_labels:
                    probs[label_idx] += float(pred["score"])
        num_inputs = max(len(inputs), 1)
        probs /= float(num_inputs)
        return probs

    def _rank_from_probs(self, probs: np.ndarray, label_idx: int) -> int:
        sorted_indices = np.argsort(probs)[::-1]
        positions = np.where(sorted_indices == label_idx)[0]
        if positions.size == 0:
            return len(sorted_indices) + 1
        return int(positions[0]) + 1

    def _build_label_mappings(self, dataset_records: List[DatasetRecord]) -> None:
        names = {record.name or record.uid or f"record_{idx}" for idx, record in enumerate(dataset_records)}
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

    def _parse_label(self, label: str) -> int:
        if "_" in label:
            return int(label.split("_")[-1])
        return int(label)

    def _should_ignore(self, text: str) -> bool:
        if not text.strip():
            return True
        if text in self._special_tokens:
            return True
        normalized = text.lower()
        if normalized in self._terms_to_ignore:
            return True
        return False

    def _build_terms_to_ignore(self, annotations: Dict[str, List[TextAnnotation]], name: Optional[str]) -> set:
        marks = {self.mask_text, "", " ", "\t", "\n"}
        if name in PII_CLASSIFIER_MODEL_LIST or name in RISK_MASKER_MODEL_LIST or name == "manual":
            for uid, anns in annotations.items():
                for ann in anns:
                    if ann.label:
                        marks.add(ann.label)
        normalized_marks = set()
        for mark in marks:
            if mark is None:
                continue
            normalized_marks.add(mark.lower())
        return normalized_marks

    def _normalize_annotation_list(self, items: Iterable) -> List[TextAnnotation]:
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
                normalized.append(TextAnnotation(start=int(item[0]), end=int(item[1])))
        normalized.sort(key=lambda ann: (ann.start, ann.end))
        return normalized

    def _align_annotations(self, state: RecordState, annotations: List[TextAnnotation]) -> List[TextAnnotation]:
        if not annotations:
            return []
        aligned: List[TextAnnotation] = []
        seen = set()
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

    def _expand_span_to_tokens(self, state: RecordState, start: int, end: int) -> Tuple[int, int]:
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
        if not matched or expanded_start >= expanded_end:
            return start, end
        return expanded_start, expanded_end

    def _clone_annotation_dict(self, data: Dict[str, List[TextAnnotation]]) -> Dict[str, List[TextAnnotation]]:
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

    def _prepare_risk_scores_for_records(self) -> None:
        if not self._risk_scores or not self._records_by_uid:
            return
        self._prepared_risk_scores = {}
        for uid, state in self._records_by_uid.items():
            raw = self._risk_scores.get(uid)
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

    def _pipeline_device(self):
        if self.device.type == "cpu":
            return -1
        if self.device.type == "cuda":
            return self.device.index or 0
        return self.device

    def _resolve_device(self, device: str) -> torch.device:
        if device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        return torch.device(device)

    def _clone_annotations(self, annotations: Dict[str, List[TextAnnotation]]) -> Dict[str, List[TextAnnotation]]:
        return self._clone_annotation_dict(annotations)

    def __del__(self):
        clear_memory()
