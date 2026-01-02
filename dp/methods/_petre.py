from __future__ import annotations

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
from dp.utils.chunking import TokenAwareChunker
from dp.utils.stopwords import build_terms_to_ignore
from dp.utils.token_ledger import TokenLedger
from dp.utils.selector.base import AnonymizerUnit, ApplyFn
from dp.utils.selector.until_k_selector import UntilKUnit, RankEvaluator

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
        self._explainer = None
        self._unit: Optional[UntilKUnit] = None
        self.tri_pipeline_path: Optional[str] = None
        self.tri_pipeline = None
        self._special_pattern = re.compile(r"[^\nA-Za-z0-9À-ÖØ-öø-ÿЀ-ӿ/]+")
        self._terms_to_ignore = set()
        self.dataset_records: List[DatasetRecord] = []
        self.label_to_name: Dict[int, str] = {}
        self.name_to_label: Dict[str, int] = {}
        self.num_labels: int = 0
        self.tri_chunker: Optional[TokenAwareChunker] = None
        self._risk_offsets_by_uid: Dict[str, List[Tuple[int, int]]] = {}
        self._risk_scores_by_uid: Dict[str, np.ndarray] = {}

    def set_unit(self, unit: AnonymizerUnit) -> None:
        self._unit = unit

    def add_dataset_records(self, dataset_records: List[DatasetRecord]):
        if not dataset_records:
            raise ValueError("dataset_records cannot be empty")
        self.dataset_records = list(dataset_records)
        self._build_label_mappings(self.dataset_records)
        self._terms_to_ignore = build_terms_to_ignore(self.mask_text)

    def pre_stream_anonymize(self, *args, risk_scores: Optional[Dict[str, Dict[str, object]]] = None, **kwargs) -> None:
        if risk_scores is not None:
            self.set_risk_scores(risk_scores, records=self.dataset_records or None)

    def set_risk_scores(
        self,
        risk_scores: Dict[str, Dict[str, object]],
        records: Optional[Sequence[DatasetRecord]] = None,
    ) -> None:
        self._risk_offsets_by_uid = {}
        self._risk_scores_by_uid = {}
        if not risk_scores:
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
            order = sorted(range(len(normalized_offsets)), key=lambda i: (normalized_offsets[i][0], normalized_offsets[i][1]))
            ordered_offsets = [normalized_offsets[i] for i in order]
            ordered_scores = [score_list[i] for i in order]
            self._risk_offsets_by_uid[uid] = ordered_offsets
            self._risk_scores_by_uid[uid] = np.asarray(ordered_scores, dtype=float)

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

    def set_explainer(self, explainer: TokenExplainer) -> None:
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
        if tokenizer is not None and self.use_chunking:
            max_tokens = getattr(tokenizer, "model_max_length", 512)
            if max_tokens is None or max_tokens <= 0:
                max_tokens = 512
            self.tri_chunker = TokenAwareChunker(tokenizer, max(max_tokens - 2, 1))
        else:
            self.tri_chunker = None

    def _rank_from_probs(self, probs: np.ndarray, label_idx: int) -> int:
        sorted_indices = np.argsort(probs)[::-1]
        positions = np.where(sorted_indices == label_idx)[0]
        if positions.size == 0:
            return len(sorted_indices) + 1
        return int(positions[0]) + 1

    def _should_ignore(self, text: str) -> bool:
        clean = self._special_pattern.sub("", text).strip()
        return not clean or clean.lower() in self._terms_to_ignore

    def _resolve_risk_uid(self, record_name: Optional[str], record_uid: Optional[str]) -> Optional[str]:
        if record_uid and record_uid in self._risk_offsets_by_uid:
            return record_uid
        if record_name and record_name in self._risk_offsets_by_uid:
            return record_name
        return None

    def _offsets_for_record(self, record_name: Optional[str], record_uid: Optional[str]) -> List[Tuple[int, int]]:
        uid = self._resolve_risk_uid(record_name, record_uid)
        if uid is None:
            raise ValueError("No precomputed offsets available for record")
        offsets = self._risk_offsets_by_uid.get(uid)
        if offsets is None:
            raise ValueError("No precomputed offsets available for record")
        return offsets

    def _scores_for_record(self, record_name: Optional[str], record_uid: Optional[str]) -> np.ndarray:
        uid = self._resolve_risk_uid(record_name, record_uid)
        if uid is None:
            raise ValueError("No precomputed scores available for record")
        scores = self._risk_scores_by_uid.get(uid)
        if scores is None:
            raise ValueError("No precomputed scores available for record")
        return scores

    def _evaluate_text(self, text: str) -> np.ndarray:
        if self.tri_pipeline is None:
            raise RuntimeError("Scoring strategy must be set before running PETRE")
        rendered = text if text.strip() else self.mask_text
        if self.tri_chunker is not None:
            chunks = self.tri_chunker.chunk(rendered)
            if not chunks:
                pipeline_inputs = [rendered]
            else:
                pipeline_inputs = [chunk.text if chunk.text.strip() else self.mask_text for chunk in chunks]
        else:
            pipeline_inputs = [rendered]
        results = self.tri_pipeline(pipeline_inputs, batch_size=min(self.batch_size, len(pipeline_inputs)))
        probs = np.zeros(self.num_labels, dtype=float)
        for split_result in results:
            for pred in split_result:
                label_idx = self._parse_label(pred["label"])
                if 0 <= label_idx < self.num_labels:
                    probs[label_idx] += float(pred["score"])
        num_inputs = max(len(results), 1)
        probs /= float(num_inputs)
        return probs

    def _make_rank_evaluator(self) -> RankEvaluator:
        def rank_evaluator(current_text: str, target_label: int) -> int:
            probs = self._evaluate_text(current_text)
            return self._rank_from_probs(probs, target_label)
        return rank_evaluator

    def _token_texts_and_indices(
        self,
        text: str,
        offsets: List[Tuple[int, int]],
    ) -> Tuple[List[str], Dict[str, List[int]]]:
        token_texts = [text[start:end] for start, end in offsets]
        indices_by_text: Dict[str, List[int]] = defaultdict(list)
        for idx, token_text in enumerate(token_texts):
            indices_by_text[token_text].append(idx)
        return token_texts, indices_by_text

    def _make_apply_fn(
        self,
        offsets: List[Tuple[int, int]],
        token_texts: List[str],
        indices_by_text: Dict[str, List[int]],
        runtime_stats: Dict[str, int],
    ) -> ApplyFn:
        def apply_fn(idx: int, ledger: TokenLedger) -> None:
            if idx >= len(offsets):
                return
            
            token_text = token_texts[idx]
            if self._should_ignore(token_text):
                return
            
            ledger.replace(idx, self.mask_text)
            runtime_stats["masked"] += 1
            
            if self.mask_all_instances:
                for related_idx in indices_by_text.get(token_text, []):
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
        if idx < 0 or idx >= len(self.dataset_records):
            raise IndexError(f"Index {idx} is out of bounds")
        
        if len(buckets) != 1 or not isinstance(buckets[0], KParams):
            raise ValueError("PetreAnonymizer only supports KParams for grid anonymization.")
        k_params: KParams = buckets[0]
        
        record = self.dataset_records[idx]
        text = record.text or ""
        record_name = record.name or record.uid or f"record_{idx}"
        record_uid = record.uid or None
        offsets = self._offsets_for_record(record_name, record_uid)
        scores = self._scores_for_record(record_name, record_uid)
        if len(offsets) != len(scores):
            raise ValueError("Offsets and scores length mismatch for record")
        
        if self._unit is None:
            self._unit = UntilKUnit()
        
        self._unit.set_thresholds(k_params.values(), name="k")
        
        self._unit.set_risk_scores(scores)
        if record_name not in self.name_to_label:
            raise ValueError(f"unknown identity '{record_name}' for record {idx}")
        self._unit.set_target_label(self.name_to_label[record_name])
        self._unit.set_rank_evaluator(self._make_rank_evaluator())
        
        runtime_stats: Dict[str, int] = {"masked": 0}

        token_texts, indices_by_text = self._token_texts_and_indices(text, offsets)
        apply_fn = self._make_apply_fn(offsets, token_texts, indices_by_text, runtime_stats)

        outputs: List[Tuple[BucketDict, AnonymizationResult]] = []
        
        ledger = TokenLedger(text, offsets)
        
        prior_edits = record.metadata.get("prior_token_edits", []) if record.metadata else []
        if prior_edits:
            ledger.apply_prior_edits(prior_edits)

        for step in self._unit.anonymize(
            text,
            offsets,
            apply_fn,
            ledger=ledger,
            prior_edits=prior_edits,
        ):
            k_value = step.threshold
            private_text = step.text
            ledger = step.ledger
            
            result_edits = ledger.result_edits_metadata()
            result_spans: List[TextAnnotation] = []
            for edit in result_edits:
                if edit.get("kind") != "replaced":
                    continue
                span = edit.get("span")
                if not span:
                    continue
                result_spans.append(
                    TextAnnotation(
                        start=span[0],
                        end=span[1],
                        label="petre",
                        text=str(edit.get("text", "")),
                        replacement=str(edit.get("replacement", "")),
                    )
                )
            
            metadata: Dict[str, Any] = {
                "k": k_value,
                "perturbed_tokens": runtime_stats["masked"],
                "method": "petre",
                "uid": record_uid or record_name,
                "rank": step.metadata.get("rank"),
                **step.metadata,
            }
            token_edits = [TokenEdit.from_mapping(e) for e in result_edits]

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
