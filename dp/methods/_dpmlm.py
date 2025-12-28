from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple, Union
from pathlib import Path
import json
import torch
import numpy as np
import string
from collections import Counter

from dp.loaders import DatasetRecord
from dp.methods.anonymizer import AnonymizationResult, Anonymizer
from dp.methods.constants import Buckets, EpsilonParam, BucketDict, buckets_to_dicts
from dp.utils.splitter import TextSplitter
from dp.utils.memory import clear_memory
from dp.utils.token_ledger import TokenLedger
from dp.loaders.base import TextAnnotation, TextAnnotations, TokenEdit
from dp.utils.explainer.base import TokenExplainer, load_tri_label_mapping
from dp.utils.selector.base import AnonymizerUnit, ApplyFn, AnonymizationStep


class DPMlmAnonymizer(Anonymizer):   
    MODEL_NAME = "dpmlm"
    def __init__(
        self,
        *args,
        model_checkpoint: str = "roberta-base",
        clip_min: float = -3.2093127,
        clip_max: float = 16.304797887802124,
        k_candidates: int = 5,
        use_temperature: bool = True,
        compensate_epsilon: bool = False,
        add_probability: float = 0.0,
        delete_probability: float = 0.0,
        risk_temperature: Optional[float] = None,
        **kwargs
    ):
        super().__init__(*args, model=self.MODEL_NAME, **kwargs)

        self.model_checkpoint = model_checkpoint
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.sensitivity = abs(clip_max - clip_min)
        self.k_candidates = k_candidates
        self.use_temperature = use_temperature
        self.compensate_epsilon = compensate_epsilon
        self.add_probability = add_probability
        self.delete_probability = delete_probability
        self.risk_temperature = risk_temperature

        self._unit: Optional[AnonymizerUnit] = None
        self._explainer = None
        self.splitter = TextSplitter()
        self._risk_scores_by_uid: Dict[str, Dict[Tuple[int, int], float]] = {}
        self.dataset_records: List[DatasetRecord] = []

        self._tri_label_mapping: Optional[Dict[str, int]] = None
        self._tri_label_mapping_source: Optional[str] = None

        self._pre_stream_starting_by_uid: Dict[
            str,
            Tuple[
                List[Tuple[int, int]],
                List[int],
                Dict[int, str],
                Dict[int, str],
            ],
        ] = {}

        self._pre_stream_direct_ledger_by_uid: Dict[str, TokenLedger] = {}

        try:
            from transformers import AutoTokenizer, AutoModelForMaskedLM
            from nltk.tokenize.treebank import TreebankWordDetokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_checkpoint)
            self.model = AutoModelForMaskedLM.from_pretrained(self.model_checkpoint).to(self.device)
            self.detokenizer = TreebankWordDetokenizer()

        except ImportError as exc:
            raise ImportError("Required packages not found. Install with: uv pip install transformers nltk") from exc

    def set_unit(self, unit: AnonymizerUnit) -> None:
        self._unit = unit

    def set_risk_scores(
        self,
        risk_scores: Dict[str, Dict[str, object]],
        records: Optional[Sequence[DatasetRecord]] = None,
    ) -> None:
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
            span_map: Dict[Tuple[int, int], float] = {}
            for span, value in zip(offsets, scores):
                if not isinstance(span, (list, tuple)) or len(span) < 2:
                    continue
                try:
                    span_key = (int(span[0]), int(span[1]))
                    span_map[span_key] = float(value)
                except (TypeError, ValueError):
                    continue
            if span_map:
                self._risk_scores_by_uid[uid] = span_map

    def add_dataset_records(self, dataset_records: Sequence[DatasetRecord]) -> None:
        self.dataset_records = list(dataset_records)

    def pre_stream_anonymize(
        self,
        *args,
        risk_scores: Optional[Dict[str, Dict[str, object]]] = None,
        **kwargs,
    ) -> None:
        if risk_scores is not None:
            if not self.dataset_records:
                raise ValueError(
                    "DPMlmAnonymizer received precomputed risk_scores but has no dataset records; "
                    "run in dataset mode (use --indices) so risk scores can be matched to records"
                )
            self.set_risk_scores(risk_scores, records=self.dataset_records or None)

    def anonymize_from_dataset(
        self,
        idx: int,
        *args,
        buckets: Buckets = [],
        **kwargs,
    ) -> List[Tuple[BucketDict, AnonymizationResult]]:
        if idx < 0 or idx >= len(self.dataset_records):
            raise IndexError(f"Index {idx} is out of bounds")
        record = self.dataset_records[idx]
        
        masked_spans = record.spans or []
        
        return self.anonymize_any_text(
            record.text,
            *args,
            buckets=buckets,
            record_name=record.name,
            record_uid=str(record.uid),
            masked_spans=masked_spans,
            **kwargs,
        )

    def _target_label_id_for_record(self, record_name: Optional[str]) -> int:
        if not record_name:
            raise ValueError("record_name is required for TRI rank evaluation")
        mapping, source = load_tri_label_mapping(self._explainer, self._tri_label_mapping, self._tri_label_mapping_source)
        self._tri_label_mapping = mapping
        self._tri_label_mapping_source = source
        if record_name not in mapping:
            raise ValueError(f"record_name {record_name!r} not present in TRI label mapping")
        return int(mapping[record_name])

    def _make_rank_evaluator(self) -> Callable[[str, int], int]:
        if self._explainer is None:
            raise ValueError("DPMlmAnonymizer requires explainer for rank evaluation")
        if hasattr(self._explainer, "_load_pipeline"):
            self._explainer._load_pipeline()
        pipe = getattr(self._explainer, "pipeline", None)
        if pipe is None:
            raise ValueError("Explainer pipeline is not available for rank evaluation")

        def rank_evaluator(current_text: str, target_label: int) -> int:
            target = f"LABEL_{int(target_label)}"
            entries = pipe([current_text], batch_size=1)[0]
            if not isinstance(entries, list) or not entries:
                raise ValueError("TRI pipeline returned no predictions")
            scored = [e for e in entries if isinstance(e, dict) and "label" in e and "score" in e]
            if not scored:
                raise ValueError("TRI pipeline returned invalid predictions")
            scored.sort(key=lambda e: float(e["score"]), reverse=True)
            for i, e in enumerate(scored, start=1):
                if str(e.get("label")) == target:
                    return i
            raise ValueError(f"Target label {target!r} not found in TRI predictions")

        return rank_evaluator

    def _tokenize(self, text: str) -> Tuple[List[str], List[Tuple[int, int]]]:
        tokens = []
        offsets = []
        for start, end, token in self.splitter.tokenize_with_spans(text):
            tokens.append(token)
            offsets.append((start, end))
        return tokens, offsets

    def _privatize_token(
        self,
        sentence: str,
        token: str,
        offset: Tuple[int, int],
        epsilon: float
    ) -> str:
        masked_sentence = self._replace_token(sentence, self.tokenizer.mask_token, offset)
        if masked_sentence == sentence:
            raise ValueError(f"Token {token} not found in sentence during masking: {sentence}")

        input_ids = self.tokenizer.encode(masked_sentence, add_special_tokens=True, truncation=True, max_length=512)
        
        try:
            mask_pos = input_ids.index(self.tokenizer.mask_token_id)
        except ValueError:
            return token
        
        model_input = torch.tensor(input_ids).reshape(1, -1).to(self.device)
        
        with torch.no_grad():
            output = self.model(model_input)
        
        logits = output[0].squeeze().detach().cpu().numpy()
        mask_logits = logits[mask_pos]
        
        if self.use_temperature:
            temperature = 2 * self.sensitivity / epsilon
            mask_logits = np.clip(mask_logits, self.clip_min, self.clip_max)
            mask_logits = mask_logits / temperature
            
            scores = torch.softmax(torch.from_numpy(mask_logits), dim=0)
            scores = scores / scores.sum()
            
            chosen_idx = np.random.choice(len(mask_logits), p=scores.numpy())
            return self.tokenizer.decode(chosen_idx).strip()
        else:
            top_tokens = torch.topk(torch.from_numpy(mask_logits), k=self.k_candidates, dim=0)[1]
            return self.tokenizer.decode(top_tokens[0].item()).strip()

    def _generate_additional_token(
        self,
        text: str,
        epsilon: float
    ) -> str:
        masked_text = text + " " + self.tokenizer.mask_token
        
        input_ids = self.tokenizer.encode(masked_text, add_special_tokens=True, truncation=True, max_length=512)
        
        try:
            mask_pos = input_ids.index(self.tokenizer.mask_token_id)
        except ValueError:
            return ""
        
        model_input = torch.tensor(input_ids).reshape(1, -1).to(self.device)
        
        with torch.no_grad():
            output = self.model(model_input)
        
        logits = output[0].squeeze().detach().cpu().numpy()
        mask_logits = logits[mask_pos]
        
        if self.use_temperature:
            temperature = 2 * self.sensitivity / epsilon
            mask_logits = np.clip(mask_logits, self.clip_min, self.clip_max)
            mask_logits = mask_logits / temperature
            
            scores = torch.softmax(torch.from_numpy(mask_logits), dim=0)
            scores = scores / scores.sum()
            
            chosen_idx = np.random.choice(len(mask_logits), p=scores.numpy())
            return self.tokenizer.decode(chosen_idx).strip()
        else:
            top_tokens = torch.topk(torch.from_numpy(mask_logits), k=self.k_candidates, dim=0)[1]
            return self.tokenizer.decode(top_tokens[0].item()).strip()

    def _replace_token(self, text: str, replacement: str, offset: Tuple[int, int]) -> str:
        start, end = offset
        return text[:start] + replacement + text[end:]

    def _weights_to_probs(self, weights: np.ndarray, temperature: float) -> np.ndarray:
        if temperature is None:
            temperature = 1.0
        scores = np.asarray(weights, dtype=float)
        if scores.size == 0:
            return scores
        positive_scores = np.exp(scores / temperature)
        probs = positive_scores / positive_scores.sum()
        return probs

    def _lookup_precomputed_scores(
        self,
        text: str,
        offsets: Sequence[Tuple[int, int]],
        record_name: Optional[str],
        record_uid: Optional[str] = None,
        indices: Optional[Sequence[int]] = None,
    ) -> Optional[np.ndarray]:
        uid = self._resolve_risk_uid(text, record_name, record_uid)
        if uid is None:
            return None
        mapping = self._risk_scores_by_uid.get(uid)
        if mapping is None:
            return None
        values: List[float] = []
        for idx in range(len(offsets)):
            if idx < 0 or idx >= len(offsets):
                return None
            if indices is not None and idx not in indices:
                continue
            span = offsets[idx]
            key = (int(span[0]), int(span[1]))
            if key not in mapping:
                return None
            values.append(float(mapping[key]))
        if not values:
            return None
        return np.asarray(values, dtype=float)

    def _resolve_risk_uid(self, text: str, record_name: Optional[str], record_uid: Optional[str] = None) -> Optional[str]:
        if record_uid and record_uid in self._risk_scores_by_uid:
            return record_uid
        if record_name and record_name in self._risk_scores_by_uid:
            return record_name
        return None

    def _make_apply_fn(
        self,
        text: str,
        offsets: List[Tuple[int, int]],
        epsilon: float,
        runtime_stats: Dict[str, int],
        masked_spans: Sequence[TextAnnotation],
    ) -> ApplyFn:
        def apply_fn(idx: int, ledger: TokenLedger) -> None:
            if idx >= len(offsets):
                return
            
            entry = ledger.entry(idx)
            token = entry.original_text
            token_start, token_end = entry.start, entry.end

            if self._is_token_masked(token_start, token_end, masked_spans):
                ledger.replace(idx, token)
                runtime_stats["total"] += 1
                return

            if token in string.punctuation:
                ledger.replace(idx, token)
                runtime_stats["total"] += 1
                return

            is_last_token = idx == len(offsets) - 1
            if self.delete_probability > 0 and not is_last_token:
                if np.random.rand() < self.delete_probability:
                    ledger.delete(idx)
                    runtime_stats["deleted"] += 1
                    return

            private_token = self._privatize_token(text, token, (token_start, token_end), epsilon)

            original_text_span = text[token_start:token_end]
            if len(private_token) == len(original_text_span):
                private_token = "".join(
                    p.upper() if o.isupper() else p.lower()
                    for p, o in zip(private_token, original_text_span)
                )
            elif original_text_span and original_text_span[0].isupper():
                private_token = private_token.capitalize()

            ledger.replace(idx, private_token)
            if private_token != token:
                runtime_stats["perturbed"] += 1
            runtime_stats["total"] += 1

            if self.add_probability > 0:
                while np.random.rand() < self.add_probability:
                    context_text = ledger.render_offsets(text)
                    additional_token = self._generate_additional_token(context_text, epsilon)
                    if not additional_token:
                        break
                    ledger.add_after(idx, additional_token)
                    runtime_stats["added"] += 1

        return apply_fn

    def _collect_risk_scores(
        self,
        text: str,
        offsets: Sequence[Tuple[int, int]],
        record_name: Optional[str],
        record_uid: Optional[str] = None,
        critical_indices: Optional[Sequence[int]] = None,
    ) -> Tuple[np.ndarray, bool]:
        precomputed_scores = self._lookup_precomputed_scores(text, offsets, record_name, record_uid=record_uid, indices=critical_indices)
        if precomputed_scores is not None:
            return precomputed_scores, True

        critical_offsets = offsets if critical_indices is None else [offsets[i] for i in critical_indices]
        from dp.utils.explainer.uniform import UniformExplainer
        if self._explainer is not None and isinstance(self._explainer, UniformExplainer):
            scores = self._explainer.explain(text, critical_offsets)
            return scores, False

        raise NotImplementedError("DPMlmAnonymizer requires precomputed risk scores for each record.")

    def _is_token_masked(
        self,
        token_start: int,
        token_end: int,
        masked_spans: Sequence[TextAnnotation],
    ) -> bool:
        for span in masked_spans:
            if token_start >= span.start and token_end <= span.end:
                return True
        return False

    def anonymize_any_text(
        self,
        text: str,
        *args,
        buckets: Buckets = [],
        record_name: Optional[str] = None,
        record_uid: Optional[str] = None,
        masked_spans: Optional[Sequence[TextAnnotation]] = None,
        **kwargs,
    ) -> List[Tuple[BucketDict, AnonymizationResult]]:
        if not text or not text.strip():
            return []
        
        if masked_spans is None:
            masked_spans = []

        combos = buckets_to_dicts(buckets)
        outputs: List[Tuple[BucketDict, AnonymizationResult]] = []

        for hp in combos:
            eps_val = hp.get("epsilon")
            if eps_val is None:
                raise ValueError("DPMlmAnonymizer requires epsilon via Buckets (EpsilonParam)")

            try:
                tokens, offsets = self._tokenize(text)

                if self._unit is None:
                    from dp.utils.selector.all_selector import AllUnit
                    self._unit = AllUnit()

                from dp.utils.selector.by_risk_selector import ByRiskUnit
                from dp.utils.selector.pii_only_selector import PIIOnlyUnit
                from dp.utils.selector.until_k_selector import UntilKUnit
                if isinstance(self._unit, UntilKUnit):
                    k_val = hp.get("k")
                    if k_val is None:
                        raise ValueError("DPMlmAnonymizer using until_k selector requires KParams buckets")
                    self._unit.set_thresholds([int(k_val)], name="k")
                elif isinstance(self._unit, ByRiskUnit):
                    rho_val = hp.get("rho")
                    if rho_val is None:
                        raise ValueError("DPMlmAnonymizer using by_risk selector requires RhoParams buckets")
                    self._unit.set_thresholds([float(rho_val)], name="rho")
                elif isinstance(self._unit, PIIOnlyUnit):
                    lambda_val = hp.get("lambda")
                    if lambda_val is None:
                        raise ValueError("DPMlmAnonymizer using pii_only selector requires LambdaParams buckets")
                    self._unit.set_thresholds([float(lambda_val)], name="lambda")

                if isinstance(self._unit, UntilKUnit):
                    target_label_id = self._target_label_id_for_record(record_name)
                    self._unit.set_target_label(target_label_id)
                    self._unit.set_rank_evaluator(self._make_rank_evaluator())
                context: Dict[str, Any] = {"record_name": record_name}
                used_precomputed = False

                risk_scores, used_precomputed = self._collect_risk_scores(text, offsets, record_name, record_uid=record_uid)
                if risk_scores.size and len(risk_scores) == len(offsets):
                    self._unit.set_risk_scores(risk_scores)

                perturbation_ratio = 1.0
                compensated_epsilon = float(eps_val) * perturbation_ratio

                runtime_stats: Dict[str, int] = {
                    "perturbed": 0,
                    "total": 0,
                    "added": 0,
                    "deleted": 0,
                }
                
                ledger = TokenLedger(text, offsets)

                apply_fn = self._make_apply_fn(
                    text,
                    offsets,
                    compensated_epsilon,
                    runtime_stats,
                    masked_spans,
                )

                last_step: Optional[AnonymizationStep] = None
                for step in self._unit.anonymize(text, offsets, apply_fn, ledger, **context):
                    hp_with_threshold = {**hp}
                    threshold = step.threshold
                    if threshold is not None:
                        threshold_type = step.threshold_type
                        if threshold_type not in {"lambda", "rho", "k"}:
                            raise ValueError(f"Unknown threshold name: {threshold_type!r}")
                        hp_with_threshold[threshold_type] = threshold

                    private_text = step.text
                    ledger = step.ledger

                    metadata: Dict[str, Any] = {
                        "epsilon": eps_val,
                        "method": "dpmlm",
                        "model": self.model_checkpoint,
                        "threshold": threshold,
                        **runtime_stats,
                        **step.metadata,
                    }
                    token_edits = [TokenEdit.from_mapping(e) for e in ledger.result_edits_metadata()]
                    if self.compensate_epsilon:
                        metadata["effective_epsilon"] = compensated_epsilon
                    if used_precomputed:
                        metadata["explainer"] = "PrecomputedRisk"
                    elif self._explainer is not None:
                        metadata["explainer"] = self._explainer.__class__.__name__

                    outputs.append((
                        hp_with_threshold,
                        AnonymizationResult(
                            text=private_text,
                            annotations=TextAnnotations(
                                spans=[],
                                token_edits=token_edits,
                            ),
                            metadata=metadata,
                        ),
                    ))
                    last_step = step

                if last_step is None:
                    metadata = {
                        "epsilon": eps_val,
                        "method": "dpmlm",
                        "model": self.model_checkpoint,
                        "perturbed": 0,
                        "total": 0,
                    }
                    outputs.append((hp, AnonymizationResult(text=text, annotations=TextAnnotations(), metadata=metadata)))

            finally:
                clear_memory()

        return outputs
