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
from dp.utils.risk import _scores_to_inverse_probs
from dp.utils.precomputed_risk import align_precomputed_risk_scores
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
        add_probability: float = 0.0,
        delete_probability: float = 0.0,
        risk_temperature: Optional[float] = None,
        max_retry_rounds: int = 1,
        verbose: bool = False,
        **kwargs
    ):
        super().__init__(*args, model=self.MODEL_NAME, **kwargs)

        self.model_checkpoint = model_checkpoint
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.sensitivity = abs(clip_max - clip_min)
        self.k_candidates = k_candidates
        self.use_temperature = use_temperature
        self.add_probability = add_probability
        self.delete_probability = delete_probability
        self.risk_temperature = risk_temperature
        self.max_retry_rounds = max_retry_rounds
        self.verbose = bool(verbose)

        self._unit: Optional[AnonymizerUnit] = None
        self._explainer = None
        self.splitter = TextSplitter()
        self._risk_scores_by_uid: Dict[str, Dict[Tuple[int, int], float]] = {}
        self._risk_offsets_by_uid: Dict[str, List[Tuple[int, int]]] = {}
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
        absolute_risk: bool = False,
    ) -> None:
        self._risk_scores_by_uid = {}
        self._risk_offsets_by_uid = {}
        resolved = align_precomputed_risk_scores(
            risk_scores,
            records=records,
            absolute_risk=absolute_risk,
        )
        for uid, entry in resolved.items():
            self._risk_scores_by_uid[uid] = entry.span_scores
            self._risk_offsets_by_uid[uid] = entry.ordered_offsets

    def add_dataset_records(self, dataset_records: Sequence[DatasetRecord]) -> None:
        self.dataset_records = list(dataset_records)

    def pre_stream_anonymize(
        self,
        *args,
        risk_scores: Optional[Dict[str, Dict[str, object]]] = None,
        absolute_risk: bool = False,
        **kwargs,
    ) -> None:
        if risk_scores is not None:
            if not self.dataset_records:
                raise ValueError(
                    "DPMlmAnonymizer received precomputed risk_scores but has no dataset records; "
                    "run in dataset mode (use --indices) so risk scores can be matched to records"
                )
            self.set_risk_scores(risk_scores, records=self.dataset_records or None, absolute_risk=absolute_risk)

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
        epsilon: float,
        exclude_original: bool = False,
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
        
        if exclude_original:
            original_ids = self.tokenizer.encode(token, add_special_tokens=False)
            for oid in original_ids:
                if 0 <= oid < len(mask_logits):
                    mask_logits[oid] = float("-inf")
        
        if self.use_temperature:
            temperature = 2 * self.sensitivity / epsilon
            mask_logits = np.clip(mask_logits, self.clip_min, self.clip_max)
            mask_logits = mask_logits / temperature
            
            scores = torch.softmax(torch.from_numpy(mask_logits), dim=0)

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

    def _resolve_precomputed_offsets(
        self,
        record_name: Optional[str],
        record_uid: Optional[str],
    ) -> Optional[List[Tuple[int, int]]]:
        if record_uid and record_uid in self._risk_offsets_by_uid:
            return self._risk_offsets_by_uid[record_uid]
        if record_name and record_name in self._risk_offsets_by_uid:
            return self._risk_offsets_by_uid[record_name]
        return None

    def _make_apply_fn(
        self,
        text: str,
        offsets: List[Tuple[int, int]],
        epsilon: float,
        token_epsilons: Optional[np.ndarray],
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
                runtime_stats["skipped_masked"] += 1
                return

            if token in string.punctuation:
                ledger.replace(idx, token)
                runtime_stats["total"] += 1
                runtime_stats["skipped_punctuation"] += 1
                return

            is_last_token = idx == len(offsets) - 1
            if self.delete_probability > 0 and not is_last_token:
                if np.random.rand() < self.delete_probability:
                    ledger.delete(idx)
                    runtime_stats["deleted"] += 1
                    return

            epsilon_val = epsilon
            if token_epsilons is not None and idx < len(token_epsilons):
                epsilon_val = float(token_epsilons[idx])
            private_token = self._privatize_token(text, token, (token_start, token_end), epsilon_val)

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
            else:
                runtime_stats["same_token"] += 1
            runtime_stats["total"] += 1

            if self.add_probability > 0:
                while np.random.rand() < self.add_probability:
                    context_text = ledger.render_offsets(text)
                    additional_token = self._generate_additional_token(context_text, epsilon_val)
                    if not additional_token:
                        break
                    ledger.add_after(idx, additional_token)
                    runtime_stats["added"] += 1

        return apply_fn

    def _wrap_apply_fn_with_retry(
        self,
        base_apply_fn: ApplyFn,
        ledger: TokenLedger,
        offsets: List[Tuple[int, int]],
        max_rounds: int,
    ) -> ApplyFn:
        retry_counts: Dict[int, int] = {}

        def apply_fn_with_retry(idx: int, led: TokenLedger) -> None:
            for _ in range(max_rounds):
                base_apply_fn(idx, led)
                if led.is_modified(idx):
                    return
                retry_counts[idx] = retry_counts.get(idx, 0) + 1

        return apply_fn_with_retry

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

    def _print_verbose_risk_tokens(
        self,
        text: str,
        offsets: Sequence[Tuple[int, int]],
        risk_scores: np.ndarray,
        record_uid: Optional[str],
        record_name: Optional[str],
    ) -> None:
        if not self.verbose:
            return
        if risk_scores.size == 0 or len(risk_scores) != len(offsets):
            return
        record_ref = record_uid or record_name or "<unknown>"
        ranked = sorted(
            enumerate(zip(offsets, risk_scores)),
            key=lambda item: float(item[1][1]),
            reverse=True,
        )
        print(f"[dpmlm][verbose] record={record_ref} risky tokens:")
        for token_idx, ((start, end), score) in ranked:
            token = text[int(start):int(end)]
            print(f"Token nr. {token_idx}: {token} ({float(score):.3f})")

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

        from dp.utils.selector.by_risk_selector import ByRiskUnit
        from dp.utils.selector.pii_only_selector import PIIOnlyUnit
        from dp.utils.selector.until_k_selector import UntilKUnit

        combos_by_epsilon: Dict[float, List[BucketDict]] = {}
        for hp in combos:
            eps_val = hp.get("epsilon")
            if eps_val is None:
                raise ValueError("DPMlmAnonymizer requires epsilon via Buckets (EpsilonParam)")
            combos_by_epsilon.setdefault(float(eps_val), []).append(hp)

        for eps_val, eps_combos in combos_by_epsilon.items():
            try:
                precomputed_offsets = self._resolve_precomputed_offsets(record_name, record_uid)
                if precomputed_offsets is None:
                    _, offsets = self._tokenize(text)
                else:
                    offsets = precomputed_offsets

                if self._unit is None:
                    from dp.utils.selector.all_selector import AllUnit
                    self._unit = AllUnit()

                if isinstance(self._unit, UntilKUnit):
                    k_vals = [int(hp.get("k")) for hp in eps_combos if hp.get("k") is not None]
                    if not k_vals:
                        raise ValueError("DPMlmAnonymizer using until_k selector requires KParams buckets")
                    self._unit.set_thresholds(k_vals, name="k")
                elif isinstance(self._unit, ByRiskUnit):
                    rho_vals = [float(hp.get("rho")) for hp in eps_combos if hp.get("rho") is not None]
                    if not rho_vals:
                        raise ValueError("DPMlmAnonymizer using by_risk selector requires RhoParams buckets")
                    self._unit.set_thresholds(rho_vals, name="rho")
                elif isinstance(self._unit, PIIOnlyUnit):
                    lambda_vals = [float(hp.get("lambda")) for hp in eps_combos if hp.get("lambda") is not None]
                    if not lambda_vals:
                        raise ValueError("DPMlmAnonymizer using pii_only selector requires LambdaParams buckets")
                    self._unit.set_thresholds(lambda_vals, name="lambda")

                if isinstance(self._unit, UntilKUnit):
                    target_label_id = self._target_label_id_for_record(record_name)
                    self._unit.set_target_label(target_label_id)
                    self._unit.set_rank_evaluator(self._make_rank_evaluator())
                context: Dict[str, Any] = {"record_name": record_name}
                used_precomputed = False

                risk_scores, used_precomputed = self._collect_risk_scores(text, offsets, record_name, record_uid=record_uid)
                if risk_scores.size and len(risk_scores) == len(offsets):
                    self._unit.set_risk_scores(risk_scores)
                self._print_verbose_risk_tokens(
                    text=text,
                    offsets=offsets,
                    risk_scores=risk_scores,
                    record_uid=record_uid,
                    record_name=record_name,
                )

                token_epsilons: Optional[np.ndarray] = None
                if risk_scores.size and len(risk_scores) == len(offsets):
                    from dp.utils.explainer.uniform import UniformExplainer
                    if self._explainer is not None and isinstance(self._explainer, UniformExplainer):
                        weights = np.ones_like(risk_scores, dtype=float)
                        weights = weights / float(len(weights))
                    else:
                        weights = _scores_to_inverse_probs(risk_scores, temperature=self.risk_temperature)
                    token_epsilons = eps_val * weights * len(weights)

                runtime_stats: Dict[str, int] = {
                    "perturbed": 0,
                    "total": 0,
                    "added": 0,
                    "deleted": 0,
                    "skipped_punctuation": 0,
                    "skipped_masked": 0,
                    "same_token": 0,
                }
                
                ledger = TokenLedger(text, offsets)

                apply_fn = self._make_apply_fn(
                    text,
                    offsets,
                    eps_val,
                    token_epsilons,
                    runtime_stats,
                    masked_spans,
                )
                if self.max_retry_rounds > 0:
                    apply_fn = self._wrap_apply_fn_with_retry(
                        apply_fn, ledger, offsets, self.max_retry_rounds
                    )

                for step in self._unit.anonymize(text, offsets, apply_fn, ledger, **context):
                    threshold = step.threshold
                    threshold_type = step.threshold_type
                    
                    matching_hp = None
                    for hp in eps_combos:
                        if threshold_type == "k" and hp.get("k") == threshold:
                            matching_hp = hp
                            break
                        elif threshold_type == "rho" and hp.get("rho") == threshold:
                            matching_hp = hp
                            break
                        elif threshold_type == "lambda" and hp.get("lambda") == threshold:
                            matching_hp = hp
                            break
                    
                    if matching_hp is None:
                        matching_hp = eps_combos[0] if eps_combos else {"epsilon": eps_val}
                    
                    hp_with_threshold = {**matching_hp}
                    if threshold is not None:
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

            finally:
                clear_memory()

        return outputs
