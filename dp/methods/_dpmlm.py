from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple
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
from dp.loaders.base import TextAnnotations, TokenEdit
from dp.utils.explainer.base import TokenExplainer
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
        sort_tokens_by_risk: bool = True,
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
        self.sort_tokens_by_risk = sort_tokens_by_risk
        self.risk_temperature = risk_temperature

        self._unit: Optional[AnonymizerUnit] = None
        self._explainer = None
        self.splitter = TextSplitter()
        self._risk_scores_by_uid: Dict[str, Dict[Tuple[int, int], float]] = {}
        self._risk_text_to_uid: Dict[str, List[str]] = {}
        self._risk_text_positions: Dict[str, int] = {}
        self.dataset_records: List[DatasetRecord] = []

        self._tri_label_mapping: Optional[Dict[str, int]] = None
        self._tri_label_mapping_source: Optional[str] = None

        try:
            from transformers import AutoTokenizer, AutoModelForMaskedLM
            from nltk.tokenize.treebank import TreebankWordDetokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_checkpoint)
            self.model = AutoModelForMaskedLM.from_pretrained(self.model_checkpoint).to(self.device)
            self.detokenizer = TreebankWordDetokenizer()

        except ImportError as exc:
            raise ImportError("Required packages not found. Install with: pip install transformers nltk") from exc

    def set_unit(self, unit: AnonymizerUnit) -> None:
        self._unit = unit

    def set_filtering_strategy(self, detector: AnonymizerUnit) -> None:
        self._unit = detector

    def set_scoring_strategy(self, explainer: TokenExplainer) -> None:
        self.set_explainer(explainer)

    def set_risk_scores(
        self,
        risk_scores: Dict[str, Dict[str, object]],
        records: Optional[Sequence[DatasetRecord]] = None,
    ) -> None:
        self._risk_scores_by_uid = {}
        if not risk_scores:
            self._risk_text_to_uid = {}
            self._risk_text_positions = {}
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
        self._risk_text_to_uid = {}
        self._risk_text_positions = {}
        if records is None:
            return
        for record in records:
            uid = record.uid
            if uid not in self._risk_scores_by_uid:
                continue
            text_key = record.text or ""
            entries = self._risk_text_to_uid.setdefault(text_key, [])
            entries.append(uid)
            self._risk_text_positions.setdefault(text_key, 0)

    def add_dataset_records(self, dataset_records: Sequence[DatasetRecord]) -> None:
        self.dataset_records = list(dataset_records)

    def pre_stream_anonymize(self, texts_or_indices, *args, **kwargs) -> None:
        risk_scores = kwargs.get("risk_scores")
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
        return self.anonymize_any_text(
            record.text,
            *args,
            buckets=buckets,
            record_name=record.name,
            record_uid=str(record.uid),
            **kwargs,
        )

    def _load_tri_label_mapping(self) -> Dict[str, int]:
        if self._explainer is None:
            raise ValueError("DPMlmAnonymizer requires explainer to load TRI label mapping")
        model_name = getattr(self._explainer, "model_name", None)
        if not model_name:
            raise ValueError("DPMlmAnonymizer requires explainer.model_name to load TRI label mapping")
        source = str(model_name)
        if self._tri_label_mapping is not None and self._tri_label_mapping_source == source:
            return self._tri_label_mapping

        mapping_path = Path(source) / "label_mapping.json"
        if not mapping_path.exists():
            raise ValueError(f"TRI label mapping not found at {mapping_path}")
        with mapping_path.open("r", encoding="utf-8") as handle:
            mapping = json.load(handle)
        if not isinstance(mapping, dict) or not mapping:
            raise ValueError(f"Invalid TRI label mapping at {mapping_path}")

        normalized: Dict[str, int] = {}
        for name, value in mapping.items():
            if not isinstance(name, str):
                continue
            try:
                normalized[name] = int(value)
            except (TypeError, ValueError):
                continue
        if not normalized:
            raise ValueError(f"TRI label mapping at {mapping_path} has no usable entries")

        self._tri_label_mapping = normalized
        self._tri_label_mapping_source = source
        return normalized

    def _target_label_id_for_record(self, record_name: Optional[str]) -> int:
        if not record_name:
            raise ValueError("record_name is required for TRI rank evaluation")
        mapping = self._load_tri_label_mapping()
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
        indices: Optional[Sequence[int]] = None,
    ) -> Optional[np.ndarray]:
        uid = self._resolve_risk_uid(text, record_name)
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

    def _resolve_risk_uid(self, text: str, record_name: Optional[str]) -> Optional[str]:
        if record_name and record_name in self._risk_scores_by_uid:
            return record_name
        text_key = text or ""
        entries = self._risk_text_to_uid.get(text_key)
        if not entries:
            return None
        position = self._risk_text_positions.get(text_key, 0)
        if position >= len(entries):
            position = len(entries) - 1
        if position < len(entries) - 1:
            self._risk_text_positions[text_key] = position + 1
        else:
            self._risk_text_positions[text_key] = position
        return entries[position]

    def _make_apply_fn(
        self,
        text: str,
        offsets: List[Tuple[int, int]],
        epsilon: float,
        runtime_stats: Dict[str, int],
    ) -> ApplyFn:
        def apply_fn(idx: int, ledger: TokenLedger) -> None:
            if idx >= len(offsets):
                return
            
            entry = ledger.entry(idx)
            token = entry.original_text
            token_start, token_end = entry.start, entry.end

            active_source = getattr(ledger, "active_edit_source", None)
            if active_source:
                if token in string.punctuation:
                    ledger.replace(idx, token)
                    runtime_stats["total"] += 1
                    return

                candidate = self._privatize_token(text, token, (token_start, token_end), epsilon)

                original_text = text[token_start:token_end]
                if len(candidate) == len(original_text):
                    candidate = "".join(
                        p.upper() if o.isupper() else p.lower()
                        for p, o in zip(candidate, original_text)
                    )
                elif original_text and original_text[0].isupper():
                    candidate = candidate.capitalize()

                ledger.replace(idx, candidate)
                runtime_stats["direct_rewritten"] = int(runtime_stats.get("direct_rewritten", 0)) + 1
                runtime_stats["perturbed"] += 1
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

            original_text = text[token_start:token_end]
            if len(private_token) == len(original_text):
                private_token = "".join(
                    p.upper() if o.isupper() else p.lower()
                    for p, o in zip(private_token, original_text)
                )
            elif original_text and original_text[0].isupper():
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
        critical_indices: Optional[Sequence[int]] = None,
    ) -> Tuple[np.ndarray, bool]:

        precomputed_scores = self._lookup_precomputed_scores(text, offsets, record_name, indices=critical_indices)
        if precomputed_scores is not None:
            return precomputed_scores, True

        if self._explainer is not None:
            if record_name is None:
                raise ValueError(
                    "record_name is required for TRI-based risk scoring; run in dataset mode (use --indices)"
                )
            critical_offsets = offsets
            if critical_indices is not None:
                critical_offsets = [offsets[i] for i in critical_indices]

            target_label_id = self._target_label_id_for_record(record_name)
            target_label = f"LABEL_{target_label_id}"
            scores = self._explainer.explain(text, critical_offsets, target_label=target_label)
            if scores is not None and len(scores) == len(critical_offsets):
                return scores, False
            
        return np.array([], dtype=float), False

    def anonymize_any_text(
        self,
        text: str,
        *args,
        buckets: Buckets = [],
        record_name: Optional[str] = None,
        record_uid: Optional[str] = None,
        **kwargs,
    ) -> List[Tuple[BucketDict, AnonymizationResult]]:
        if not text or not text.strip():
            return []

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

                from dp.utils.selector.until_k_selector import UntilKUnit
                if isinstance(self._unit, UntilKUnit):
                    target_label_id = self._target_label_id_for_record(record_name)
                    self._unit.set_target_label(target_label_id)
                    self._unit.set_rank_evaluator(self._make_rank_evaluator())
                context: Dict[str, Any] = {"record_name": record_name}
                used_precomputed = False

                risk_scores, used_precomputed = self._collect_risk_scores(text, offsets, record_name)
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
                apply_fn = self._make_apply_fn(text, offsets, compensated_epsilon, runtime_stats)

                if record_uid is not None:
                    context["starting_indices"] = self._starting_indices_for_uid(record_uid, offsets)
                    context["starting_annotations_name"] = self._starting_annotations_name
                    context["starting_edit_source"] = getattr(self, "_starting_edit_source", None)

                last_step: Optional[AnonymizationStep] = None
                for step in self._unit.anonymize(text, offsets, apply_fn, **context):
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
                        "token_edits": ledger.edits_metadata(),
                        **runtime_stats,
                        **step.metadata,
                    }
                    token_edits = [TokenEdit.from_mapping(e) for e in metadata["token_edits"]]
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
                            annotations=TextAnnotations(token_edits=token_edits),
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
                    outputs.append((hp, AnonymizationResult(text=text, metadata=metadata)))

            finally:
                clear_memory()

        return outputs
