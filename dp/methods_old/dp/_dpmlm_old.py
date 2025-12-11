from typing import Dict, Iterator, List, Optional, Sequence, Tuple
import inspect
import torch
import numpy as np
import string
from collections import Counter

from dp.loaders import DatasetRecord
from dp.methods.anonymizer import AnonymizationResult
from dp.methods.dp import DPAnonymizer
from dp.utils.splitter import TextSplitter
from dp.utils.memory import clear_memory
from dp.utils.token_ledger import TokenLedger
from dp.utils.explainer.base import TokenExplainer
from dp.utils.selector.base import TokenSelector


class DPMlmAnonymizer(DPAnonymizer):
    """
    Differential Privacy Masked Language Model (DPMLM) Anonymizer.
    
    This anonymizer uses a masked language model to apply differential privacy
    to text by replacing tokens with semantically similar alternatives. It supports
    plug-and-play filtering and scoring strategies via utility classes.
    
    Architecture:
        1. Tokenization: Text is split into tokens
        2. Filtering: Optional PII detection to skip sensitive tokens
        3. Scoring: Optional explainability to prioritize token importance
        4. Privatization: Tokens are replaced using masked language model with DP
    
    Usage:
        Basic usage with default settings (all tokens privatized uniformly):
        
        >>> from dp.methods.dp import DPMlmAnonymizer
        >>> anonymizer = DPMlmAnonymizer(model_checkpoint="roberta-base")
        >>> result = anonymizer.anonymize("Hello world", epsilon=1.0)
        >>> print(result.text)
        
        With PII filtering (skip PII tokens):
        
        >>> from dp.utils.selector import AllSelector, PIIOnlySelector
        >>> anonymizer = DPMlmAnonymizer()
        >>> anonymizer.set_filtering_strategy(AllSelector())  # or PIIOnlySelector()
        >>> result = anonymizer.anonymize("My name is John", epsilon=1.0)
        
        With importance-based scoring:
        
        >>> from dp.utils.explainer import UniformExplainer
        >>> anonymizer = DPMlmAnonymizer()
        >>> anonymizer.set_scoring_strategy(UniformExplainer())
        >>> result = anonymizer.anonymize("Sensitive text here", epsilon=1.0)
    
    Args:
        model_checkpoint: HuggingFace model name for masked LM (default: "roberta-base")
        clip_min: Minimum logit value for clipping (default: -3.2093127)
        clip_max: Maximum logit value for clipping (default: 16.304797887802124)
        k_candidates: Number of top candidates to consider (default: 5)
        use_temperature: Whether to use temperature scaling (default: True)
        compensate_epsilon: Whether to compensate epsilon based on perturbation ratio (default: False)
        add_probability: Probability of adding an additional token after replacement (default: 0.0)
        delete_probability: Probability of deleting a token instead of replacing (default: 0.0)
        **kwargs: Additional arguments passed to parent DPAnonymizer
    
    Attributes:
        pii_detector: Selector instance for filtering (set via set_filtering_strategy)
        explainer: Explainer instance for scoring (set via set_scoring_strategy)
    """
    
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
        super().__init__(*args, **kwargs)

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

        self.pii_detector = None
        self.explainer = None
        self.splitter = TextSplitter()
        self._risk_scores_by_uid: Dict[str, Dict[Tuple[int, int], float]] = {}
        self._risk_text_to_uid: Dict[str, List[str]] = {}
        self._risk_text_positions: Dict[str, int] = {}

        try:
            from transformers import AutoTokenizer, AutoModelForMaskedLM
            from nltk.tokenize.treebank import TreebankWordDetokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_checkpoint)
            # self.tokenizer.mask_token can be changed
            self.model = AutoModelForMaskedLM.from_pretrained(self.model_checkpoint).to(self.device)
            self.detokenizer = TreebankWordDetokenizer()

        except ImportError as exc:
            raise ImportError("Required packages not found. Install with: pip install transformers nltk") from exc

    def set_filtering_strategy(self, detector: TokenSelector):
        """
        Set filtering strategy for DPMLM anonymizer.
        
        Args:
            detector: Selector instance (must have .select(text) method that returns
                     list of TextAnnotation objects marking spans to skip)
        
        Example:
            from dp.utils.selector import AllSelector, PIIOnlySelector
            
            # Use AllSelector to privatize all tokens
            anonymizer.set_filtering_strategy(AllSelector())
            
            # Or use PIIOnlySelector to skip PII spans
            pii_selector = PIIOnlySelector(pii_model=my_model, threshold=0.5)
            anonymizer.set_filtering_strategy(pii_selector)
        """
        self.pii_detector = detector

    def set_scoring_strategy(self, explainer: TokenExplainer):
        """
        Set scoring strategy for DPMLM anonymizer.
        
        Args:
            explainer: Explainer instance (must have .explain(text) method that returns
                      importance scores for tokens)
        
        Example:
            from dp.utils.explainer import UniformExplainer, GreedyExplainer
            
            # Use UniformExplainer for equal privacy budget allocation
            anonymizer.set_scoring_strategy(UniformExplainer())
            
            # Or use GreedyExplainer for importance-based allocation
            greedy = GreedyExplainer(risk_model=my_risk_model)
            anonymizer.set_scoring_strategy(greedy)
        """
        self.explainer = explainer

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

    def _selector_supports_risks(self) -> bool:
        if self.pii_detector is None:
            return False
        try:
            signature = inspect.signature(self.pii_detector.select)
        except (TypeError, ValueError):
            return False
        if "risks" in signature.parameters:
            return True
        return any(
            param.kind == inspect.Parameter.VAR_KEYWORD
            for param in signature.parameters.values()
        )

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

        if self.explainer is not None:
            critical_offsets = offsets
            if critical_indices is not None:
                critical_offsets = [offsets[i] for i in critical_indices]

            scores = self.explainer.explain(text, critical_offsets)
            if scores is not None and len(scores) == len(critical_offsets):
                return scores, False
            
        return np.array([], dtype=float), False

    def _anonymize_single_token(self, entry, pii_spans, ledger, processed_indices, critical_map, compensated_epsilon, text, token, i, tokens, total, deleted, added):
        token_start, token_end = entry.start, entry.end
        is_pii = False
        if pii_spans:
            is_pii = any(
                not (token_end <= span.start or token_start >= span.end)
                for span in pii_spans
            )

        if pii_spans and not is_pii:
            total += 1
            return total, deleted, added

        is_last_token = i == len(tokens) - 1
        if self.delete_probability > 0 and not is_last_token:
            deletions_applied = 0
            cursor = i
            while cursor < len(tokens) - 1 and np.random.rand() < self.delete_probability:
                if cursor == len(tokens) - 1:
                    break
                ledger.delete(cursor)
                processed_indices.add(cursor)
                deleted += 1
                deletions_applied += 1
                cursor += 1
            if deletions_applied:
                return total, deleted, added

        token_epsilon = critical_map.get(i, compensated_epsilon)
        private_token = self._privatize_token(text, token, (token_start, token_end), token_epsilon)

        original_text = text[token_start:token_end]
        if len(private_token) == len(original_text):
            private_token = "".join(
                p.upper() if o.isupper() else p.lower()
                for p, o in zip(private_token, original_text)
            )
        elif original_text and original_text[0].isupper():
            private_token = private_token.capitalize()

        ledger.replace(i, private_token)
        if private_token != token:
            perturbed += 1
        total += 1

        if self.add_probability > 0:
            while np.random.rand() < self.add_probability:
                context_text = ledger.render(self.detokenizer.detokenize)
                additional_token = self._generate_additional_token(context_text, token_epsilon)
                if not additional_token:
                    break
                ledger.add_after(i, additional_token)
                added += 1
        
        return total, deleted, added

    def _grid_anonymize_stream(
        self,
        text: str,
        epsilon: List[float],
        *args,
        record_name: Optional[str] = None,
        **kwargs,
    ) -> Iterator[Tuple[float, List[AnonymizationResult]]]:
        sort_tokens_by_risk = self.sort_tokens_by_risk

        try:
            if not epsilon:
                return
            if not text or not text.strip():
                raise StopIteration

            tokens, offsets = self._tokenize(text)
            used_precomputed = False

            selector_requires_risks = self._selector_supports_risks()
            if selector_requires_risks:
                risk_scores, used_precomputed = self._collect_risk_scores(text, offsets, record_name)

            pii_spans = []
            if self.pii_detector is not None:
                detector_kwargs = {"offsets": offsets}
                if selector_requires_risks:
                    detector_kwargs["risks"] = self._weights_to_probs(
                        risk_scores,
                        temperature=self.risk_temperature or 1.0,
                    )
                pii_spans = self.pii_detector.select(text, **detector_kwargs)

            if pii_spans:
                critical_indices = []
                for i, (token, (token_start, token_end)) in enumerate(zip(tokens, offsets)):
                    if pii_spans and not any(
                        not (token_end <= span.start or token_start >= span.end)
                        for span in pii_spans
                    ):
                        continue
                    critical_indices.append(i)

                if not critical_indices:
                    for eps in epsilon:
                        yield eps, [AnonymizationResult(text=text, metadata={"epsilon": eps, "method": "dpmlm", "perturbed": 0, "total": 0})]
                    return
            else:
                critical_indices = list(range(len(tokens)))

            critical_tokens = [tokens[i] for i in critical_indices]

            perturbation_ratio = 1.0
            if self.compensate_epsilon:
                perturbation_ratio = len(critical_tokens) / len(tokens)
                perturbation_ratio = max(perturbation_ratio, 1e-6)

            probs = np.ones(len(critical_tokens)) / len(critical_tokens)
            processing_order = list(range(len(tokens)))
            if self.explainer is not None or used_precomputed:
                risk_scores, used_precomputed = self._collect_risk_scores(text, offsets, record_name, critical_indices=critical_indices)
                if len(risk_scores) == len(critical_tokens):
                    probs = self._weights_to_probs(risk_scores, temperature=self.risk_temperature)

                if sort_tokens_by_risk:
                    risk_prob_map: Dict[int, float] = {}
                    for idx_pos, idx in enumerate(critical_indices):
                        risk_prob_map[idx] = float(probs[idx_pos])

                    sorted_risk_indices = [idx for idx, _ in sorted(risk_prob_map.items(), key=lambda item: item[1], reverse=True)]
                    remaining_indices = [idx for idx in processing_order if idx not in risk_prob_map]
                    processing_order = sorted_risk_indices + remaining_indices

            for eps in epsilon:
                compensated_epsilon = eps * perturbation_ratio
                epsilon_values = [compensated_epsilon / (w * len(probs)) for w in probs]
                critical_map = {idx: eps_val for idx, eps_val in zip(critical_indices, epsilon_values)}

                ledger = TokenLedger(text, offsets)
                perturbed = 0
                total = 0
                added = 0
                deleted = 0
                processed_indices = set()

                for i in processing_order:
                    if i in processed_indices:
                        continue
                    processed_indices.add(i)
                    if i >= len(ledger):
                        continue

                    entry = ledger.entry(i)
                    token = entry.original_text
                    if token in string.punctuation:
                        ledger.replace(i, token)
                        total += 1
                        continue



                private_text = ledger.render(self.detokenizer.detokenize)

                metadata = {
                    "epsilon": eps,
                    "method": "dpmlm",
                    "model": self.model_checkpoint,
                    "perturbed": perturbed,
                    "total": total,
                    "added": added,
                    "deleted": deleted,
                    "token_edits": ledger.edits_metadata(),
                }
                if self.compensate_epsilon:
                    metadata["effective_epsilon"] = compensated_epsilon
                if self.pii_detector is not None:
                    metadata["pii_detection"] = "enabled"
                    metadata["pii_spans_count"] = len(pii_spans)
                if used_precomputed:
                    metadata["explainer"] = "PrecomputedRisk"
                    metadata["critical_tokens"] = len(critical_tokens)
                elif self.explainer is not None:
                    metadata["explainer"] = self.explainer.__class__.__name__
                    metadata["critical_tokens"] = len(critical_tokens)

                yield eps, [AnonymizationResult(text=private_text, metadata=metadata)]
        finally:
            clear_memory()
