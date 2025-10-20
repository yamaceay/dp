from typing import List
import nltk
import torch
import numpy as np
import string
from collections import Counter

from dp.methods.anonymizer import AnonymizationResult
from dp.methods.dp import DPAnonymizer


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

        self.pii_detector = None
        self.explainer = None

        try:
            from transformers import AutoTokenizer, AutoModelForMaskedLM
            from nltk.tokenize.treebank import TreebankWordDetokenizer

            nltk.download('punkt', quiet=True)

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_checkpoint)
            self.model = AutoModelForMaskedLM.from_pretrained(self.model_checkpoint).to(self.device)
            self.detokenizer = TreebankWordDetokenizer()

        except ImportError as exc:
            raise ImportError("Required packages not found. Install with: pip install transformers nltk") from exc

    def set_filtering_strategy(self, detector):
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

    def set_scoring_strategy(self, explainer):
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

    def _tokenize(self, text: str) -> List[str]:
        return nltk.word_tokenize(text)

    def _get_token_offsets(self, text: str, tokens: List[str]) -> List[tuple]:
        offsets = []
        pos = 0
        for token in tokens:
            start = text.find(token, pos)
            if start == -1:
                offsets.append((pos, pos))
            else:
                offsets.append((start, start + len(token)))
                pos = start + len(token)
        return offsets

    def _sentence_enum(self, tokens: List[str]) -> List[int]:
        counts = Counter()
        occurrences = []
        for token in tokens:
            counts[token] += 1
            occurrences.append(counts[token])
        return occurrences

    def _nth_replace(self, text: str, target: str, replacement: str, occurrence: int) -> str:
        parts = text.split()
        count = 0
        for i, part in enumerate(parts):
            if part == target:
                count += 1
                if count == occurrence:
                    parts[i] = replacement
                    break
        return " ".join(parts)

    def _privatize_token(
        self,
        sentence: str,
        token: str,
        occurrence: int,
        epsilon: float
    ) -> str:
        masked_sentence = self._nth_replace(sentence, token, self.tokenizer.mask_token, occurrence)
        
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

    def grid_anonymize(self, text: str, *args, epsilon: List[float] = None, **kwargs) -> List[AnonymizationResult]:
        if epsilon is None:
            epsilon = [100.0]
        
        if not isinstance(epsilon, list):
            epsilon = [float(epsilon)]
        
        epsilon = [float(e) for e in epsilon]

        if not text or not text.strip():
            return [AnonymizationResult(
                text="",
                metadata={"epsilon": e, "method": "dpmlm"}
            ) for e in epsilon]

        tokens = self._tokenize(text)
        offsets = self._get_token_offsets(text, tokens)
        occurrences = self._sentence_enum(tokens)

        pii_spans = []
        if self.pii_detector is not None:
            pii_spans = self.pii_detector.select(text)

        critical_indices = []
        critical_tokens = []
        for i, (token, (token_start, token_end)) in enumerate(zip(tokens, offsets)):
            if token in string.punctuation:
                continue
            
            is_pii = False
            if pii_spans:
                is_pii = any(
                    not (token_end <= span.start or token_start >= span.end)
                    for span in pii_spans
                )
            
            if pii_spans and not is_pii:
                continue
            
            critical_indices.append(i)
            critical_tokens.append(token)

        perturbation_ratio = 1.0
        if self.compensate_epsilon and critical_indices:
            non_punctuation_total = sum(1 for t in tokens if t not in string.punctuation)
            if non_punctuation_total > 0:
                perturbation_ratio = len(critical_indices) / non_punctuation_total
                perturbation_ratio = max(perturbation_ratio, 1e-6)

        scores = None
        weights = None
        if self.explainer is not None and critical_tokens:
            try:
                scores = self.explainer.explain(text, critical_tokens)
                if scores is not None and len(scores) == len(critical_tokens):
                    positive_scores = np.maximum(scores, 1e-6)
                    weights = positive_scores / positive_scores.sum()
            except Exception as e:
                print(f"Warning: Explainer failed ({e}), using uniform epsilon")
        
        results = []
        for eps in epsilon:
            compensated_epsilon = eps * perturbation_ratio
            epsilon_values = [compensated_epsilon] * len(critical_indices)
            
            if weights is not None:
                epsilon_values = [compensated_epsilon / (w * len(weights)) for w in weights]
                epsilon_values = np.clip(epsilon_values, 1e-6, compensated_epsilon * len(weights))
            
            critical_map = {idx: eps_val for idx, eps_val in zip(critical_indices, epsilon_values)}
            
            perturbed = 0
            total = 0
            added = 0
            deleted = 0
            replaced_tokens = []
            
            for i, (token, (token_start, token_end), occurrence) in enumerate(zip(tokens, offsets, occurrences)):
                if token in string.punctuation:
                    replaced_tokens.append(token)
                    total += 1
                    continue

                is_pii = False
                if pii_spans:
                    is_pii = any(
                        not (token_end <= span.start or token_start >= span.end)
                        for span in pii_spans
                    )
                
                if pii_spans and not is_pii:
                    replaced_tokens.append(token)
                    total += 1
                    continue

                is_last_token = (i == len(tokens) - 1)
                delete_prob = np.random.rand()
                
                if delete_prob < self.delete_probability and not is_last_token:
                    deleted += 1
                    continue

                token_epsilon = critical_map.get(i, compensated_epsilon)
                private_token = self._privatize_token(text, token, occurrence, token_epsilon)

                original_text = text[token_start:token_end]
                if len(private_token) == len(original_text):
                    private_token = ''.join(
                        p.upper() if o.isupper() else p.lower()
                        for p, o in zip(private_token, original_text)
                    )
                elif original_text and original_text[0].isupper():
                    private_token = private_token.capitalize()

                replaced_tokens.append(private_token)

                if private_token != token:
                    perturbed += 1
                total += 1
                
                add_prob = np.random.rand()
                if add_prob < self.add_probability:
                    additional_token = self._generate_additional_token(
                        self.detokenizer.detokenize(replaced_tokens),
                        token_epsilon
                    )
                    if additional_token:
                        replaced_tokens.append(additional_token)
                        added += 1

            private_text = self.detokenizer.detokenize(replaced_tokens)

            metadata = {
                "epsilon": eps,
                "method": "dpmlm",
                "model": self.model_checkpoint,
                "perturbed": perturbed,
                "total": total,
                "added": added,
                "deleted": deleted,
            }
            if self.compensate_epsilon:
                metadata["effective_epsilon"] = compensated_epsilon
            if self.pii_detector is not None:
                metadata["pii_detection"] = "enabled"
                metadata["pii_spans_count"] = len(pii_spans)
            if self.explainer is not None:
                metadata["explainer"] = self.explainer.__class__.__name__
                metadata["critical_tokens"] = len(critical_tokens)

            results.append(AnonymizationResult(text=private_text, metadata=metadata))
        
        return results
    
    def anonymize(self, text: str, *args, epsilon: float = 100.0, **kwargs) -> AnonymizationResult:
        return self.grid_anonymize(text, epsilon=[epsilon], **kwargs)[0]