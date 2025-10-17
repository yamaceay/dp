from typing import Union, List
import nltk
import torch
import numpy as np
import string
from collections import Counter

from dp.methods.anonymizer import AnonymizationResult
from dp.methods.dp import DPAnonymizer


class DPMlmAnonymizer(DPAnonymizer):
    def __init__(
        self,
        *args,
        model_checkpoint: str = "roberta-base",
        clip_min: float = -3.2093127,
        clip_max: float = 16.304797887802124,
        k_candidates: int = 5,
        use_temperature: bool = True,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        
        self.model_checkpoint = model_checkpoint
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.sensitivity = abs(clip_max - clip_min)
        self.k_candidates = k_candidates
        self.use_temperature = use_temperature
        
        try:
            from transformers import AutoTokenizer, AutoModelForMaskedLM
            from nltk.tokenize.treebank import TreebankWordDetokenizer

            nltk.download('punkt', quiet=True)
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_checkpoint)
            self.model = AutoModelForMaskedLM.from_pretrained(self.model_checkpoint).to(self.device)
            self.detokenizer = TreebankWordDetokenizer()
            
        except ImportError as exc:
            raise ImportError("Required packages not found. Install with: pip install transformers nltk") from exc

    def _tokenize(self, text: str) -> List[str]:
        return nltk.word_tokenize(text)

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
        
        input_ids = self.tokenizer.encode(masked_sentence, add_special_tokens=True)
        
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

    def anonymize(self, text: str, *args, epsilon: Union[float, List[float]] = 100.0, **kwargs) -> AnonymizationResult:
        if isinstance(epsilon, list):
            epsilon = epsilon[0] if epsilon else 100.0
        
        epsilon = float(epsilon)
        
        if not text or not text.strip():
            return AnonymizationResult(
                text="",
                metadata={"epsilon": epsilon, "method": "dpmlm"}
            )
        
        tokens = self._tokenize(text)
        occurrences = self._sentence_enum(tokens)
        
        perturbed = 0
        total = 0
        replaced_tokens = []
        
        for token, occurrence in zip(tokens, occurrences):
            if token in string.punctuation:
                replaced_tokens.append(token)
                total += 1
                continue
            
            private_token = self._privatize_token(text, token, occurrence, epsilon)
            
            if token and token[0].isupper():
                private_token = private_token.capitalize() if private_token else token
            elif token and token[0].islower():
                private_token = private_token.lower() if private_token else token
            
            replaced_tokens.append(private_token)
            
            if private_token != token:
                perturbed += 1
            total += 1
        
        private_text = self.detokenizer.detokenize(replaced_tokens)
        
        metadata = {
            "epsilon": epsilon,
            "method": "dpmlm",
            "model": self.model_checkpoint,
            "perturbed": perturbed,
            "total": total,
        }
        
        return AnonymizationResult(text=private_text, metadata=metadata)
