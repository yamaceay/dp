from typing import Union, List, Optional
import torch

from dp.methods.anonymizer import AnonymizationResult
from dp.methods.dp import DPAnonymizer


class DPPromptAnonymizer(DPAnonymizer):
    def __init__(
        self,
        *args,
        model_checkpoint: str = "google/flan-t5-base",
        min_logit: float = -19.22705113016047,
        max_logit: float = 7.48324937989716,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        
        self.model_checkpoint = model_checkpoint
        self.min_logit = min_logit
        self.max_logit = max_logit
        self.sensitivity = abs(max_logit - min_logit)

        try:
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, LogitsProcessorList
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_checkpoint)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_checkpoint).to(self.device)
            
            self.logits_processor = LogitsProcessorList([
                self._create_clip_processor(self.min_logit, self.max_logit)
            ])
        except ImportError as exc:
            raise ImportError("transformers package is required for DPPromptAnonymizer. Install with: pip install transformers") from exc

    def _create_clip_processor(self, min_val: float, max_val: float):
        from transformers import LogitsProcessor
        
        class ClipLogitsProcessor(LogitsProcessor):
            def __init__(self, min_logit: float, max_logit: float):
                self.min = min_logit
                self.max = max_logit

            def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
                return torch.clamp(scores, min=self.min, max=self.max)
        
        return ClipLogitsProcessor(min_val, max_val)

    def _create_prompt(self, text: str) -> str:
        return f"Document : {text}\nParaphrase of the document :"

    def _encode_without_special(self, text: str) -> List[int]:
        encoding = self.tokenizer(text, add_special_tokens=False)
        input_ids = encoding.get("input_ids", [])
        if isinstance(input_ids, list) and input_ids and isinstance(input_ids[0], list):
            return input_ids[0]
        return input_ids or []

    def anonymize(self, text: str, *args, epsilon: Union[float, List[float]] = 100.0, **kwargs) -> AnonymizationResult:
        if isinstance(epsilon, list):
            epsilon = epsilon[0] if epsilon else 100.0
        
        epsilon = float(epsilon)
        
        if not text or not text.strip():
            return AnonymizationResult(
                text="",
                metadata={"epsilon": epsilon, "method": "dpprompt"}
            )
        
        temperature = 2 * self.sensitivity / epsilon
        prompt = self._create_prompt(text)
        prompt_ids = self._encode_without_special(prompt)
        max_new_tokens = len(prompt_ids)
        
        model_inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
        ).to(self.device)
        
        with torch.no_grad():
            output = self.model.generate(
                **model_inputs,
                do_sample=True,
                top_k=0,
                top_p=1.0,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                logits_processor=self.logits_processor,
            )
        
        private_text = self.tokenizer.decode(output[0], skip_special_tokens=True).strip()
        
        metadata = {
            "epsilon": epsilon,
            "method": "dpprompt",
            "model": self.model_checkpoint,
            "temperature": temperature,
        }
        
        return AnonymizationResult(text=private_text, metadata=metadata)

