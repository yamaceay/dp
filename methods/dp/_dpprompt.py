from typing import Dict, List, Optional
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

    def _grid_anonymize(self, text: str, epsilon: List[float], *args, **kwargs) -> Dict[float, List[AnonymizationResult]]:
        if not epsilon:
            raise ValueError("epsilon must contain at least one value")
        ordered_eps = [float(e) for e in dict.fromkeys(epsilon)]
        if not text or not text.strip():
            return {
                eps: [
                    AnonymizationResult(
                        text="",
                        metadata={"epsilon": eps, "method": "dpprompt"},
                    )
                ]
                for eps in ordered_eps
            }
        
        prompt = self._create_prompt(text)
        prompt_ids = self._encode_without_special(prompt)
        max_new_tokens = len(prompt_ids)
        
        model_inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
        ).to(self.device)
        
        results: Dict[float, List[AnonymizationResult]] = {eps: [] for eps in ordered_eps}
        with torch.no_grad():
            for eps in ordered_eps:
                temperature = 2 * self.sensitivity / eps
                
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
                    "epsilon": eps,
                    "method": "dpprompt",
                    "model": self.model_checkpoint,
                    "temperature": temperature,
                }
                
                results[eps].append(AnonymizationResult(text=private_text, metadata=metadata))

        return results
