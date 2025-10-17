from typing import Union, List
import torch

from dp.methods.anonymizer import AnonymizationResult
from dp.methods.dp import DPAnonymizer


class DPParaphraseAnonymizer(DPAnonymizer):
    def __init__(
        self,
        *args,
        model_checkpoint: str = "./models/gpt2-paraphraser",
        min_logit: float = -96.85249956065758,
        max_logit: float = -8.747697966442914,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        
        self.model_checkpoint = model_checkpoint
        self.min_logit = min_logit
        self.max_logit = max_logit
        self.sensitivity = abs(max_logit - min_logit)
        
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, LogitsProcessorList
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_checkpoint)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_checkpoint,
                pad_token_id=self.tokenizer.eos_token_id
            ).to(self.device)
            
            self.logits_processor = LogitsProcessorList([
                self._create_clip_processor(self.min_logit, self.max_logit)
            ])
            
            pipeline_device = self._get_pipeline_device(self.device)
            self.pipe = pipeline(
                task="text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                logits_processor=self.logits_processor,
                device=pipeline_device,
                pad_token_id=self.tokenizer.eos_token_id
            )
        except ImportError as exc:
            raise ImportError("transformers package is required for DPParaphraseAnonymizer. Install with: pip install transformers") from exc

    def _get_pipeline_device(self, device: torch.device) -> int:
        if device.type == "cuda":
            return device.index or 0
        return -1

    def _create_clip_processor(self, min_val: float, max_val: float):
        from transformers import LogitsProcessor
        
        class ClipLogitsProcessor(LogitsProcessor):
            def __init__(self, min_logit: float, max_logit: float):
                self.min = min_logit
                self.max = max_logit

            def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
                return torch.clamp(scores, min=self.min, max=self.max)
        
        return ClipLogitsProcessor(min_val, max_val)

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
                metadata={"epsilon": epsilon, "method": "dpparaphrase"}
            )
        
        temperature = 2 * self.sensitivity / epsilon
        prompt = text + " >>>>> "
        prompt_ids = self._encode_without_special(prompt)
        length = len(prompt_ids)
        
        with torch.no_grad():
            generated = self.pipe(
                prompt,
                max_new_tokens=length,
                temperature=temperature,
            )[0]["generated_text"]
        
        private_text = (
            generated.replace(prompt, "")
            .replace(prompt.strip(), "")
            .replace("\xa0", " ")
            .replace(">", "")
            .strip()
        )
        
        metadata = {
            "epsilon": epsilon,
            "method": "dpparaphrase",
            "model": self.model_checkpoint,
            "temperature": temperature,
        }
        
        return AnonymizationResult(text=private_text, metadata=metadata)
