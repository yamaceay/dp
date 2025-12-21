from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, LogitsProcessorList

from dp.methods.anonymizer import AnonymizationResult, Anonymizer
from dp.methods.constants import Buckets, BucketDict, EpsilonParam
from dp.loaders.base import TextAnnotations

class DPParaphraseAnonymizer(Anonymizer):
    MODEL_NAME = "dpparaphrase"
    def __init__(
        self,
        *args,
        model_checkpoint: str = "./models/gpt2-paraphraser",
        min_logit: float = -96.85249956065758,
        max_logit: float = -8.747697966442914,
        chunking: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        super().__init__(*args, model=self.MODEL_NAME, **kwargs)
        
        self.model_checkpoint = model_checkpoint
        self.min_logit = min_logit
        self.max_logit = max_logit
        self.sensitivity = abs(max_logit - min_logit)
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_checkpoint)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_checkpoint,
            pad_token_id=self.tokenizer.eos_token_id
        ).to(self.device)

        self._prompt_suffix = " >>>>> "
        self._chunking_enabled = bool((chunking or {}).get("enabled", False))
        self._chunk_overlap_tokens = int((chunking or {}).get("overlap_tokens", 0) or 0)
        if self._chunk_overlap_tokens < 0:
            raise ValueError("chunking.overlap_tokens must be >= 0")

        max_total = (chunking or {}).get("max_total_tokens")
        if max_total is None:
            max_total = getattr(self.model.config, "n_positions", None)
        if max_total is None:
            max_total = getattr(self.tokenizer, "model_max_length", 1024)
        self._max_total_tokens = int(max_total)
        if self._max_total_tokens <= 0:
            raise ValueError("chunking.max_total_tokens must be > 0")

        prompt_fraction = float((chunking or {}).get("prompt_fraction", 0.5))
        if not (0.0 < prompt_fraction < 1.0):
            raise ValueError("chunking.prompt_fraction must be between 0 and 1")
        self._max_prompt_tokens = max(1, int(self._max_total_tokens * prompt_fraction))
        
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

    def _split_text_for_generation(self, text: str) -> List[str]:
        from dp.utils.chunking import TokenAwareChunker

        max_prompt_tokens = self._max_prompt_tokens - len(self._encode_without_special(self._prompt_suffix))
        max_prompt_tokens = max(1, int(max_prompt_tokens))

        if not self._chunking_enabled:
            ids = self._encode_without_special(text)
            if len(ids) <= max_prompt_tokens:
                return [text]
            raise ValueError(
                f"Input is too long for {self.model_checkpoint!r} (tokens={len(ids)} > {max_prompt_tokens}). "
                "Enable chunking in configs/model/dpparaphrase.yaml under chunking.enabled=true."
            )

        if self._chunk_overlap_tokens != 0:
            raise ValueError("dpparaphrase chunking does not support overlap_tokens; set it to 0")

        chunker = TokenAwareChunker(tokenizer=self.tokenizer, max_tokens=max_prompt_tokens)
        chunks = chunker.chunk(text)
        return [c.text for c in chunks]

    def anonymize_any_text(
        self,
        text: str,
        *args,
        buckets: Buckets = [],
        **kwargs,
    ) -> List[Tuple[BucketDict, AnonymizationResult]]:
        if not text or not text.strip():
            return []

        if len(buckets) != 1 or not isinstance(buckets[0], EpsilonParam):
            raise ValueError("DPParaphraseAnonymizer expects Buckets=[EpsilonParam(...)]")

        eps_val = buckets[0].value()
        epsilon = float(eps_val)
        if epsilon <= 0:
            raise ValueError(f"epsilon must be > 0, got {eps_val!r}")

        hp = BucketDict({"epsilon": eps_val})

        chunks = self._split_text_for_generation(text)

        with torch.no_grad():
            temperature = 2 * self.sensitivity / epsilon
            outputs: List[str] = []

            for chunk_text in chunks:
                prompt = chunk_text + self._prompt_suffix
                prompt_ids = self._encode_without_special(prompt)
                prompt_len = len(prompt_ids)

                max_new = min(prompt_len, max(0, self._max_total_tokens - prompt_len))
                if max_new <= 0:
                    raise ValueError(
                        "dpparaphrase chunking configuration leaves no room for generation; "
                        "reduce chunking.prompt_fraction or increase chunking.max_total_tokens"
                    )

                generated = self.pipe(
                    prompt,
                    max_new_tokens=max_new,
                    temperature=temperature,
                )[0]["generated_text"]

                if generated.startswith(prompt):
                    private_chunk = generated[len(prompt) :]
                else:
                    private_chunk = generated.replace(prompt, "")

                private_chunk = (
                    private_chunk.replace("\xa0", " ")
                    .replace(">", "")
                    .strip()
                )
                outputs.append(private_chunk)

            private_text = " ".join([t for t in outputs if t])
            metadata = {
                "epsilon": eps_val,
                "method": "dpparaphrase",
                "model": self.model_checkpoint,
                "temperature": temperature,
                "chunking_enabled": self._chunking_enabled,
                "chunks": len(chunks),
            }
            return [(hp, AnonymizationResult(text=private_text, annotations=TextAnnotations(), metadata=metadata))]