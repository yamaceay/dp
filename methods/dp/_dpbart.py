from typing import Dict, List
import torch
import numpy as np
from transformers.modeling_outputs import BaseModelOutput

from dp.methods.anonymizer import AnonymizationResult
from dp.methods.dp import DPAnonymizer


class DPBartAnonymizer(DPAnonymizer):
    def __init__(
        self,
        *args,
        sigma: float = 0.2,
        num_sigmas: float = 0.5,
        delta: float = 1e-5,
        max_length: int = 512,
        model_name: str = "facebook/bart-base",
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        
        self.sigma = sigma
        self.num_sigmas = num_sigmas
        self.delta = delta
        self.max_length = max_length
        self.model_name = model_name
        
        # Initialize models
        try:
            from transformers import BartTokenizerFast, BartModel, BartForConditionalGeneration
            
            self.tokenizer = BartTokenizerFast.from_pretrained(self.model_name)
            self.model = BartModel.from_pretrained(self.model_name).to(self.device)
            self.decoder = BartForConditionalGeneration.from_pretrained(self.model_name).to(self.device)
        except ImportError as exc:
            raise ImportError("transformers package is required for DPBartAnonymizer. Install with: pip install transformers") from exc
        
        self.c_min = -self.sigma
        self.c_max = self.num_sigmas * self.sigma

    def _clip(self, vector: torch.Tensor) -> torch.Tensor:
        """Clip vector values to [c_min, c_max]."""
        return torch.clip(vector, self.c_min, self.c_max)

    def _add_noise(self, vector: torch.Tensor, epsilon: float) -> torch.Tensor:
        """Add Gaussian noise to vector for differential privacy."""
        k = vector.shape[-1]
        sensitivity = 2 * self.sigma * self.num_sigmas * np.sqrt(k)
        scale = np.sqrt((sensitivity**2 / epsilon**2) * 2 * np.log(1.25 / self.delta))
        noise = torch.from_numpy(np.random.normal(0, scale, size=vector.shape)).float()
        return vector + noise.to(vector.device)

    def _grid_anonymize(self, text: str, epsilon: List[float], *args, **kwargs) -> Dict[float, List[AnonymizationResult]]:
        if not epsilon:
            raise ValueError("epsilon must contain at least one value")
        ordered_eps = [float(e) for e in dict.fromkeys(epsilon)]
        if not text or not text.strip():
            return {
                eps: [
                    AnonymizationResult(
                        text="",
                        metadata={"epsilon": eps, "delta": self.delta, "method": "dpbart"},
                    )
                ]
                for eps in ordered_eps
            }
        
        inputs = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)
        
        if inputs["input_ids"].shape[-1] == 0:
            return {
                eps: [
                    AnonymizationResult(
                        text="",
                        metadata={"epsilon": eps, "delta": self.delta, "method": "dpbart"},
                    )
                ]
                for eps in ordered_eps
            }
        
        num_tokens = len(inputs["input_ids"][0])
        
        with torch.no_grad():
            enc_output = self.model.encoder(**inputs)
            clipped = self._clip(enc_output["last_hidden_state"].cpu())
            
            results: Dict[float, List[AnonymizationResult]] = {eps: [] for eps in ordered_eps}
            for eps in ordered_eps:
                noisy = self._add_noise(clipped.clone(), eps).to(self.device)
                
                encoder_outputs_obj = BaseModelOutput(last_hidden_state=noisy)
                
                dec_out = self.decoder.generate(
                    encoder_outputs=encoder_outputs_obj,
                    max_new_tokens=num_tokens
                )
                
                private_text = self.tokenizer.decode(dec_out[0], skip_special_tokens=True).strip()
                
                metadata = {
                    "epsilon": eps,
                    "delta": self.delta,
                    "method": "dpbart",
                    "model": self.model_name,
                }
                
                results[eps].append(AnonymizationResult(text=private_text, metadata=metadata))
        
        return results
