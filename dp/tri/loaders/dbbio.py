from __future__ import annotations

from typing import List, Optional, Tuple, Union

from dp.loaders.base import DatasetRecord
from dp.loaders.dbbio import DBBioDatasetAdapter
from dp.tri.loaders.base import AttackerDatasetAdapter
from dp.utils.rewriter import BartRewriter
from dp.utils.device import resolve_device

class DBBioAttackerDatasetAdapter(AttackerDatasetAdapter):
    def __init__(
        self,
        n_samples: int = 3,
        rewriter_model_name: str = "facebook/bart-large-cnn",
        rewriter_device: Optional[Union[str, int]] = None,
        rewriter_max_length: int = 256,
        rewriter_min_length: int = 80,
        max_background_tokens: int = 512,
        **data_kwargs,
    ) -> None:
        adapter = DBBioDatasetAdapter(**data_kwargs)
        super().__init__(
            adapter=adapter,
            use_records_list=True,
            max_background_tokens=max_background_tokens,
            rewriter_max_length=rewriter_max_length,
            rewriter_min_length=rewriter_min_length,
        )
        if rewriter_device is None:
            rewriter_device = resolve_device()
        rewriter = BartRewriter(
            model_name=rewriter_model_name,
            device=rewriter_device,
            max_input_tokens=max_background_tokens,
        )
        self.set_rewriter(rewriter)
        self.n_samples = n_samples

    def extract_background_knowledge(self, record: DatasetRecord) -> List[Tuple[str, str]]:
        background_knowledge: List[Tuple[str, str]] = []
        
        kwargs = {"max_length": 256, "min_length": 128, "do_sample": True, "top_k": 50, "top_p": 0.95, "temperature": 1.2}
        for _ in range(self.n_samples):
            content = self.rewriter.rewrite(text=record.text, **kwargs)
            background_knowledge.append((record.uid, content))

        return background_knowledge


__all__ = ["DBBioAttackerDatasetAdapter"]
