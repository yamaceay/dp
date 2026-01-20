from __future__ import annotations

from typing import List, Optional, Tuple, Union

from dp.loaders.base import DatasetRecord
from dp.tri.loaders.tab import TabAttackerDatasetAdapter


class TabLongformerAttackerDatasetAdapter(TabAttackerDatasetAdapter):
    def __init__(
        self,
        data: Optional[str] = None,
        data_in: Optional[str] = None,
        start: Optional[int] = None,
        end: Optional[int] = None,
        step: Optional[int] = None,
        max_records: Optional[int] = None,
        rewriter_model_name: str = "facebook/bart-large-cnn",
        rewriter_device: Optional[Union[str, int]] = None,
        rewriter_max_length: int = 256,
        rewriter_min_length: int = 80,
        max_background_tokens: int = 4096,
        rewrite_background: bool = True,
        n_samples: int = 3,
    ) -> None:
        super().__init__(
            data=data,
            data_in=data_in,
            start=start,
            end=end,
            step=step,
            max_records=max_records,
            rewriter_model_name=rewriter_model_name,
            rewriter_device=rewriter_device,
            rewriter_max_length=rewriter_max_length,
            rewriter_min_length=rewriter_min_length,
            max_background_tokens=max_background_tokens,
            rewrite_background=rewrite_background,
            n_samples=n_samples,
        )

    def extract_background_knowledge(self, record: DatasetRecord) -> List[Tuple[str, str]]:
        background: List[Tuple[str, str]] = []
        for section_name in ["PROCEDURE", "THE FACTS", "THE LAW", "AS TO THE FACTS", "COMPLAINTS"]:
            section_text = self._extract_section(record.text, section_name)
            if not section_text:
                continue
            key = section_name.lower().replace(" ", "_")
            background.append((key, section_text))

        if self.rewrite_background:
            rewritten_background: List[Tuple[str, str]] = []
            for background_key, background_text in background:
                kwargs = {
                    "max_length": 256,
                    "min_length": 128,
                    "do_sample": True,
                    "top_k": 50,
                    "top_p": 0.95,
                    "temperature": 1.2,
                }
                for _ in range(self.n_samples):
                    rewritten_text = self.rewriter.rewrite(background_text, **kwargs)
                    rewritten_background.append((background_key, rewritten_text))
            background = rewritten_background

        return background
