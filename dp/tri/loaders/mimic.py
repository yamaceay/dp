from __future__ import annotations

from typing import List, Optional, Tuple, Union
import re

from dp.loaders.base import DatasetRecord
from dp.loaders.mimic import MIMICDatasetAdapter
from dp.tri.loaders.base import AttackerDatasetAdapter
from dp.utils.chunking import TokenAwareChunker
from dp.utils.rewriter import BartRewriter
from dp.utils.device import resolve_device

class MIMICAttackerDatasetAdapter(AttackerDatasetAdapter):
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
        rewriter_max_length: int = 256, # originally 150
        rewriter_min_length: int = 128,  # originally 40
        max_background_tokens: int = 512,
        rewrite_background: bool = True,
        n_samples: int = 3,
    ):
        adapter = MIMICDatasetAdapter(
            data=data,
            data_in=data_in,
            start=start,
            end=end,
            step=step,
            max_records=max_records,
        )
        super().__init__(
            adapter=adapter,
            max_background_tokens=max_background_tokens,
            rewriter_max_length=rewriter_max_length,
            rewriter_min_length=rewriter_min_length,
        )
        self._background_chunker: Optional[TokenAwareChunker] = None
        if rewriter_device is None:
            rewriter_device = resolve_device()
        rewriter = BartRewriter(
            model_name=rewriter_model_name,
            device=rewriter_device,
            max_input_tokens=max_background_tokens,
        )
        self.set_rewriter(rewriter)
        self.rewrite_background = rewrite_background
        self.n_samples = n_samples
        self.section_headers = ["HISTORY OF PRESENT ILLNESS", "PAST MEDICAL HISTORY", "SOCIAL HISTORY", "HOSPITAL COURSE"]

    def _get_background_chunker(self) -> TokenAwareChunker:
        if self._background_chunker is None:
            self._background_chunker = TokenAwareChunker(
                tokenizer=self.rewriter.rewriting_pipeline.tokenizer,
                max_tokens=self.max_background_tokens,
            )
        return self._background_chunker

    def _extract_section(self, text: str, section_name: str) -> str:
        section_pattern = rf"{re.escape(section_name)}:\s*(.*?)(?=\n\s*(?:[A-Z]+(?:\s+[A-Z]+)*):|$)"
        
        match = re.search(section_pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return ""

    def extract_background_knowledge(self, record: DatasetRecord) -> List[Tuple[str, str]]:
        background = []
        chunker = self._get_background_chunker()
        
        for section_name in self.section_headers:
            section_text = self._extract_section(record.text, section_name)
            if not section_text:
                continue
            
            key = section_name.lower().replace(" ", "_")
            chunks = chunker.chunk(section_text)
            
            if len(chunks) == 1:
                background.append((key, section_text))
            else:
                for chunk in chunks:
                    background.append((key, chunk.text))

        if self.rewrite_background:
            rewritten_background = []
            for background_key, background_text in background:
                kwargs = {"max_length": 256, "min_length": 128, "do_sample": True, "top_k": 50, "top_p": 0.95, "temperature": 1.2}
                for _ in range(self.n_samples):
                    rewritten_text = self.rewriter.rewrite(background_text, **kwargs)
                    rewritten_background.append((background_key, rewritten_text))
            background = rewritten_background

        return background