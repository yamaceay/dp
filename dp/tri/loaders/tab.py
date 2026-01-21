from __future__ import annotations

from typing import List, Optional, Tuple, Union

from dp.loaders.base import DatasetRecord
from dp.loaders.tab import TabDatasetAdapter
from dp.tri.loaders.base import AttackerDatasetAdapter
from dp.utils.chunking import TokenAwareChunker
from dp.utils.rewriter import BartRewriter
from dp.utils.device import resolve_device

class TabAttackerDatasetAdapter(AttackerDatasetAdapter):
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
        rewriter_min_length: int = 80,  # originally 40
        max_background_tokens: int = 512,
        rewrite_background: bool = True,
        n_samples: int = 3,
    ):
        adapter = TabDatasetAdapter(
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

    def _get_background_chunker(self) -> TokenAwareChunker:
        if self._background_chunker is None:
            self._background_chunker = TokenAwareChunker(
                tokenizer=self.rewriter.rewriting_pipeline.tokenizer,
                max_tokens=self.max_background_tokens,
            )
        return self._background_chunker

    def _extract_section(self, text: str, section_name: str) -> str:
        main_sections = {"PROCEDURE", "THE FACTS", "THE LAW", "AS TO THE FACTS", "COMPLAINTS"}
        
        lines = text.split('\n')
        in_section = False
        section_content = []
        
        for i, line in enumerate(lines):
            stripped = line.strip()
            
            if stripped == section_name:
                in_section = True
                continue
            
            if in_section and stripped in main_sections and stripped != section_name:
                break
            
            if in_section:
                section_content.append(line)
        
        return '\n'.join(section_content).strip()

    def extract_background_knowledge(self, record: DatasetRecord) -> List[Tuple[str, str]]:
        background = []
        chunker = self._get_background_chunker()
        
        for section_name in ["PROCEDURE", "THE FACTS", "THE LAW", "AS TO THE FACTS", "COMPLAINTS"]:
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
