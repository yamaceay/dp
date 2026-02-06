from __future__ import annotations

from typing import List, Optional, Tuple, Union

from dp.loaders.base import DatasetRecord
from dp.loaders._tab import TabDatasetAdapter
from dp.tri.loaders.base import AttackerDatasetAdapter, AttackerDatasetRecord, _normalize_texts
from dp.utils.chunking import TokenAwareChunker
from dp.utils.rewriter import BartRewriter
from dp.utils.device import resolve_device

class TabAttackerDatasetAdapter(AttackerDatasetAdapter):
    def __init__(
        self,
        rewriter_model_name: str = "facebook/bart-large-cnn",
        rewriter_device: Optional[Union[str, int]] = None,
        rewriter_max_length: int = 256,
        rewriter_min_length: int = 128,
        do_sample: bool = True,
        top_k: int = 50,
        top_p: float = 0.95,
        temperature: float = 1.2,
        max_background_tokens: int = 512,
        rewrite_background: bool = True,
        n_train_samples: int = 3,
        n_eval_samples: int = 1,
        **data_kwargs,
    ):
        adapter = TabDatasetAdapter(**data_kwargs)
        super().__init__(
            adapter=adapter,
            max_background_tokens=max_background_tokens,
            rewriter_max_length=rewriter_max_length,
            rewriter_min_length=rewriter_min_length,
        )
        self.max_background_tokens = max_background_tokens
        if rewriter_device is None:
            rewriter_device = resolve_device()
        rewriter = BartRewriter(
            model_name=rewriter_model_name,
            device=rewriter_device,
            max_input_tokens=max_background_tokens,
        )
        self.set_rewriter(rewriter)
        self.rewriter_kwargs = {
            "max_length": rewriter_max_length, 
            "min_length": rewriter_min_length, 
            "do_sample": do_sample, 
            "top_k": top_k, 
            "top_p": top_p, 
            "temperature": temperature,
        }
        self.rewrite_background = rewrite_background
        self.chunker = TokenAwareChunker(
                tokenizer=self.rewriter.rewriting_pipeline.tokenizer,
                max_tokens=max_background_tokens,
            )
        self.n_train_samples = n_train_samples
        self.n_eval_samples = n_eval_samples
        self.section_headers = ["PROCEDURE", "THE FACTS", "THE LAW", "AS TO THE FACTS", "COMPLAINTS"]

    def extract_section(self, text: str, section_name: str) -> str:
        lines = text.split('\n')
        in_section = False
        section_content = []
        
        for i, line in enumerate(lines):
            stripped = line.strip()
            
            if stripped == section_name:
                in_section = True
                continue
            
            if in_section and stripped in self.section_headers and stripped != section_name:
                break
            
            if in_section:
                section_content.append(line)
        
        return '\n'.join(section_content).strip()

    def prepare_eval_texts(self, record: DatasetRecord) -> List[str]:
        return [self.rewriter.rewrite(text=record.text, **self.rewriter_kwargs) for _ in range(self.n_eval_samples)]

    def prepare_train_texts(self, record: DatasetRecord) -> List[str]:
        background = []
        
        for section_name in self.section_headers:
            section_text = self.extract_section(record.text, section_name)
            if not section_text:
                continue
            
            key = section_name.lower().replace(" ", "_")
            chunks = self.chunker.chunk(section_text)
            
            if len(chunks) == 1:
                background.append(section_text)
            else:
                for chunk in chunks:
                    background.append(chunk.text)

        if self.rewrite_background:
            background = [self.rewriter.rewrite(bk, **self.rewriter_kwargs) for bk in background for _ in range(self.n_train_samples)]

        return background

    def iter_records(self, progress: bool = False) -> List[AttackerDatasetRecord]:
        for record in self.adapter.iter_records():
            if self._cache_map is not None and record.name in self._cache_map:
                ext = self._cache_map.get(record.name, {})
                train_texts = _normalize_texts(ext.get("train_texts", []))
                eval_texts = _normalize_texts(ext.get("eval_texts", []))
            else:
                train_texts = self.prepare_train_texts(record)
                eval_texts = self.prepare_eval_texts(record)
            yield AttackerDatasetRecord(
                name=record.name,
                train_texts=train_texts,
                eval_texts=eval_texts,
            )
