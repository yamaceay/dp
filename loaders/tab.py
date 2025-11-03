from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from dp.loaders.base import AttackerDatasetRecord, DatasetAdapter, DatasetRecord, TextAnnotation
from dp.tri.attacker_adapter import AttackerDatasetAdapter
from dp.utils.chunking import Chunk, TokenAwareChunker
from dp.utils.summarizer import BartSummarizer

class TabDatasetAdapter(DatasetAdapter):
    def __init__(self, data: Optional[str] = None, data_in: Optional[str] = None, max_records: Optional[int] = None):
        self.data_in = Path(data_in)
        self.max_records = max_records
        try:
            with self.data_in.open("r", encoding="utf-8") as handle:
                self._records: List[dict] = json.load(handle)
        except Exception as exc:
            raise RuntimeError(f"Failed to load TAB dataset from {self.data_in}") from exc

    def __len__(self) -> int:
        return len(self._records)

    def iter_records(self) -> Iterable[DatasetRecord]:
        for idx, row in enumerate(self._records):
            if self.max_records is not None and idx >= self.max_records:
                break

            uid = str(row.get("doc_id", idx))
            text = row.get("text", "")
            annotations_raw = row.get("annotations")
            spans = self._read_annotations(annotations_raw)

            name = row.get("meta", {}).get("applicant", "")
            metadata = {
                "country": row.get("meta", {}).get("countries"),
                "years": row.get("meta", {}).get("years"),
                "quality_checked": row.get("quality_checked"),
                "task": row.get("task"),
                "dataset_type": row.get("dataset_type"),
                "meta": row.get("meta"),
            }

            yield DatasetRecord(
                text=text,
                uid=uid,
                name=name,
                spans=spans,
                metadata=metadata,
            )

    def _read_annotations(self, annotations_raw: Optional[List[dict]]) -> Optional[List[TextAnnotation]]:
        if not annotations_raw:
            return None
        annotations_processed = []
        for annotator, annotations_one_person in annotations_raw.items():
            entity_mentions = annotations_one_person.get("entity_mentions", [])
            if not entity_mentions:
                continue
            for mention in entity_mentions:
                annotation = TextAnnotation(
                    start=mention.get("start_offset"),
                    end=mention.get("end_offset"),
                    label=mention.get("entity_type"),
                    text=mention.get("span_text"),
                    annotator=annotator,
                    metadata=mention.get("metadata", {
                        "identifier_type": mention.get("identifier_type"),
                        "confidential_status": mention.get("confidential_status"),
                    }),
                )
                annotations_processed.append(annotation)
        return annotations_processed

class TabAttackerDatasetAdapter(AttackerDatasetAdapter):
    def __init__(
        self,
        *args,
        data: Optional[str] = None,
        data_in: Optional[str] = None,
        max_records: Optional[int] = None,
        summarizer_model_name: str = "facebook/bart-large-cnn",
        summarizer_device: int = -1,
        max_background_tokens: int = 512,
        **kwargs
    ):
        self.adapter = TabDatasetAdapter(data=data, data_in=data_in, max_records=max_records)
        self.max_background_tokens = max_background_tokens
        self._background_chunker: Optional[TokenAwareChunker] = None
        super().__init__(
            adapter=self.adapter,
            max_background_tokens=self.max_background_tokens,
            *args,
            **kwargs,
        )
        # initialize summarizer lazily via set_summarizer, as requested
        summarizer = BartSummarizer(model_name=summarizer_model_name, device=summarizer_device)
        self.set_summarizer(summarizer)

    def _get_background_chunker(self) -> TokenAwareChunker:
        if self._background_chunker is None:
            self._background_chunker = TokenAwareChunker(
                tokenizer=self.summarizer.summarization_pipeline.tokenizer,
                max_tokens=self.max_background_tokens,
            )
        return self._background_chunker

    def _extract_section(self, text: str, section_name: str) -> str:
        main_sections = {"PROCEDURE", "THE FACTS", "THE LAW", "AS TO THE FACTS", "COMPLAINTS", "FOR THESE REASONS THE COURT"}
        
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
        
        return background