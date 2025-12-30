from __future__ import annotations

import hashlib
import random
from typing import List, Optional, Tuple, Dict, Iterable

from dp.loaders.base import DatasetRecord
from dp.loaders.trustpilot import TrustpilotDatasetAdapter
from dp.tri.loaders.base import AttackerDatasetAdapter

class TrustpilotAttackerDatasetAdapter(AttackerDatasetAdapter):
    def __init__(
        self,
        data: Optional[str] = None,
        data_in: Optional[str] = None,
        start: Optional[int] = None,
        end: Optional[int] = None,
        step: Optional[int] = None,
        max_records: Optional[int] = None,
        seed: int = 42,
        deidentify: bool = False,
    ) -> None:
        adapter = TrustpilotDatasetAdapter(
            data=data,
            data_in=data_in,
            start=start,
            end=end,
            step=step,
            max_records=max_records,
        )
        super().__init__(
            adapter=adapter,
            use_records_list=True,
        )
        self._seed = seed
        self._deidentify = deidentify

        try:
            from presidio_analyzer import AnalyzerEngine
        except Exception:
            self._analyzer = None
            self._presidio_available = False
        else:
            try:
                self._analyzer = AnalyzerEngine()
                self._presidio_available = True
            except Exception:
                self._analyzer = None
                self._presidio_available = False

        self._records_original = self.adapter.iter_records()
        self._records = self._group_by_company(self._records_original)

    def _group_by_company(self, records: Iterable[DatasetRecord]) -> Iterable[DatasetRecord]:
        records_by_company: Dict[str, DatasetRecord] = {}
        company_descs: Dict[str, str] = {}
        for record in records:
            company_name = record.name
            if company_name not in company_descs:
                company_desc = record.metadata.pop("company_description")
                company_desc = company_desc.strip() if isinstance(company_desc, str) else ""
                deid_company_desc = self._deidentify(company_desc) if self._deidentify else company_desc
                company_descs[company_name] = deid_company_desc
            company_desc = company_descs[company_name]
            category = record.metadata.pop("category")
            deid_text = self._deidentify(record.text) if self._deidentify else record.text
            review = {
                "review_id": record.uid,
                "text": deid_text,
                **record.metadata,
            }
            target_record = records_by_company.get(company_name, DatasetRecord(
                text=company_desc,
                uid=company_name,
                name=company_name,
                spans=[],
                metadata={"category": category, "records": []}
            ))
            target_record.metadata["records"].append(review)
            records_by_company[company_name] = target_record
        
        for record in records_by_company.values():
            yield record

    def _deidentify(self, text: str) -> str:
        if not text or not self._presidio_available or self._analyzer is None:
            return text or ""
        results = self._analyzer.analyze(text=text, language="en")
        anonymized_text = text
        results_sorted = sorted(results, key=lambda r: r.start, reverse=True)
        for r in results_sorted:
            start = int(r.start)
            end = int(r.end)
            anonymized_text = anonymized_text[:start] + f"[{r.entity_type}]" + anonymized_text[end:] 
        return anonymized_text

    def _background_seed(self, record: DatasetRecord) -> int:
        seed_source = f"{self._seed}|{record.uid}|{record.name}"
        digest = hashlib.sha256(seed_source.encode("utf-8")).hexdigest()
        return int(digest[:16], 16)

    def _review_template(self, rng: random.Random) -> str:
        templates = [
            "A {stars}-star review titled '{title}'. Details: {text}",
            "Review ({stars} stars): '{title}'. {text}",
            "Title: '{title}'. Rating: {stars} stars. Review: {text}",
            "{stars} stars for '{title}'. The reviewer wrote: {text}",
            "Rated {stars} stars. Title: '{title}'. Comment: {text}",
        ]
        return rng.choice(templates)

    def _bio_template(self, rng: random.Random) -> str:
        templates = [
            "Company bio: {bio}",
            "About the company: {bio}",
            "Company description: {bio}",
        ]
        return rng.choice(templates)

    def extract_background_knowledge(self, record: DatasetRecord) -> List[Tuple[str, str]]:
        background_knowledge: List[Tuple[str, str]] = []

        for entry in record.metadata.get("records", []):
            review_id = entry.get("review_id")
            stars = entry.get("stars")
            title = entry.get("title")
            text = entry.get("text")
            company_description = record.text.strip() if record.text else ""
            rng = random.Random(self._background_seed(record) + int(review_id, 36))
            title_clean = title.strip() if isinstance(title, str) else ""
            text_clean = text.strip() if isinstance(text, str) else ""
            content = self._review_template(rng).format(stars=stars, title=title_clean, text=text_clean)
            if company_description:
                content = f"{content} {self._bio_template(rng).format(bio=company_description)}"
            background_knowledge.append((review_id, content))

        return background_knowledge


__all__ = ["TrustpilotAttackerDatasetAdapter"]
