from __future__ import annotations

import hashlib
import random
from typing import List, Optional, Tuple

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
        max_worst_review_per_company: int = 10,
        max_best_review_per_company: int = 10,
        max_num_companies: int = 50,
        seed: int = 42,
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
        )
        self._seed = seed

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

    def _format_review_entry(self, entry: dict, rng: random.Random) -> Tuple[str, str]:
        review_id = entry.get("review_id")
        stars = entry.get("stars")
        title = entry.get("title")
        text = entry.get("text")
        company_description = entry.get("company_description")
        deid_desc = self._deidentify(company_description).strip()
        title_clean = title.strip() if isinstance(title, str) else ""
        text_clean = text.strip() if isinstance(text, str) else ""
        content = self._review_template(rng).format(stars=stars, title=title_clean, text=text_clean)
        if deid_desc:
            content = f"{content} {self._bio_template(rng).format(bio=deid_desc)}"
        return f"review_{review_id}", content

    def extract_background_knowledge(self, record: DatasetRecord) -> List[Tuple[str, str]]:
        records = record.metadata.get("records") if record.metadata else None
        if not isinstance(records, list):
            raise ValueError("records must be a list")
        rng = random.Random(self._background_seed(record))
        shuffled_records = list(records)
        rng.shuffle(shuffled_records)
        background: List[Tuple[str, str]] = []
        for entry in shuffled_records:
            if not isinstance(entry, dict):
                raise ValueError("records entries must be dicts")
            background.append(self._format_review_entry(entry, rng))
        return background


__all__ = ["TrustpilotAttackerDatasetAdapter"]
