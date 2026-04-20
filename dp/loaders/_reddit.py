from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, Iterable

from dp.loaders.base import DatasetAdapter, DatasetRecord


class RedditDatasetAdapter(DatasetAdapter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._records = list(self._read_records())
        self.data_in = Path(self.data_in)

    def _read_records(self) -> Iterable[Dict]:
        if isinstance(self.data_in, str):
            self.data_in = Path(self.data_in)
        if self.data_in.is_dir():
            train_file = self.data_in / "train.jsonl"
            test_file = self.data_in / "test.jsonl"
            files = [train_file, test_file]
            for file in files:
                assert file.exists(), f"Expected file not found: {file}"
                yield from self._read_record(file)
            return

        with self.data_in.open("r", encoding="utf-8") as handle:
            yield from self._read_record(self.data_in)
    def __len__(self) -> int:
        return len(self._records)

    def _read_record(self, file: Path) -> Iterable[Dict]:
        with file.open("r", encoding="utf-8") as handle:
            for raw_idx, line in enumerate(handle):
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON on line {raw_idx + 1} in {file}") from exc
                yield item

    def iter_records(self) -> Iterable[DatasetRecord]:
        base_iter = ((idx, row) for idx, row in enumerate(self._records))
        for idx, row in self._slice_records(base_iter):
            text = (row.get("response") or "").strip()
            if not text:
                continue
            persona_raw = row.get("personality") or {}
            persona_hash = self._hash_persona(persona_raw)

            persona = {}
            if persona_raw:
                for key, value in persona_raw.items():
                    persona[f"persona_{key}"] = value

            metadata = {
                "label": row.get("label"),
                "feature": row.get("feature"),
                "hardness": row.get("hardness"),
                "question": row.get("question_asked"),
                "guess": row.get("guess"),
                "guess_correctness": row.get("guess_correctness"),
                **persona,
            }
            yield DatasetRecord(
                text=text,
                uid=f"reddit_{idx + 1}",
                name=persona_hash,
                metadata=metadata,
            )

    @staticmethod
    def _identity_key(persona: Dict) -> str:
        if not persona:
            return "unknown_persona"
        pairs = [f"{key}={persona.get(key)}" for key in sorted(persona.keys())]
        return "|".join(pairs)

    @staticmethod
    def _hash_persona(persona: Dict) -> str:
        """Generate a deterministic hash from personality attributes."""
        if not persona:
            return hashlib.sha256(b"unknown_persona").hexdigest()[:16]
        pairs = [f"{key}={persona.get(key)}" for key in sorted(persona.keys())]
        persona_str = "|".join(pairs)
        return hashlib.sha256(persona_str.encode("utf-8")).hexdigest()[:16]


__all__ = ["RedditDatasetAdapter"]
