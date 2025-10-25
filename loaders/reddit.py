from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, Iterable, Optional

from dp.loaders.base import DatasetAdapter, DatasetRecord


class RedditDatasetAdapter(DatasetAdapter):
    def __init__(self, data: Optional[str] = None, data_in: Optional[str] = None, max_records: Optional[int] = None):
        if data_in is None:
            raise ValueError("data_in must point to a JSONL file")
        path = Path(data_in)
        if not path.exists():
            raise ValueError(f"Reddit dataset file not found: {path}")
        self.data_in = path
        self.max_records = max_records
        self._records = list(self._read_records())

    def _read_records(self) -> Iterable[Dict]:
        with self.data_in.open("r", encoding="utf-8") as handle:
            for raw_idx, line in enumerate(handle):
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON on line {raw_idx + 1} in {self.data_in}") from exc
                yield item

    def __len__(self) -> int:
        return len(self._records)

    def iter_records(self) -> Iterable[DatasetRecord]:
        count = 0
        for idx, row in enumerate(self._records):
            if self.max_records is not None and count >= self.max_records:
                break
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
            count += 1

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
        # Create a stable string representation
        pairs = [f"{key}={persona.get(key)}" for key in sorted(persona.keys())]
        persona_str = "|".join(pairs)
        return hashlib.sha256(persona_str.encode("utf-8")).hexdigest()[:16]


__all__ = ["RedditDatasetAdapter"]
