from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from dp.loaders.base import DatasetAdapter, DatasetRecord, TextAnnotation

# Priority order when picking which direct identifier anchors a record's name.
_DIRECT_KEY_PRIORITY = ["name", "email", "phone number", "SSN", "address", "credit card number"]
_EMAIL_TOKEN_RE = re.compile(r"[._]+")
_TRAILING_DIGITS_RE = re.compile(r"\d+$")


def _parse_py_literal(raw: Any) -> Any:
    if not isinstance(raw, str):
        return raw
    try:
        return ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        return raw


def _flatten_key(key: str) -> str:
    return key.strip().lower().replace(" ", "_").replace("-", "_")


def _name_from_email(email: str) -> Optional[str]:
    # RAT-Bench's synthetic emails are generated from the person's assigned
    # first/last name (e.g. "Connor Mason" -> "connor.mason2025@gmail.com"),
    # so this recovers it deterministically instead of inventing a new name.
    local = email.split("@", 1)[0]
    tokens = [_TRAILING_DIGITS_RE.sub("", t) for t in _EMAIL_TOKEN_RE.split(local)]
    tokens = [t for t in tokens if t]
    if not tokens:
        return None
    return " ".join(t.capitalize() for t in tokens)


def _same_person(profile_a: Dict[str, Any], profile_b: Dict[str, Any]) -> bool:
    shared = set(profile_a) & set(profile_b)
    if len(shared) < 2:
        return False
    return all(profile_a[k] == profile_b[k] for k in shared)


def _locate_offset(text: str, value: str) -> tuple[int, int]:
    # Indirect identifiers (ACS/PUMS codes like ESR, RAC2P) rarely appear
    # verbatim in the generated text, so most of these legitimately stay
    # unlocatable (-1, -1); only direct identifiers reliably match.
    if not value:
        return -1, -1
    start = text.find(value)
    if start < 0:
        return -1, -1
    return start, start + len(value)


def relocate_spans(text: str, spans: Iterable[TextAnnotation]) -> List[TextAnnotation]:
    """Recompute span offsets against `text` (e.g. after BART-summarizing it)."""
    relocated: List[TextAnnotation] = []
    for span in spans:
        start, end = _locate_offset(text, span.text or "")
        relocated.append(
            TextAnnotation(
                start=start,
                end=end,
                label=span.label,
                text=span.text,
                annotator=span.annotator,
                metadata=span.metadata,
            )
        )
    return relocated


class RatBenchDatasetAdapter(DatasetAdapter):
    """Adapter for imperial-cpg/rat-bench (english config).

    data_in may be:
      - a local directory saved via dataset.save_to_disk()  → loaded from disk
      - any other string (or None)                          → downloaded from HuggingFace and cached

    Each `id` (0-99) is reused across three difficulty variants, but only
    difficulty 1 and 2 reliably share the same synthetic person in practice;
    difficulty 3 frequently reuses the same id number with an unrelated
    profile. Identity grouping (the `name` used as the TRI target) is
    therefore computed by comparing actual profile overlap across the rows
    of an id, not by id alone - see `_assign_names`/`_cluster_by_identity`.
    """

    HF_REPO = "imperial-cpg/rat-bench"
    HF_CONFIG = "english"

    def __init__(self, *args, **kwargs):
        data_in: Optional[str] = kwargs.get("data_in") or (args[1] if len(args) > 1 else None)
        local_path = Path(data_in) if data_in else None
        use_local = local_path is not None and local_path.exists()

        if use_local:
            super().__init__(*args, **kwargs)
        else:
            self.data = kwargs.get("data") or (args[0] if args else "rat_bench")
            self.data_in = data_in or ""
            self.max_records = kwargs.get("max_records")
            self.start = kwargs.get("start")
            self.end = kwargs.get("end")
            self.step = kwargs.get("step")
            self.split = kwargs.get("split")

        self._dataset = self._load(local_path if use_local else None)

    def _load(self, local_path: Optional[Path]):
        try:
            from datasets import load_dataset, load_from_disk
        except ImportError as exc:
            raise RuntimeError("pip install datasets is required for RatBenchDatasetAdapter") from exc

        if local_path is not None:
            ds = load_from_disk(str(local_path))
            # load_from_disk may return Dataset or DatasetDict
            if hasattr(ds, "keys"):
                ds = ds["train"] if "train" in ds else next(iter(ds.values()))
            return ds

        ds = load_dataset(self.HF_REPO, self.HF_CONFIG)
        return ds["train"]

    def __len__(self) -> int:
        return len(self._dataset)

    def iter_records(self) -> Iterable[DatasetRecord]:
        rows: List[Dict[str, Any]] = []
        for idx in range(len(self._dataset)):
            row = dict(self._dataset[idx])
            row["profile"] = _parse_py_literal(row.get("profile") or {})
            row["direct_identifiers"] = _parse_py_literal(row.get("direct_identifiers") or {})
            row["indirect_identifiers"] = _parse_py_literal(row.get("indirect_identifiers") or {})
            rows.append(row)

        names_by_row = self._assign_names(rows)

        base_iter = ((idx, row) for idx, row in enumerate(rows))
        for idx, row in self._slice_records(base_iter):
            profile = row["profile"]
            direct = row["direct_identifiers"]
            indirect = row["indirect_identifiers"]

            uid = f"{row.get('id', idx)}_{row.get('difficulty', idx)}"
            metadata: Dict[str, Any] = {
                "scenario": row.get("scenario"),
                "difficulty": row.get("difficulty"),
            }
            for key, value in profile.items():
                metadata[f"profile_{_flatten_key(key)}"] = value

            spans = self._identifier_spans(direct, indirect, row.get("text", ""))

            yield DatasetRecord(
                text=row.get("text", ""),
                uid=uid,
                name=names_by_row[idx],
                spans=spans,
                metadata=metadata,
            )

    @staticmethod
    def _identifier_spans(direct: Dict[str, Any], indirect: Dict[str, Any], text: str) -> List[TextAnnotation]:
        # RAT-Bench gives attribute/value pairs, not character offsets. We search
        # for each value's literal occurrence in `text` (direct identifiers like
        # names/emails/phones usually appear verbatim; coded indirect identifiers
        # like ESR/RAC2P usually don't) and fall back to dummy (-1, -1) when unfound.
        spans: List[TextAnnotation] = []
        for category, identifiers in (("direct", direct), ("indirect", indirect)):
            for key, value in identifiers.items():
                value_str = str(value)
                start, end = _locate_offset(text, value_str)
                spans.append(
                    TextAnnotation(
                        start=start,
                        end=end,
                        label=key,
                        text=value_str,
                        annotator="rat_bench",
                        metadata={"category": category},
                    )
                )
        return spans

    @classmethod
    def _assign_names(cls, rows: List[Dict[str, Any]]) -> Dict[int, str]:
        rows_by_id: Dict[Any, List[int]] = {}
        for idx, row in enumerate(rows):
            rows_by_id.setdefault(row.get("id"), []).append(idx)

        names: Dict[int, str] = {}
        for record_id, indices in rows_by_id.items():
            clusters = cls._cluster_by_identity(rows, indices)
            multi = len(clusters) > 1
            for rank, cluster in enumerate(clusters, start=1):
                descriptor = cls._cluster_descriptor(rows, cluster)
                suffix = f"#{record_id}.{rank}" if multi else f"#{record_id}"
                name = f"{descriptor} {suffix}"
                for idx in cluster:
                    names[idx] = name
        return names

    @staticmethod
    def _cluster_by_identity(rows: List[Dict[str, Any]], indices: List[int]) -> List[List[int]]:
        # Union-find over the rows sharing an id, linking two rows only when
        # their profiles actually agree wherever they overlap (>=2 shared keys).
        parent = {i: i for i in indices}

        def find(i: int) -> int:
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                a, b = indices[i], indices[j]
                if _same_person(rows[a]["profile"], rows[b]["profile"]):
                    union(a, b)

        clusters: Dict[int, List[int]] = {}
        for i in indices:
            clusters.setdefault(find(i), []).append(i)
        return sorted(clusters.values(), key=lambda c: min(rows[i].get("difficulty", 0) for i in c))

    @staticmethod
    def _cluster_descriptor(rows: List[Dict[str, Any]], cluster: List[int]) -> str:
        for key in _DIRECT_KEY_PRIORITY:
            for idx in cluster:
                value = rows[idx]["direct_identifiers"].get(key)
                if not value:
                    continue
                if key == "name":
                    return str(value)
                if key == "email":
                    derived = _name_from_email(str(value))
                    if derived:
                        return derived
                    continue
                return f"{value} [{key}]"
        merged_profile: Dict[str, Any] = {}
        for idx in cluster:
            merged_profile.update(rows[idx]["profile"])
        if merged_profile:
            return ", ".join(f"{k}={v}" for k, v in sorted(merged_profile.items()))
        return "unknown_profile"


__all__ = ["RatBenchDatasetAdapter", "relocate_spans"]
