#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List

from dp.loaders import get_adapter, DatasetRecord
from dp.utils import PIIDetector


def mask_text(text: str, spans: Iterable) -> str:
    if not spans:
        return text
    masked = text
    ordered = sorted(spans, key=lambda span: span.start, reverse=True)
    for span in ordered:
        if span.start < 0 or span.end > len(masked) or span.start >= span.end:
            continue
        masked = masked[: span.start] + "[MASK]" + masked[span.end :]
    return masked


def anonymize_records(records: List[DatasetRecord], detector: PIIDetector) -> List[dict]:
    predicted = detector.predict(records)
    outputs = []
    for original, predicted_record in zip(records, predicted):
        anonymized_text = mask_text(original.text, predicted_record.spans or [])
        outputs.append(
            {
                "uid": original.uid,
                "identity": original.name,
                "original": original.text,
                "anonymized": anonymized_text,
                "metadata": original.metadata,
            }
        )
    return outputs


def write_jsonl(path: Path, items: List[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for item in items:
            handle.write(json.dumps(item, ensure_ascii=False))
            handle.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Anonymization inference for Reddit dataset")
    parser.add_argument("--data-path", type=str, default="data/reddit/train.jsonl")
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--output-path", type=str, default="outputs/reddit/anonymized.jsonl")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--max-length", type=int, default=512)
    args = parser.parse_args()

    adapter = get_adapter("reddit", data_in=args.data_path, max_records=args.max_records)
    records = list(adapter.iter_records())
    if not records:
        raise ValueError("No records loaded from Reddit dataset")

    detector = PIIDetector(model_name=args.model_path, max_length=args.max_length, device=args.device, use_chunking=True)
    detector.set_test_dataset(records)
    anonymized = anonymize_records(records, detector)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_path, anonymized)
    print(f"✓ Saved anonymized Reddit records to {output_path}")


if __name__ == "__main__":
    main()
