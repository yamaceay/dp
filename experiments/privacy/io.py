from __future__ import annotations

from typing import List

from dp.loaders import DatasetRecord, TextAnnotation
from dp.loaders.annotations import apply_annotations


def build_privacy_evaluation_dataset(
    records: List[DatasetRecord],
    annotations_batch: List[List[TextAnnotation]],
    mask_token: str,
) -> List[DatasetRecord]:
    dataset: List[DatasetRecord] = []
    total = len(annotations_batch)
    for index, record in enumerate(records):
        annotations = annotations_batch[index] if index < total else []
        text = apply_annotations(record.text, annotations, replacement=mask_token)
        dataset.append(
            DatasetRecord(
                text=text,
                uid=record.uid,
                name=record.name,
                spans=record.spans,
                metadata=dict(record.metadata),
            )
        )
    return dataset
