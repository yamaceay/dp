#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from experiments.privacy_annotations import AnnotationPrivacyExperiment
from loaders import DatasetRecord, TextAnnotation, get_adapter
from loaders.annotations import apply_annotations, read_batch_annotations_from_path


def load_dataset_records(
    dataset: str,
    data_in: Optional[str],
    max_records: Optional[int],
) -> List[DatasetRecord]:
    adapter = get_adapter(dataset, data=dataset, data_in=data_in, max_records=max_records)
    return list(adapter.iter_records())

def build_evaluation_dataset(
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
                utilities=dict(record.utilities),
                metadata=dict(record.metadata),
            )
        )
    return dataset


def collect_annotation_sources(
    *paths: str,
) -> Dict[str, Path]:
    paths = [Path(p) for p in paths]
    list_discovered = set()

    names = []
    def add_name(path: Path) -> None:
        dataset_name = path.parent.parent.stem
        method_name = path.parent.stem
        name = f"{method_name}_{dataset_name}"
        count = sum(1 for n in names if n[0] == name)
        names.append((name, count))
    for path in paths:
        if path.is_file():
            list_discovered.add(path)
            add_name(path)
        elif path.is_dir():
            for file_path in path.rglob("*.jsonl"):
                list_discovered.add(file_path)
                add_name(file_path)

    discovered_dict = {f"{name[0]}_{name[1]}": p for name, p in zip(names, list_discovered)}
    return discovered_dict

def main() -> None:
    parser = argparse.ArgumentParser(description="Run annotation-driven privacy experiment")
    parser.add_argument("--dataset", required=True, help="Dataset adapter name")
    parser.add_argument("--data_in", required=True, help="Dataset input path")
    parser.add_argument("--max_records", type=int, default=None, help="Maximum records to evaluate")
    parser.add_argument("--tri_pipeline", required=True, help="Path to TRI pipeline directory")
    parser.add_argument("--tri_max_length", type=int, default=512, help="TRI maximum sequence length")
    parser.add_argument("--tri_device", type=str, default="auto", help="TRI device (auto, cpu, cuda, mps)")
    parser.add_argument("--annotations_in", nargs='+', help="One or more directories or files to search recursively for annotation files")
    parser.add_argument("--mask_token", type=str, default="[MASK]", help="Token used to mask annotations")
    args = parser.parse_args()

    original_dataset = load_dataset_records(args.dataset, args.data_in, args.max_records)
    if not original_dataset:
        raise RuntimeError("No records loaded from dataset")

    evaluation_datasets: Dict[str, List[DatasetRecord]] = {}
    annotation_sources: Dict[str, Path] = {}
    annotations_found = collect_annotation_sources(*args.annotations_in)
    if not annotations_found:
        raise RuntimeError("No annotation files discovered")
    for name, path in annotations_found.items():
        annotations_batch = read_batch_annotations_from_path(str(path))
        evaluation_dataset = build_evaluation_dataset(original_dataset, annotations_batch, args.mask_token)
        evaluation_datasets[name] = evaluation_dataset
        annotation_sources[name] = path

    print(f"Original dataset records: {len(original_dataset)}")
    for name, dataset in evaluation_datasets.items():
        source = annotation_sources[name]
        print(f"Evaluation dataset '{name}' from {source}: {len(dataset)} records")

    experiment = AnnotationPrivacyExperiment(
        dataset_name=args.dataset,
        original_dataset=original_dataset,
        evaluation_datasets=evaluation_datasets,
        tri_pipeline=args.tri_pipeline,
        tri_max_length=args.tri_max_length,
        tri_device=args.tri_device,
    )

    experiment.setup(progress=True)
    result = experiment.run(progress=True)
    experiment.cleanup()

    print("Original ranks")
    uid_to_name = result.metrics.get("uids", {})
    for uid, rank in sorted(result.metrics["original_ranks"].items()):
        name = uid_to_name.get(uid, "")
        suffix = f" ({name})" if name else ""
        print(f"  {uid}{suffix}: {rank}")

    print("Evaluation datasets")
    evaluations: Iterable[Tuple[str, Dict[str, Any]]] = result.metrics["evaluations"].items()
    for eval_name, payload in evaluations:
        ranks: Dict[str, int] = payload.get("ranks", {})
        deltas: Dict[str, int] = payload.get("rank_deltas", {})
        summary = payload.get("rank_delta_summary")
        print(f"  {eval_name}")
        if summary:
            print(
                f"    summary: count={summary['count']} "
                f"improved={summary['improved']} degraded={summary['degraded']} "
                f"unchanged={summary['unchanged']} mean={summary['mean']:.4f} "
                f"median={summary['median']:.4f} min={summary['min']} max={summary['max']}"
            )
        for uid, rank in sorted(ranks.items()):
            name = uid_to_name.get(uid, "")
            suffix = f" ({name})" if name else ""
            if uid in deltas:
                print(f"    {uid}{suffix}: {rank} delta={deltas[uid]:+d}")
            else:
                print(f"    {uid}{suffix}: {rank}")


if __name__ == "__main__":
    main()
