#!/usr/bin/env python3

from __future__ import annotations

import os
from typing import Callable, Dict

from dp.experiments import ExperimentResult
from dp.experiments.reidentification import ReidentificationRiskExperiment
from dp.experiments.utility.db_bio_label import DBBioLabelExperiment
from dp.experiments.utility.tab_country_year import TabMetadataExperiment
from dp.experiments.utility.trustpilot_stars_category import TrustpilotStarsExperiment


def identity_anonymizer(text: str) -> str:
    return text


def run_experiment(name: str, experiment_factory: Callable[[], ExperimentResult]) -> ExperimentResult:
    result = experiment_factory()
    print(f"{name}")
    for key, value in result.metrics.items():
        print(f"  {key}: {value:.4f}")
    print(f"  score: {result.score:.4f}")
    return result


def aggregate_losses(results: Dict[str, ExperimentResult], weights: Dict[str, float]) -> float:
    total = 0.0
    for key, weight in weights.items():
        if key not in results:
            continue
        loss = results[key].metrics.get("loss", results[key].score)
        total += weight * loss
    return total


def main() -> None:
    os.chdir("/Users/yay/work/dp")
    os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
    anonymize = identity_anonymizer

    trustpilot_experiment = TrustpilotStarsExperiment(
        data_dir="data/trustpilot",
        anonymize=anonymize,
        max_records=5000,
    )
    db_bio_experiment = DBBioLabelExperiment(
        data_dir="data/db_bio",
        split="train",
        anonymize=anonymize,
        max_records=5000,
    )
    tab_experiment = TabMetadataExperiment(
        data_file="data/tab/tab.json",
        target="year_decade",
        anonymize=anonymize,
        max_records=5000,
    )

    results: Dict[str, ExperimentResult] = {}
    results["trustpilot_utility"] = run_experiment("Trustpilot Utility", trustpilot_experiment.execute)
    results["trustpilot_privacy"] = run_experiment(
        "Trustpilot Reidentification",
        lambda: ReidentificationRiskExperiment(
            records=trustpilot_experiment.load_records(),
            anonymize=anonymize,
            max_records=trustpilot_experiment.max_records,
        ).execute(),
    )
    results["db_bio_utility"] = run_experiment("DB-Bio Utility", db_bio_experiment.execute)
    results["tab_utility"] = run_experiment("TAB Utility", tab_experiment.execute)

    weights = {
        "trustpilot_utility": 1.0,
        "trustpilot_privacy": 1.0,
        "db_bio_utility": 1.0,
        "tab_utility": 1.0,
    }
    aggregate = aggregate_losses(results, weights)
    print("Aggregate Loss")
    for key, weight in weights.items():
        if key not in results:
            continue
        loss = results[key].metrics.get("loss", results[key].score)
        contribution = weight * loss
        print(f"  {key}: weight={weight:.2f} contribution={contribution:.4f}")
    print(f"  total: {aggregate:.4f}")


if __name__ == "__main__":
    main()
