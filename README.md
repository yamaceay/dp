# Differentially Private Text Anonymization

This repository implements and benchmarks risk-aware text anonymization methods, with a focus on DP-MLM-X, a token-level differential privacy approach that allocates privacy budget according to per-token re-identification risk rather than uniformly.

## Table of Contents

- [Overview](#overview)
- [Pipeline](#pipeline)
- [Methods](#methods)
- [Datasets](#datasets)
- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Running Anonymization](#running-anonymization)
- [Evaluation](#evaluation)

---

## Overview

The core idea is to replace uniform DP noise injection with risk-aware budget allocation: tokens with higher re-identification risk receive more perturbation, while low-risk tokens are left closer to their original form. Risk scores are derived from a trained Text Re-Identification (TRI) model via SHAP attribution.

The codebase supports a full experiment pipeline: dataset loading, PII detection, TRI model training, risk precomputation, anonymization, and evaluation of privacy, utility, and divergence.

---

## Pipeline

Each experiment follows these ordered stages:

1. **De-identification**: mask direct identifiers (names, locations) using Presidio before further processing.
2. **TRI model training**: fine-tune a `distilbert-base-uncased` classifier to re-identify individuals from text.
3. **Risk precomputation**: run SHAP attribution over the TRI model to assign per-token risk scores (`risk.py`).
4. **Anonymization**: apply the selected method using the risk scores and configured stopping conditions.
5. **Evaluation**: measure privacy (MRR, TRIR), utility (accuracy, MAE), and divergence (cosine similarity, BERTScore, PP) via `run.py`.

Logs from each stage are merged into analysis-ready artifacts by `merge_logs.py` and `transform_logs.py`.

---

## Methods

| Name | Type | Key idea |
|------|------|----------|
| `spacy`, `presidio` | Masking | Entity-based deterministic masking |
| `manual` | Masking | Dataset-provided annotation spans |
| `baroud` | Masking | Trainable PII detector with a confidence threshold |
| `risk` | Masking | Risk-scored masking until cumulative risk falls below a target |
| `petre`, `iter_petre` | k-anonymity | Iterative masking until TRI attacker rank reaches a target |
| `dpmlm`, `iter_dpmlm` | DP rewriting | Risk-aware DP masked language model (DP-MLM-X) |
| `dpbart` | DP rewriting | Gaussian noise on BART encoder logits |
| `dpprompt` | DP rewriting | Prompted seq2seq with clipped logits |
| `dpparaphrase` | DP rewriting | Autoregressive rewriting with DP noise |

Method configs live in `configs/model/<method>/`; runtime parameter sweeps live under `configs/runtime/`.

---

## Datasets

| Dataset | Description |
|---------|-------------|
| **TAB** | ECHR case documents with PII span annotations and utility labels (year, countries). |
| **DB-Bio** | Wikipedia biographies of public figures with DBpedia class labels. |
| **RAT-Bench** | Benchmark used for the BART vs. no-BART background-knowledge ablation. |

---

## Installation

```bash
git clone https://github.com/yamaceay/dp.git
cd dp
uv sync
source .venv/bin/activate
python -m spacy download en_core_web_sm
```

See `pyproject.toml` for the exact dependency versions.

---

## Data Preparation

Datasets are not shipped with the repository due to size. Expected layout:

| Dataset | Path |
|---------|------|
| TAB | `data/tab/echr_{train,dev,test}.json` |
| DB-Bio | `data/db_bio/` (Hugging Face `dataset_dict`, `train`/`validation`/`test`) |
| RAT-Bench | `data/rat_bench/`, `data/rat_bench_nobart/` |

---

## Running Anonymization

```bash
python model.py \
  --data tab \
  --data_in data/tab/echr_test.json \
  --model dpmlm \
  --model_in configs/model/dpmlm/tab/shap_k.yaml \
  --runtime_in configs/runtime/dp/eps_100.yaml \
  --output jsonl
```

On HPC, see the Slurm job tables in `slurm/tables/` for full experiment batches.

---

## Evaluation

```bash
python run.py privacy --config configs/5_experiments/tab/privacy/shap.yaml
python run.py utility --config configs/5_experiments/tab/utility/shap.yaml
python run.py divergence --config configs/5_experiments/tab/divergence/shap.yaml
```

`run.py` reads anonymized JSONL outputs, computes metrics, and writes structured logs consumed by `merge_logs.py` and the tables under `mds/`.
