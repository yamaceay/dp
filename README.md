# Differentially Private Text Anonymization

This repository implements and benchmarks risk-aware text anonymization methods, with a focus on DP-MLM-X — a token-level differential privacy approach that allocates privacy budget according to per-token re-identification risk rather than uniformly. It accompanies a Master's thesis evaluating privacy-utility trade-offs across multiple anonymization strategies on two English text datasets.

## Table of Contents

- [Overview](#overview)
- [Pipeline](#pipeline)
- [Methods](#methods)
- [Datasets](#datasets)
- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Running Anonymization](#running-anonymization)
- [Evaluation](#evaluation)
- [Reproducing the Thesis](#reproducing-the-thesis)

---

## Overview

The core idea is to replace uniform DP noise injection with risk-aware budget allocation: tokens with higher re-identification risk receive more perturbation, while low-risk tokens are left closer to their original form. Risk scores are derived from a trained Text Re-Identification (TRI) model via SHAP attribution.

The codebase supports a full experiment pipeline: dataset loading, PII detection, TRI model training, risk precomputation, anonymization, and evaluation of privacy, utility, and divergence.

---

## Pipeline

Each experiment follows these ordered stages:

1. **De-identification** — mask direct identifiers (names, locations) using Presidio before further processing.
2. **Background knowledge generation** — summarize original records with `facebook/bart-large-cnn` to simulate attacker background knowledge.
3. **TRI model training** — fine-tune a `distilbert-base-uncased` classifier to re-identify individuals from text.
4. **Risk precomputation** — run SHAP attribution over the TRI model to assign per-token risk scores (`risk.py`).
5. **Anonymization** — apply the selected method using the risk scores and configured stopping conditions.
6. **Evaluation** — measure privacy (MRR, TRIR), utility (accuracy, MAE), and divergence (cosine similarity, BERTScore, PP) via `run.py`.

Logs from each stage are merged into analysis-ready artifacts by `merge_logs.py`, then exported to thesis CSV tables by `docs/thesis/data_export.py`.

---

## Methods

| Name | Type | Key idea |
|------|------|----------|
| `spacy`, `presidio` | Masking | Entity-based deterministic masking |
| `manual` | Masking | Dataset-provided annotation spans |
| `baroud` | Masking | Trainable PII detector with λ confidence threshold |
| `risk` | Masking | Risk-scored masking until cumulative risk ≤ ρ |
| `petre` | k-anonymity | Iterative masking until TRI attacker rank ≥ k |
| `dpmlm` | DP rewriting | Risk-aware DP masked language model (DP-MLM-X) |
| `dpbart` | DP rewriting | Gaussian noise on BART encoder logits |
| `dpprompt` | DP rewriting | Prompted seq2seq with clipped logits |
| `dpparaphrase` | DP rewriting | Autoregressive rewriting with DP noise |

DP-MLM-X variants (`k-`, `ρ-`, `λ-DP-MLM-X`) combine risk-aware budget allocation with configurable stopping conditions. Method configs live in `configs/model/`, runtime parameter sweeps (ε, λ, ρ, k) in `configs/runtime/`.

---

## Datasets

| Dataset | Description |
|---------|-------------|
| **TAB** | 1268 ECHR case documents with PII span annotations and utility labels (year, countries). |
| **DB-Bio** | 2419 Wikipedia biographies of public figures with DBpedia class labels. |

Both are used with their original train/validation/test splits for all pipeline stages.

---

## Installation

```bash
git clone https://github.com/yamaceay/dp.git
cd dp
uv sync
source .venv/bin/activate
python -m spacy download en_core_web_sm
```

Requires Python 3.12.3, CUDA-capable GPU, and matching GPU driver. Set `PYTORCH_ENABLE_MPS_FALLBACK=1` for Apple Silicon.

---

## Data Preparation

Datasets are not shipped with the repository. Place them at:

| Dataset | Path |
|---------|------|
| TAB | `data/TAB/splitted/{train,dev,test}.json` |
| DB-Bio | `data/DB_Bio/` (Arrow format) |

---

## Running Anonymization

```bash
python model.py \
  --data tab \
  --data_in data/TAB/splitted/test.json \
  --model dpmlm \
  --model_in configs/model/dpmlm/tab/greedy_risk.yaml \
  --runtime_in configs/runtime/dp/eps_100.yaml configs/runtime/risk_tolerance/rho_090.yaml \
  --output jsonl
```

On HPC, use the Slurm job tables in `slurm/tables/` to reproduce the full experiment batches.

---

## Evaluation

```bash
python run.py --config configs/experiments/privacy_tab.yaml
python run.py --config configs/experiments/utility_tab.yaml
python run.py --config configs/experiments/divergence_tab.yaml
```

`run.py` reads anonymized JSONL outputs from `outputs/`, computes metrics, and writes structured logs for downstream analysis.

---

## Reproducing the Thesis

Pinned revision: `https://github.com/yamaceay/dp/tree/d36fc7af2aae6269d2a55987c9fed9cd942d7712`

1. Check out the pinned revision above.
2. Run `uv sync` and verify GPU and CUDA availability.
3. Place TAB and DB-Bio in the expected `data/` paths with original splits.
4. Run the pipeline stages via local scripts or Slurm job tables in `slurm/tables/`.
5. Merge logs and regenerate thesis CSV exports: `python3 docs/thesis/data_export.py`.
6. Regenerate figures: `python3 plot_summary.py`, `python3 pu_tradeoff.py` (and related scripts).
7. Compile the thesis (`docs/thesis/`) and verify tables and figures match the regenerated artifacts.
