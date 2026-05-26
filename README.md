# Risk-Aware Differentially Private Text Anonymization

Code and pre-computed results for the paper. The main contribution is **DP-MLM-X**: a token-level differential privacy rewriter that allocates privacy budget proportionally to per-token re-identification risk, derived from SHAP attribution over a trained Text Re-Identification (TRI) classifier.

Benchmarked against masking baselines (Presidio, SpaCy, PETRE) and other DP rewriters (DP-BART, DP-Prompt, DP-Paraphrase) on two datasets across privacy, utility, and divergence metrics.

---

## Methods

| Method | Type | Description |
|--------|------|-------------|
| `presidio`, `spacy` | Masking | Entity-based NER masking |
| `risk` | Masking | Risk-scored token masking with ρ stopping threshold |
| `petre` | k-anonymity | Iterative masking until TRI rank ≥ k |
| `dpmlm` | DP rewriting | Risk-aware DP masked language model (**DP-MLM-X**) |
| `dpbart` | DP rewriting | Gaussian noise on BART encoder logits |
| `dpprompt` | DP rewriting | Prompted seq2seq with clipped logits |
| `dpparaphrase` | DP rewriting | Autoregressive rewriting with DP noise |

DP-MLM-X variants (k-, ρ-) combine risk-aware budget allocation with configurable stopping conditions. Model configs: `configs/model/`. Runtime sweeps (ε, k, ρ): `configs/runtime/`.

---

## Datasets

| Dataset | Records | Labels |
|---------|---------|--------|
| **TAB** | 1268 ECHR case documents | Country, year classification |
| **DB-Bio** | 2419 Wikipedia biographies | DBpedia class classification |

Place datasets at:
- `data/TAB/splitted/{train,dev,test}.json`
- `data/DB_Bio/` (Arrow format)

---

## Installation

```bash
git clone <repo> && cd dp
uv sync
source .venv/bin/activate
python -m spacy download en_core_web_sm
```

Requires Python 3.12.3 and a CUDA-capable GPU. Set `PYTORCH_ENABLE_MPS_FALLBACK=1` on Apple Silicon.

---

## Pipeline

Six ordered stages. The numbered Slurm job tables in `slurm/tables/` reproduce the full experiment batches on HPC:

| Stage | Table prefix | Entry point | Description |
|-------|-------------|-------------|-------------|
| 0 | `0_*` | `model.py` | Simple masking (presidio, spacy) |
| 1 | `1_*` | `data_for_tri.py` | Prepare TRI training data |
| 2 | `2_*` | `exp.py` | Train TRI re-identification classifier |
| 3 | `3_*` | `risk.py` | Precompute SHAP per-token risk scores |
| 4 | `4_*` | `model.py` | DP and risk-aware anonymization |
| 5 | `5_*` | `run.py` | Privacy, utility, and divergence evaluation |

Example single run:

```bash
python model.py \
  --data tab --data_in data/TAB/splitted/test.json \
  --model dpmlm --model_in configs/model/dpmlm/tab/greedy_risk.yaml \
  --runtime_in configs/runtime/dp/eps_100.yaml configs/runtime/k_anon/k_5.yaml \
  --output jsonl
```

---

## Reproducing Paper Results

Pre-computed result CSVs are committed in `mds/`. SHAP explanation figures are in `images/explanations/`. To regenerate CSVs and plots from raw evaluation logs:

```bash
bash scripts/post.sh          # parse → merge → transform → plot
bash scripts/post.sh --skip-transform  # skip CSV regeneration, plot only
```

This runs `parse_runtime.py`, `merge_logs.py`, `transform_logs.py` (writes `mds/`), then `plot_shap_tokens.py` (reads `presidio/{db_bio,tab}.jsonl`).
