# Differentially Private Text Anonymization Toolkit

This repository brings together differential privacy (DP), k-anonymity, heuristic redaction, and empirical re-identification attacks in a single toolkit. The codebase powers experiments on long-form documents (e.g., asylum decisions, consumer reviews) as well as short social media posts and is meant to serve both as a reproducible benchmark and as a starting point for new anonymization research.

## Highlights
- **Multiple privacy controls**: pure DP rewriting (DP-MLM, DP-BART, DP-Prompt, DP-Paraphrase), k-anonymity via PETRE, confidence-thresholded masking, and risk-aware masking.
- **Dataset adapters**: unified loaders for TAB, Trustpilot, DB-Bio, and Reddit style JSON/JSONL corpora, each emitting `DatasetRecord` objects with annotations and metadata.
- **Token selection + explainability**: plug-in risk scorers (`Uniform`, `Greedy`, `SHAP`) and selectors (`PIIOnly`, `ByRisk`, `UntilK`, `All`) coordinate which tokens are modified.
- **Runtime grid search**: YAML-driven parameter sweeps for ε, k, λ, and ρ that stay synchronized with model configs through the runtime bundle loader.
- **Evaluation hooks**: built-in scripts for PII detector training, TRI (Text Re-Identification) attackers, and reporting utilities that score privacy, divergence, and downstream utility.

## Installation

```bash
git clone https://github.com/yay/dp.git
cd dp

# Install the dp package and core dependencies
pip install -e .
pip install -r requirements.txt

# Optional extras
python -m spacy download en_core_web_sm   # needed for spaCy baselines
pip install presidio-analyzer             # needed for Presidio baselines
```

GPU acceleration is recommended for transformer-based anonymizers, TRI attackers, and SHAP explainers. Set `PYTORCH_ENABLE_MPS_FALLBACK=1` when running on Apple Silicon with the MPS backend.

## Data preparation

Raw corpora are **not** shipped with the repository. Each dataset adapter expects a normalized JSON/JSONL export:

| Adapter | Path hint | Expected fields |
| --- | --- | --- |
| `tab` | `data/TAB/splitted/{train,dev,test}.json` | `doc_id`, `text`, nested `annotations` with `entity_mentions`, and `meta.applicant` for record names. |
| `trustpilot` | `data/Trustpilot/...` | Company- or user-level review splits with optional sentiment metadata. |
| `db_bio` | `data/DB_Bio/...` | Biomedical abstracts with entity spans. |
| `reddit` | JSONL file with `response`, `personality`, and auxiliary attributes per line. |

The adapters live in `dp/loaders/` and always emit a `DatasetRecord` with `.text`, `.uid`, `.name`, optional `.spans`, and `.metadata`. Use `--max_records`, `--start`, `--end`, and `--step` to subsample large corpora without rewriting files.

## System overview

1. `model.py` parses CLI arguments, loads dataset adapters, model configs (`configs/model`), and runtime bundles (`configs/runtime`).
2. An `Anonymizer` subclass (see `dp/methods`) is instantiated. It exposes a builder that wires in dataset records, explainers, selectors, token splitters, and output handlers.
3. Token selection happens through `dp.utils.selector` units. They maintain a `TokenLedger`, call `apply_fn` hooks, and emit `AnonymizationStep`s as thresholds change.
4. Token risk scores come from `dp.utils.explainer` implementations (uniform, greedy perturbation, or SHAP over TRI classifiers). Risk-aware methods can ingest precomputed JSONL files produced by `risk.py`.
5. Output is streamed to `dp/utils/output.py` handlers (`print` or `jsonl`) that encode metadata, spans, and token edit histories in `outputs/{dataset}/{model}/timestamp*.jsonl`.

## Implemented anonymization methods

| Name | Module | Technique | Notes |
| --- | --- | --- | --- |
| `spacy` | `dp/methods/_spacy.py` | Deterministic entity masking with spaCy NER. | Masks tokens as `[LABEL]`. Optionally filter by entity labels. |
| `presidio` | `dp/methods/_presidio.py` | Microsoft Presidio analyzer. | Falls back to a placeholder if Presidio is unavailable. |
| `manual` | `dp/methods/_manual.py` | Uses dataset-provided spans. | Requires dataset mode (`--indices`) and annotations. |
| `baroud` | `dp/methods/_baroud.py` | Trainable PII detector + λ thresholding. | Wraps `PIIOnlySelector`; sweeps λ via runtime configs. |
| `risk` | `dp/methods/_risk.py` | Risk-scored masking. | Needs a `TokenExplainer` (e.g., Greedy/SHAP) or precomputed scores and accepts ρ sweeps. |
| `petre` | `dp/methods/_petre.py` | PETRE k-anonymity with TRI ranking. | Iteratively masks high-risk spans until TRI rank ≥ k. |
| `dpmlm` | `dp/methods/_dpmlm.py` | DP masked language model editing. | Supports explainers, selectors, and k/λ/ρ sweeps plus precomputed risk. |
| `dpbart` | `dp/methods/_dpbart.py` | Logit clipping + Gaussian noise on BART encodings. | Requires ε runtime param and optional δ tuning. |
| `dpprompt` | `dp/methods/_dpprompt.py` | Prompted seq2seq rewriting with clipped logits. | Generates paraphrases with ε-calibrated temperature. |
| `dpparaphrase` | `dp/methods/_dpparaphrase.py` | Autoregressive rewriting with chunking-aware DP noise. | Supports tokenizer-aware chunking for overlong inputs. |

Method capabilities are recorded in `dp/methods/constants.py` and enforced via `MODEL_REGISTRY`/`MODEL_CAPABILITIES` (e.g., PETRE requires dataset mode, DP-Prompt forbids token-level selectors).

## Token explainers and selectors

- **Explainers (`dp/utils/explainer`)**: `UniformExplainer` assigns equal risk, `GreedyExplainer` measures TRI score drops when masking tokens, and `ShapExplainer` uses SHAP over TRI classifiers. All expose `.explain(text, offsets)` and integrate with TRI detectors stored under `models/tri_pipelines`.
- **Selectors (`dp/utils/selector`)**: `PIIOnlyUnit` filters tokens above a λ confidence threshold, `ByRiskUnit` masks high-risk spans until remaining probability mass ≤ ρ, `UntilKUnit` keeps anonymizing until the TRI attacker rank exceeds k, and `AllSelector` simply applies edits everywhere.
- **Token splitting**: `dp/utils/splitter.TextSplitter` and `dp/utils/chunking` provide character-span aware tokenization and chunking strategies used by selectors, explainers, and DP generators.

Configure these components inside the model YAMLs under `configs/model/**`. Typical blocks include `token_selection`, `explainer`, `precomputation`, and chunking policies.

## Runtime configuration and parameter grids

`runtime/load_runtime_bundle` ingests one or more YAML files (globs allowed) and merges:

- `configs/runtime/dp/eps_*.yaml` for ε.
- `configs/runtime/pii_confidence/lambda_*.yaml` for λ thresholds.
- `configs/runtime/risk_tolerance/rho_*.yaml` for ρ tolerances.
- `configs/runtime/k_anon/k_*.yaml` for target k values.

Model configs may whitelist which runtime suffixes are valid through a `params` block, preventing accidental mismatches between sweeps and models. At execution time, `buckets_to_dicts` expands k/λ/ρ combinations into grid searches; single-value ε is required for DP models.

## Running anonymization

### Dataset-driven runs

```bash
python model.py \
  --data tab \
  --data_in data/TAB/splitted/test.json \
  --model dpmlm \
  --model_in configs/model/dpmlm/tab/greedy_risk.yaml \
  --runtime_in configs/runtime/dp/eps_100.yaml configs/runtime/risk_tolerance/rho_090.yaml \
  --output jsonl \
  --max_records 32
```

Key arguments:

- `--texts` or `--indices`: mutually exclusive ways to specify inputs. Use indices with dataset-aware models (manual, petre, dpmlm when risk scores are tied to record IDs).
- `--annotations`/`--annotations_in`: attach pre-computed spans (spaCy, Presidio, manual) if a method requires them.
- `--runtime_in`: accepts multiple files or globs; values appear in `result.metadata.hyperparams`.
- `--output`: output handler name (`print` or `jsonl`) as registered in `dp/utils/output.py`.
- `--unique_name`: appended to filenames inside `outputs/{dataset}/{model}/`.

### Free-form text anonymization

```bash
python model.py \
  --model dpprompt \
  --model_in configs/model/dpprompt.yaml \
  --runtime_in configs/runtime/dp/eps_010.yaml \
  --texts "John Doe lives in Seattle and works for Acme Corp."
```

DP models expect `EpsilonParam` buckets, so make sure the runtime bundle contains exactly one ε entry.

## Training utilities and supporting scripts

### PII detector training (`pii.py`)
- Loads adapters for TAB/Trustpilot/DB-Bio.
- Trains Hugging Face token classifiers with optional Nervaluate evaluation.
- Saves checkpoints under `models/pii_detectors/{dataset}/{timestamp}`.
- Modes: `train`, `evaluate`, `predict`.

### TRI attacker training (`tri_by_deid.py`, `tri_by_bk.py`, `tri_by_split.py`)
- Build datasets of anonymized texts by reading JSON/JSONL annotations produced by anonymizers.
- Support MLM pretraining + fine-tuning with configurable epochs, batch sizes, and early stopping.
- Outputs to `models/tri_pipelines/{dataset}/{timestamp}` which are later referenced by explainers and PETRE’s rank evaluator.

### Risk scoring (`risk.py`)
- Applies Greedy or SHAP explainers to each record and dumps JSONL entries containing `uid`, `offsets`, and `scores`.
- These files can be fed back into DP-MLM or PETRE via the `precomputation.risk_scores` config block to skip repeated inference.

## Evaluation pipelines (`run.py`)

`run.py` orchestrates three experiment families defined under `configs/experiments`:

- **Privacy**: builds adversarial datasets from anonymized JSONL files and evaluates TRI success rates (`privacy_*.yaml`).
- **Divergence**: computes semantic similarity metrics (BERTScore, cosine TF-IDF) between original and anonymized texts.
- **Utility**: trains downstream classifiers/regressors (e.g., country/year prediction on TAB) to quantify retained task performance.

Each config chooses datasets, annotation sources, metrics, and output sinks. Use `--mode {report,score}` as documented in `dp/experiments/*`.

## Outputs and logs

- `outputs/{dataset}/{model}/timestamp[_?param].jsonl` – structured records with anonymized text, spans, annotations, metadata, and per-token edit logs.
- `logs/` – training or evaluation logs produced by auxiliary scripts.
- `models/` – checkpoints for TRI pipelines and PII detectors referenced across configs.

Output files capture:

```json
{
  "idx": 42,
  "text": "...anonymized text...",
  "spans": [{"start": 0, "end": 4, "label": "PERSON"}],
  "annotations": {
    "spans": [...],
    "token_edits": [{"kind": "replace", "span": [0,4], "text": "[PERSON]"}]
  },
  "metadata": {
    "unique_name": "demo",
    "hyperparams": {"epsilon": 1, "rho": 0.9},
    "_grid_param": "rho",
    "_grid_value": 0.9
  }
}
```

## Repository layout

```
dp/
├── methods/             # DP, k-anon, and heuristic anonymizers plus registry/capabilities
├── loaders/             # Dataset adapters, annotation helpers, batching utilities
├── utils/               # Selectors, explainers, chunking, output handlers, PII detector
├── tri/                 # TRI attacker implementations (deid, background knowledge, split)
├── experiments/         # Privacy/divergence/utility experiment configs and runners
└── __init__.py
configs/
├── model/               # Method-specific YAML configs
├── runtime/             # ε/λ/ρ/k runtime sweeps
└── experiments/         # Reporting presets for run.py
models/                  # Saved TRI pipelines and PII detectors
outputs/                 # JSONL anonymized corpora grouped by dataset/model
tests/                   # Unit tests for annotations and token ledgers
```

## Development guidelines

`guidelines.md` documents the coding standards enforced in this repo: self-explanatory code, strict typing, minimal dependencies, and builder-style composition. Follow those conventions when adding new anonymizers, selectors, or experiments.