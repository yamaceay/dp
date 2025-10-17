# Benchmarking of Differentially Private Mechanisms for Text Rewriting

The goal is to test different differentially private mechanisms for text rewriting, including:

- DPMLM: Differentially Private Masked Language Model
- DPPrompt: Differentially Private Prompt-based Generation
- DPParaphrase: Differentially Private Paraphrasing
- DPBART: Differentially Private BART-based Generation

We also provide other baselines, including:

- Simple annotation tools for PII detection and redaction, such as Presidio and SpaCy, but also BERT-based PII Detectors

- K-Anonymity based text rewriting, requiring a database of texts to ensure k-anonymity, most notably PETRE (Privacy Enhancement Using Text Re-Identification)

The goal is to build a baseline for all methods such that they can be compared in a unified benchmark.

## Quickstart

### Setup

```bash
# Install package in editable mode
pip install -e .

# Install dependencies
pip install -r requirements.txt

# (Optional) Install spaCy model for NER
python -m spacy download en_core_web_sm
```

## Datasets

Data can be preprocessed using `data.py`. Example commands:

```bash
python3 data.py --data tab --data_in data/TAB/tab.json # TAB
python3 data.py --data trustpilot --data_in data/trustpilot/www.amazon.com/train.json # Trustpilot
python3 data.py --data db_bio --data_in data/db_bio/test/data-00000-of-00001.arrow # DB-Bio
```

## Models

You can run different models using `model.py`. Example commands for simple PII redactors:

```bash
python3 model.py --data tab --data_in data/TAB/splitted/test.json --model spacy --model_in configs/model/spacy.yaml --runtime_in configs/runtime/simple.yaml # SpaCy PII Redactor

python3 model.py --data tab --data_in data/TAB/splitted/test.json --model presidio --model_in configs/model/presidio.yaml --runtime_in configs/runtime/simple.yaml # Presidio PII Redactor

python3 model.py --data tab --data_in data/TAB/splitted/test.json --model manual --model_in configs/model/manual.yaml --runtime_in configs/runtime/simple.yaml # Manual PII Redactor
```

