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

### Simple PII Redactors

```bash
python3 model.py --data tab --data_in data/TAB/splitted/test.json --model spacy --model_in configs/model/spacy.yaml --runtime_in configs/runtime/simple.yaml

python3 model.py --data tab --data_in data/TAB/splitted/test.json --model presidio --model_in configs/model/presidio.yaml --runtime_in configs/runtime/simple.yaml

python3 model.py --data tab --data_in data/TAB/splitted/test.json --model manual --model_in configs/model/manual.yaml --runtime_in configs/runtime/simple.yaml
```

### PII Detection (Token Classification)

```bash
# Train PII detector
python3 pii.py --dataset tab --data-path data/TAB/splitted --mode train --epochs 3 --batch-size 8 --use-nervaluate --evaluation-mode partial

# Evaluate PII detector
python3 pii.py --dataset tab --data-path data/TAB/splitted --mode evaluate --model-path models/pii_detectors/tab/20231025_143022 --use-nervaluate

# Predict with PII detector
python3 pii.py --dataset tab --data-path data/TAB/splitted --mode predict --model-path models/pii_detectors/tab/20231025_143022
```

### TRI (Text Re-Identification)

```bash
# Train TRI model
python3 tri.py --dataset tab --data-path data/TAB/tab.json --mode train --finetuning-epochs 15 --use-pretraining --annotation-folder /Users/yay/work/DPMLM/outputs/tab/samples/train_100/annotations/simple --best-metric-dataset spacy

# Evaluate TRI model
python3 tri.py --dataset tab --data-path data/TAB/tab.json --mode evaluate --model-path models/tri_pipelines/tab/20231025_143022 --annotation-folder /Users/yay/work/DPMLM/outputs/tab/samples/train_100/annotations/simple

# Predict with TRI model
python3 tri.py --dataset tab --data-path data/TAB/tab.json --mode predict --model-path models/tri_pipelines/tab/20231025_143022
```

