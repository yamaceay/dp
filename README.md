# Differentially Private Text Anonymization Benchmark

A comprehensive framework for evaluating privacy-preserving text anonymization methods with formal differential privacy guarantees, k-anonymity, and empirical re-identification attacks.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

This repository provides a unified benchmark for comparing text anonymization methods across three dimensions:
- **Privacy**: Formal ε-DP guarantees and empirical TRI (Text Re-Identification) attacks
- **Utility**: Information preservation for downstream tasks
- **Efficiency**: Computational cost and scalability

### Implemented Methods

**Differential Privacy**:
- **DPMLM**: Token-level perturbation via exponential mechanism over masked language models
- **DPPrompt**: Conditional generation with DP noise injection
- **DPParaphrase**: Encoder-decoder paraphrasing with DP-SGD training
- **DPBART**: Sequence-to-sequence rewriting with DP training

**K-Anonymity**:
- **PETRE**: Privacy Enhancement Using Text Re-Identification (risk-guided generalization)

**Baselines**:
- **Presidio**: Rule-based PII detection and redaction
- **SpaCy**: NER-based entity replacement
- **Manual**: Ground-truth annotation based masking

### Key Features

- 🔒 **Formal Privacy Guarantees**: ε-differential privacy with composition tracking
- 🎯 **Empirical Privacy Measurement**: TRI attacks to measure re-identification risk
- 📊 **Comprehensive Evaluation**: Utility metrics including semantic similarity, downstream tasks
- 🧩 **Modular Architecture**: Plug-and-play components (PII detectors, explainers, selectors)
- 🔧 **Configuration Management**: YAML-based configs for reproducible experiments
- 📈 **Multiple Datasets**: TAB (legal), Trustpilot (reviews), DB-Bio (biomedical)

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/yamaceay/dp.git
cd dp

# Install package in editable mode
pip install -e .

# Install dependencies
pip install -r requirements.txt

# (Optional) Install spaCy model for NER
python -m spacy download en_core_web_sm
```

### Simple Anonymization

```bash
# Anonymize with SpaCy NER
python3 model.py \
    --data tab \
    --data_in data/TAB/splitted/test.json \
    --model spacy \
    --model_in configs/model/spacy.yaml \
    --max_records 10

# Anonymize with differential privacy
python3 model.py \
    --data tab \
    --data_in data/TAB/splitted/test.json \
    --model dpmlm \
    --model_in configs/model/dpmlm.yaml \
    --runtime_in configs/runtime/dp.yaml
```

Output:
```json
{"text": "John Doe visited New York.", "anonymized": "[PERSON] visited [LOCATION].", "epsilon": 1.0}
```

### Training Pipelines

```bash
# Train PII detector
python3 pii.py \
    --dataset tab \
    --data-path data/TAB/splitted \
    --mode train \
    --epochs 5 \
    --use-nervaluate

# Train TRI model (for privacy evaluation)
python3 tri.py \
    --dataset tab \
    --data-path data/TAB/splitted \
    --mode train \
    --finetuning-epochs 10 \
    --use-pretraining
```

## Documentation

Comprehensive documentation is available in the `docs/` directory:

- **[Overview](docs/overview.md)**: Executive summary and research methodology
- **[Architecture](docs/architecture.md)**: System design and component interactions
- **[Data](docs/data.md)**: Dataset descriptions and preprocessing
- **[Methods](docs/methods.md)**: Anonymization algorithms and implementations
- **[Evaluation](docs/evaluation.md)**: Metrics and experimental protocols
- **[API Reference](docs/api.md)**: Detailed API documentation
- **[Examples](docs/examples.md)**: Usage examples and tutorials

## Project Structure

```
dp/
├── docs/                    # Comprehensive documentation
├── data/                    # Datasets (TAB, Trustpilot, DB-Bio)
├── configs/                 # Configuration files
│   ├── model/              # Method-specific configs
│   └── runtime/            # Execution parameters
├── methods/                 # Anonymization implementations
│   ├── simple/             # Baselines (Presidio, SpaCy)
│   ├── dp/                 # DP methods (DPMLM, DPPrompt, etc.)
│   └── k_anon/             # K-anonymity (PETRE)
├── loaders/                 # Dataset adapters
├── utils/                   # Shared utilities
│   ├── pii_detector.py     # Token classification for PII
│   ├── tri_detector.py     # Re-identification attacks
│   └── explainer/          # Importance scoring strategies
├── experiments/             # Evaluation pipelines (placeholder)
├── model.py                 # Main anonymization script
├── pii.py                   # PII detector training
├── tri.py                   # TRI model training
└── data.py                  # Dataset preprocessing
```

## Citation

If you use this benchmark in your research, please cite:

```bibtex
@software{dp_benchmark,
  title={Differentially Private Text Anonymization Benchmark},
  author={[Your Name]},
  year={2025},
  url={https://github.com/yamaceay/dp}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! To add a new anonymization method:

1. Implement the `Anonymizer` interface
2. Register capabilities in the registry
3. Add configuration file in `configs/model/`
4. Add tests and documentation

See [Architecture](docs/architecture.md#extending-with-new-methods) for details.

## Contact

For questions or issues, please open a GitHub issue or contact [maintainer email].

