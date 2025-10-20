# Overview: Benchmarking Differentially Private Text Anonymization

## Executive Summary

This repository implements a comprehensive benchmark for evaluating differentially private (DP) mechanisms for text anonymization. The framework provides a unified evaluation platform for comparing privacy-preserving text rewriting methods across multiple dimensions: privacy guarantees, utility preservation, and computational efficiency.

## Motivation

Text data contains rich information but poses significant privacy risks when shared or published. Traditional anonymization techniques like redaction or generalization often fail to provide formal privacy guarantees or result in severe utility loss. Differential privacy offers a rigorous mathematical framework for quantifying privacy-utility tradeoffs, but applying DP to text remains challenging due to the discrete, high-dimensional, and sequential nature of language.

## Research Questions

1. **Privacy-Utility Tradeoff**: How do different DP mechanisms balance privacy guarantees (measured by ε) with downstream utility preservation?

2. **Method Comparison**: What are the relative strengths of token-level perturbation (DPMLM), generative approaches (DPPrompt, DPBART), and paraphrasing methods (DPParaphrase)?

3. **Baseline Performance**: How do DP methods compare against simpler baselines like PII redaction (Presidio, SpaCy) and k-anonymity (PETRE)?

4. **Re-identification Risk**: Can we empirically measure privacy through text re-identification (TRI) attacks that attempt to link anonymized text back to individuals?

## Methodology

### Framework Architecture

The benchmark implements a **plug-and-play architecture** with three core abstractions:

```
┌─────────────┐
│   Dataset   │ → Records with text, annotations, metadata
└─────────────┘
       ↓
┌─────────────┐
│ Anonymizer  │ → Privacy mechanism (DP, k-anon, simple)
└─────────────┘
       ↓
┌─────────────┐
│  Evaluator  │ → Utility metrics, privacy attacks
└─────────────┘
```

### Privacy Mechanisms

**Differential Privacy Methods** apply ε-DP guarantees through:
- **DPMLM**: Token replacement via exponential mechanism over masked language model
- **DPPrompt**: Conditional text generation with DP noise injection
- **DPParaphrase**: Semantic paraphrasing with privacy budgets
- **DPBART**: Sequence-to-sequence rewriting with DP training

**K-Anonymity Methods** ensure indistinguishability within groups:
- **PETRE**: Text re-identification guided generalization using TRI risk scores

**Simple Baselines** provide reference points:
- **Presidio**: Rule-based PII detection and redaction
- **SpaCy**: NER-based entity replacement
- **Manual**: Ground-truth annotation based masking

### Privacy Evaluation

**Text Re-Identification (TRI)**: Train discriminative models to predict which individual authored anonymized text. Privacy is measured by:
- **Rank degradation**: Change in author ranking before/after anonymization
- **Top-k accuracy**: Percentage of texts correctly re-identified in top-k predictions
- **Privacy score**: Aggregate metric combining rank and confidence

**Theoretical Guarantees**: For DP methods, track:
- Privacy budget ε consumption
- Composition across multiple releases
- Sensitivity analysis for perturbation mechanisms

### Utility Evaluation

**Information Preservation**:
- **PII Detection**: Precision, recall, F1 for entity recognition (nervaluate)
- **Semantic Similarity**: Embedding distance between original and anonymized text
- **Perplexity**: Language model likelihood of anonymized text

**Downstream Tasks** (placeholder for experiments):
- Text classification (sentiment, topic)
- Named entity recognition
- Question answering
- Information extraction

### Datasets

**TAB (Text Anonymization Benchmark)**: Legal case documents with comprehensive PII annotations
- 1,014 training, 127 validation, 127 test documents
- 8 entity types: PERSON, ORG, LOC, DATE, CODE, QUANTITY, DEM, MISC

**Trustpilot Reviews**: Customer reviews from multiple companies
- Multi-domain evaluation (Amazon, Audible, HSBC, etc.)
- Natural distribution of PII in real-world text

**DB-Bio**: Biomedical abstracts with clinical entities
- Domain-specific terminology and entities
- Scientific text characteristics

## Implementation Strategy

### Modular Design Principles

1. **Separation of Concerns**: Data loading, anonymization, and evaluation are independent
2. **Type Safety**: Explicit type annotations and interface contracts
3. **Reproducibility**: Deterministic execution with seed control
4. **Extensibility**: New methods implement simple `Anonymizer` interface

### Builder Pattern

Complex anonymization flows use builder pattern:
```python
builder = model.builder()
builder.with_texts(texts).with_epsilons([1.0, 5.0, 10.0])
results = builder.anonymize()
```

### Configuration Management

YAML-based configuration files separate:
- **Model config**: Algorithm hyperparameters, paths to pretrained components
- **Runtime config**: Execution parameters (ε, k, batch size)
- **Data config**: Dataset paths, preprocessing options

## Current Status

### Implemented Components
- ✅ Core framework (anonymizer abstraction, builder pattern)
- ✅ All baseline methods (Presidio, SpaCy, Manual)
- ✅ All DP methods (DPMLM, DPPrompt, DPParaphrase, DPBART)
- ✅ K-anonymity method (PETRE)
- ✅ PII detection training pipeline
- ✅ TRI training and evaluation pipeline
- ✅ Data preprocessing for TAB, Trustpilot, DB-Bio

### Pending Work
- ⏳ Comprehensive experiments section
- ⏳ Utility metrics for downstream tasks
- ⏳ Statistical significance testing
- ⏳ Computational cost analysis
- ⏳ Cross-dataset generalization studies

## Expected Contributions

1. **Unified Benchmark**: First comprehensive framework for comparing DP text anonymization methods
2. **Empirical Privacy Measurement**: Novel use of TRI attacks to measure actual privacy preservation
3. **Method Comparison**: Systematic evaluation of token-level vs. generative approaches
4. **Reproducible Results**: Complete implementation with configuration management
5. **Extensible Framework**: Easy integration of new methods and datasets

## Repository Organization

```
dp/
├── data.py              # Dataset preprocessing
├── model.py             # Anonymization execution
├── pii.py               # PII detector training
├── tri.py               # TRI model training
├── methods/             # Anonymization implementations
│   ├── simple/          # Baseline methods
│   ├── dp/              # DP methods
│   └── k_anon/          # K-anonymity methods
├── loaders/             # Dataset adapters
├── utils/               # Shared utilities
│   ├── pii_detector.py  # Token classification
│   └── tri_detector.py  # Re-identification
└── experiments/         # Evaluation pipelines
    ├── reidentification.py
    └── utility/         # Downstream task experiments
```

## Getting Started

See [Examples](examples.md) for detailed usage instructions and [API Reference](api.md) for implementation details.
