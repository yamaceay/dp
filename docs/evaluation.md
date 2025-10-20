# Evaluation

## Evaluation Framework

The benchmark evaluates methods across three dimensions:

1. **Privacy**: Formal guarantees (ε, k) and empirical measurement (TRI attacks)
2. **Utility**: Information preservation for downstream tasks
3. **Efficiency**: Computational cost and scalability

## Privacy Evaluation

### Formal Privacy Guarantees

**Differential Privacy Methods**:
- **Parameter**: Privacy budget ε (lower = stronger privacy)
- **Tracking**: Composition across multiple releases or tokens
- **Validation**: Verify sensitivity calculations, noise calibration

**K-Anonymity Methods**:
- **Parameter**: k value (higher = stronger privacy)
- **Validation**: Verify indistinguishability set size
- **Limitation**: Heuristic, no worst-case guarantees

### Empirical Privacy: Text Re-Identification (TRI)

**Threat Model**: Adversary attempts to link anonymized text back to individual author

**Attack Setup**:
1. Train classifier on (text, author) pairs from dataset
2. Given anonymized text, predict which author wrote it
3. Measure success rate and confidence

**Metrics**:

**Rank Metrics**:
- **Original Rank**: Position of true author in sorted predictions on original text
- **Anonymized Rank**: Position of true author after anonymization
- **Rank Degradation**: `anonymized_rank - original_rank` (higher = better privacy)
- **Rank Drop Rate**: Percentage of texts where rank worsens

**Accuracy Metrics**:
- **Top-1 Accuracy**: Percentage of texts where true author is top prediction
- **Top-k Accuracy**: Percentage where true author is in top-k predictions
- **Mean Reciprocal Rank (MRR)**: `1 / rank` averaged across texts

**Confidence Metrics**:
- **Original Confidence**: P(true_author | original_text)
- **Anonymized Confidence**: P(true_author | anonymized_text)
- **Confidence Drop**: `original_confidence - anonymized_confidence`

**Privacy Score** (aggregate):
$$\text{Privacy} = \alpha \cdot \text{NormalizedRankDrop} + (1-\alpha) \cdot \text{ConfidenceDrop}$$

**Implementation**: `tri.py --mode evaluate`

**Training Requirements**:
- Minimum 10 texts per author for reliable classifier
- Balanced or stratified sampling across authors
- Hold-out test set for unbiased evaluation

**Limitations**:
- Only measures one attack vector (authorship)
- Requires author metadata (not always available)
- Strong attack depends on TRI model quality

### Privacy Budget Tracking

**Composition Theorems**:
- **Parallel Composition**: Independent mechanisms, ε_total = max(ε_i)
- **Sequential Composition**: Dependent mechanisms, ε_total = Σ ε_i
- **Advanced Composition**: Tighter bounds via moments accountant

**Example (DPMLM)**:
- Each token gets ε/n budget
- All tokens processed in parallel → total ε via parallel composition
- Multiple releases of same text → sequential composition

**Tracking**: Store ε consumption in `AnonymizationResult.metadata`

## Utility Evaluation

### Information Preservation Metrics

#### PII Detection Quality

**Task**: How well can PII be detected after anonymization?

**Metrics** (using nervaluate):
- **Strict**: Exact boundary and type match
  - Precision: TP / (TP + FP)
  - Recall: TP / (TP + FN)
  - F1: Harmonic mean
- **Partial**: Partial overlap counts
- **Exact**: Boundary match regardless of type
- **Entity Type**: Type match regardless of boundary

**Evaluation Modes**:
```python
evaluator = Evaluator(true_entities, pred_entities, tags=entity_types)
results = evaluator.evaluate()
strict_f1 = results["overall"]["strict"].f1
```

**Implementation**: `pii.py --mode evaluate --use-nervaluate`

**Interpretation**:
- High scores after anonymization = PII still detectable (privacy leak)
- Low scores = effective obfuscation (but may indicate over-redaction)

#### Semantic Similarity

**Task**: How much meaning is preserved?

**Metrics**:
- **Cosine Similarity**: Between sentence embeddings (SBERT, SimCSE)
  - `sim = cos(embed(original), embed(anonymized))`
  - Range: [-1, 1], higher = more similar
- **BLEU**: N-gram overlap (common in machine translation)
  - Measures surface-level preservation
- **BERTScore**: Token-level semantic similarity via BERT embeddings
  - Captures paraphrase better than BLEU

**Implementation**: `experiments/utility/semantic_similarity.py` (placeholder)

**Interpretation**:
- High similarity = good utility preservation
- Threshold depends on use case (e.g., ≥0.8 for high-quality paraphrase)

#### Perplexity

**Task**: How fluent is the anonymized text?

**Metric**: Perplexity from language model (GPT-2)
$$\text{PPL} = \exp\left(-\frac{1}{N}\sum_{i=1}^N \log P(w_i | w_{<i})\right)$$

**Implementation**:
```python
def compute_perplexity(text, model):
    loss = model(text, labels=text).loss
    return torch.exp(loss).item()
```

**Interpretation**:
- Lower perplexity = more fluent, natural text
- Baseline: original text perplexity ~20-50 on general corpora
- Redaction methods often have higher perplexity due to placeholders

### Downstream Task Utility

*Placeholder for experiments section*

#### Text Classification

**TAB Dataset**:
- **Task**: Classify legal case by type (to be defined from metadata)
- **Metric**: Accuracy, F1 (macro/micro)
- **Setup**: Train classifier on original texts, evaluate on anonymized

**Trustpilot Dataset**:
- **Task**: Predict star rating (1-5) from review text
- **Metric**: Accuracy, MAE (mean absolute error)
- **Setup**: Train on original, evaluate on anonymized

**Interpretation**:
- Utility retention = `accuracy(anonymized) / accuracy(original)`
- Target: ≥0.9 retention for high-utility methods

#### Named Entity Recognition

**Task**: Train NER model on anonymized data, evaluate on original test set

**Metric**: F1 on entity types

**Setup**:
1. Anonymize training data with method M
2. Train NER model on anonymized train set
3. Evaluate on original test set (gold labels)

**Interpretation**:
- Measures if anonymized data retains enough signal for learning
- Relevant for data sharing scenarios (release training data)

#### Question Answering

**Task**: Answer questions about text after anonymization

**Metric**: F1, exact match on answer spans

**Dataset**: SQuAD-style QA on legal/review texts

**Interpretation**:
- Tests if critical information survives anonymization
- Challenging for redaction methods (answers often removed)

### Utility-Privacy Tradeoff Curves

**Visualization**: Plot utility (y-axis) vs. privacy parameter (x-axis)

**Example (DPMLM)**:
- X-axis: ε ∈ [0.1, 1.0, 5.0, 10.0]
- Y-axis: Semantic similarity, classification accuracy
- Curve shows diminishing returns: increasing ε beyond threshold yields little utility gain

**Statistical Analysis**:
- Fit Pareto frontier to identify optimal tradeoff points
- Compute area under curve (AUC) to compare methods
- Highlight dominated methods (strictly worse on both dimensions)

## Efficiency Evaluation

### Computational Cost

**Metrics**:
- **Training Time**: Hours to train PII detector, TRI model, or DP model
- **Inference Time**: Milliseconds per document for anonymization
- **Throughput**: Documents processed per second
- **GPU Memory**: Peak memory usage during inference

**Measurement**:
```python
import time
start = time.time()
result = anonymizer.anonymize(text)
latency = time.time() - start
throughput = len(texts) / total_time
```

**Implementation**: Already added to `model.py` (time profiling)

**Comparison Baseline**: Simple methods (Presidio, SpaCy) should be fastest

### Scalability

**Dataset Size**:
- Measure how performance scales with number of documents
- Linear scaling expected for independent methods
- Sublinear for methods with shared preprocessing (e.g., TRI model loading)

**Text Length**:
- Plot latency vs. document length
- Check for quadratic scaling (attention mechanisms)
- Evaluate chunking strategies for long documents

### Resource Requirements

**Hardware**:
- GPU vs. CPU runtime comparison
- Memory constraints for large batch sizes
- Disk I/O for checkpoint loading

**Deployment**:
- Model size (MB) for storage
- Cold start time (model loading)
- Concurrent request handling

## Experimental Design

### Evaluation Protocols

#### Held-out Test Set

**Setup**: Train on train set, select hyperparameters on validation set, report on test set

**Metrics**: All utility and privacy metrics on test set only

**Justification**: Prevents overfitting to test data

#### Cross-dataset Generalization

**Setup**: Train PII detector on TAB, evaluate on Trustpilot

**Metrics**: PII detection F1 drop across domains

**Interpretation**: Tests robustness to domain shift

#### Ablation Studies

**Components to ablate**:
- Filtering strategy: All tokens vs. PII only
- Scoring strategy: Uniform vs. TRI-based
- Chunking strategy: Truncate vs. sliding window

**Metric**: Change in utility/privacy/efficiency

**Interpretation**: Quantifies contribution of each component

### Statistical Significance

**Hypothesis Testing**:
- **Null Hypothesis**: Method A and B have equal performance
- **Test**: Paired t-test (same texts, different methods) or McNemar's test (classification)
- **Significance Level**: α = 0.05 with Bonferroni correction for multiple comparisons

**Confidence Intervals**:
- Bootstrap resampling for 95% CI on metrics
- Report mean ± std across multiple runs (different seeds)

**Power Analysis**:
- Determine minimum dataset size for detecting effect size δ
- Ensure sufficient test set size (typically n ≥ 100 for text tasks)

### Reproducibility

**Random Seeds**: Fix seeds for all random operations
```python
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
```

**Version Control**: Pin library versions in `requirements.txt`

**Checkpoints**: Save all model checkpoints with timestamps

**Logs**: Record all hyperparameters and metrics to structured logs

## Experiments (Placeholder)

### Experiment 1: Privacy-Utility Tradeoff

**Goal**: Compare methods across ε/k values for optimal tradeoff

**Setup**:
- Methods: DPMLM, DPPrompt, PETRE
- Parameters: ε ∈ [0.1, 1, 5, 10], k ∈ [2, 5, 10]
- Datasets: TAB, Trustpilot
- Metrics: Semantic similarity, classification F1, TRI rank drop

**Expected Results**:
- Higher ε/k → better utility, worse privacy
- DPPrompt preserves utility best but costs more ε
- DPMLM efficient for small ε

**Analysis**: Plot Pareto frontier, identify dominated methods

### Experiment 2: Downstream Task Utility

**Goal**: Measure impact on classification, NER, QA

**Setup**:
- Methods: All methods at fixed ε=1.0 or k=5
- Tasks: Sentiment classification (Trustpilot), case classification (TAB)
- Metrics: Accuracy, F1, utility retention

**Expected Results**:
- Generative methods (DPParaphrase, DPPrompt) retain utility best
- Redaction methods (Presidio, SpaCy) lose significant utility
- Manual provides upper bound (perfect PII detection)

**Analysis**: Rank methods by utility retention

### Experiment 3: Empirical Privacy (TRI Attacks)

**Goal**: Validate that formal privacy translates to re-identification resistance

**Setup**:
- Train TRI on original TAB texts
- Anonymize test set with all methods
- Measure rank drop and confidence drop

**Expected Results**:
- DP methods show significant rank drop
- Correlation between ε and TRI confidence (smaller ε → more drop)
- PETRE explicitly optimizes for TRI, should perform well

**Analysis**: Compare formal ε with empirical TRI metrics

### Experiment 4: Computational Efficiency

**Goal**: Benchmark runtime and resource usage

**Setup**:
- Process 1000 documents from TAB
- Measure total time, peak memory, throughput
- Compare GPU vs. CPU

**Expected Results**:
- Simple methods: <1s total (CPU sufficient)
- DP methods: 100-500s total (GPU beneficial)
- PETRE: slowest due to iterative masking

**Analysis**: Cost-benefit analysis (runtime vs. utility)

### Experiment 5: Component Ablation

**Goal**: Quantify impact of filtering and scoring strategies

**Setup**:
- DPMLM with: (1) All tokens + uniform, (2) PII only + uniform, (3) PII only + TRI
- Measure utility and privacy

**Expected Results**:
- PII-only filtering improves efficiency (fewer tokens)
- TRI-based scoring improves privacy (targets risky tokens)
- Trade-off: more components = more complexity

**Analysis**: Decide on default configuration

### Experiment 6: Cross-dataset Generalization

**Goal**: Test robustness to domain shift

**Setup**:
- Train PII detector on TAB, evaluate on Trustpilot and DB-Bio
- Anonymize with trained models, measure utility

**Expected Results**:
- Performance drop on out-of-domain data
- Domain-specific models outperform general models
- Simple methods more robust (don't rely on training data)

**Analysis**: Identify transferable components

### Experiment 7: Long Document Handling

**Goal**: Evaluate chunking strategies for texts exceeding 512 tokens

**Setup**:
- Filter TAB to documents >1000 tokens
- Compare truncate vs. sliding window
- Measure entity coverage, utility retention

**Expected Results**:
- Truncate loses information in later parts
- Sliding window preserves more but increases cost
- Trade-off: coverage vs. efficiency

**Analysis**: Recommend chunking strategy per method

## Reporting

### Tables

**Table 1: Method Comparison**
| Method      | Privacy | Semantic Sim | Classification F1 | TRI Rank Drop | Inference Time |
|-------------|---------|--------------|-------------------|---------------|----------------|
| Presidio    | None    | 0.82         | 0.76              | +2.3          | 10ms           |
| DPMLM ε=1   | 1.0     | 0.88         | 0.84              | +5.7          | 120ms          |
| DPPrompt ε=1| 1.0     | 0.91         | 0.88              | +4.2          | 480ms          |
| PETRE k=5   | k=5     | 0.85         | 0.79              | +8.1          | 1800ms         |

### Figures

**Figure 1: Privacy-Utility Tradeoff**
- X-axis: TRI rank drop (privacy)
- Y-axis: Classification F1 (utility)
- Points: Each method at different ε/k values
- Pareto frontier highlighted

**Figure 2: Epsilon Sensitivity**
- X-axis: ε
- Y-axis: Semantic similarity, classification F1
- Lines: One per method
- Shows diminishing returns

**Figure 3: Computational Cost**
- Bar chart: Inference time per method
- Grouped by dataset (TAB, Trustpilot)
- Log scale for y-axis

### Qualitative Examples

**Table 2: Anonymization Examples**
| Original                                    | Presidio                          | DPMLM (ε=1)                       | DPPrompt (ε=1)                    |
|---------------------------------------------|-----------------------------------|-----------------------------------|-----------------------------------|
| John Doe was born on March 15, 1985 in NYC | [PERSON] was born on [DATE] in [LOC] | James Smith was born around spring 1984 near NYC | A person was born in the mid-1980s in New York |

Demonstrates qualitative differences in anonymization strategies.
