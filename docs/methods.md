# Methods

## Taxonomy

Privacy-preserving text anonymization methods are categorized by their privacy guarantees and operational mechanisms:

```
Methods
├── Simple (No Formal Guarantees)
│   ├── Rule-based Redaction
│   └── NER-based Replacement
├── Differential Privacy (ε-DP)
│   ├── Token-level Perturbation
│   └── Sequence-level Generation
└── K-Anonymity (Indistinguishability)
    └── Risk-guided Generalization
```

## Simple Methods

These baselines provide **no formal privacy guarantees** but serve as reference points for utility and computational cost.

### Presidio

**Type**: Rule-based PII detection

**Mechanism**:
1. Pattern matching with regex for common PII types (email, phone, SSN)
2. NER using SpaCy for contextual entities (names, locations)
3. Redaction or replacement with placeholders

**Configuration** (`configs/model/presidio.yaml`):
```yaml
entities: [PERSON, LOCATION, ORGANIZATION, DATE, EMAIL, PHONE]
replacement_strategy: placeholder  # or mask, hash
```

**Implementation**: `methods/simple/_presidio.py`

**Strengths**:
- Fast execution (no model inference)
- Interpretable rules
- No training data required

**Limitations**:
- High false positive rate (e.g., "Amazon" as ORG vs. company name in context)
- Brittle to formatting variations
- Language and domain specific

**Use Case**: Quick baseline, production systems with strict latency requirements

### SpaCy

**Type**: NER-based entity replacement

**Mechanism**:
1. Run pretrained SpaCy NER model (`en_core_web_sm`)
2. Extract entities matching specified types
3. Replace with entity type placeholders: `[PERSON]`, `[ORG]`, etc.

**Configuration** (`configs/model/spacy.yaml`):
```yaml
model: en_core_web_sm
entities: [PERSON, ORG, GPE, LOC, DATE]
replacement: type_placeholder
```

**Implementation**: `methods/simple/_spacy.py`

**Strengths**:
- Better contextual understanding than regex
- Handles unseen entity values
- Pre-trained on large corpora

**Limitations**:
- Limited entity types (no email, phone)
- Lower recall on domain-specific entities (legal, medical)
- Redaction reduces utility significantly

**Use Case**: General-purpose baseline when training data unavailable

### Manual

**Type**: Ground-truth annotation based

**Mechanism**:
1. Load gold-standard annotations from dataset
2. Replace annotated spans with type placeholders
3. Handle overlapping spans via deduplication

**Configuration** (`configs/model/manual.yaml`):
```yaml
data: tab
data_in: data/TAB/splitted/train.json
replacement: type_placeholder
```

**Implementation**: `methods/simple/_manual.py`

**Strengths**:
- Perfect precision on annotated entities
- Upper bound for PII detection methods
- Useful for oracle experiments

**Limitations**:
- Requires expensive human annotation
- Not applicable to new texts
- Still loses utility from redaction

**Use Case**: Upper bound analysis, debugging other methods

### Baroud

**Type**: Placeholder for additional baseline

**Status**: To be implemented

**Planned**: Additional rule-based or heuristic method

## Differential Privacy Methods

These methods provide **ε-differential privacy guarantees**, ensuring that the presence or absence of any individual in the dataset minimally affects the output distribution.

### Formal Definition

A mechanism M satisfies ε-DP if for all neighboring datasets D, D' differing in one record, and all outputs S:

$$P(M(D) \in S) \leq e^\varepsilon \cdot P(M(D') \in S)$$

Lower ε = stronger privacy, but typically lower utility.

### DPMLM (DP Masked Language Model)

**Type**: Token-level perturbation via exponential mechanism

**Mechanism**:
1. Mask each sensitive token (identified by PII detector or selector)
2. Query pretrained MLM (BERT) for replacement distribution
3. Sample replacement using exponential mechanism with privacy budget ε
4. Utility function: MLM log-probability of replacement

**Privacy Guarantee**: Each token consumes ε/n budget (n = number of tokens), total ε via parallel composition

**Configuration** (`configs/model/dpmlm.yaml`):
```yaml
mlm_model: bert-base-uncased
temperature: 1.0              # Controls sampling sharpness
pii_detector:
  model_path: models/pii_detectors/tab/20231025_143022
  chunking:
    max_length: 512
    strategy: truncate
filtering_strategy: pii_only  # Only perturb PII tokens
```

**Implementation**: `methods/dp/_dpmlm.py`

**Exponential Mechanism**:
```python
def sample_token(logits, epsilon):
    scores = logits / sensitivity  # sensitivity = max logit range
    probabilities = softmax(epsilon * scores)
    return categorical_sample(probabilities)
```

**Strengths**:
- Formal DP guarantee per token
- Preserves local context (surrounding tokens unchanged)
- Efficient inference (single MLM forward pass per token)

**Limitations**:
- Grammaticality issues from independent token replacements
- Privacy budget splits across many tokens (weaker per-token privacy)
- Requires PII detector or risks exposing non-PII

**Use Case**: When local edits acceptable, ε budget is limited, and PII locations known

### DPPrompt

**Type**: Conditional generation with DP noise injection

**Mechanism**:
1. Create prompt: "Rewrite the following text privately: {text}"
2. Use causal LM (GPT-2) to generate anonymized version
3. Add DP noise to model logits during decoding
4. Sample from noised distribution at each generation step

**Privacy Guarantee**: Total ε consumed across all decoding steps via sequential composition

**Configuration** (`configs/model/dpprompt.yaml`):
```yaml
generator_model: gpt2
prompt_template: "Rewrite privately: "
max_new_tokens: 256
temperature: 1.0
noise_mechanism: gaussian  # or laplace
```

**Implementation**: `methods/dp/_dpprompt.py`

**Strengths**:
- Generates fluent, coherent text
- Can perform semantic transformations beyond replacement
- No PII detector required (model learns implicitly)

**Limitations**:
- High privacy cost (ε consumed at every token)
- Computationally expensive (autoregressive generation)
- May hallucinate or drop information

**Use Case**: When fluency is critical, ε budget is large, and generation quality outweighs cost

### DPParaphrase

**Type**: Encoder-decoder paraphrasing with DP training

**Mechanism**:
1. Train encoder-decoder model (T5, BART) on paraphrase pairs with DP-SGD
2. At inference: encode input text, decode paraphrased version
3. Privacy budget consumed during training, not inference

**Privacy Guarantee**: ε-DP for training dataset (protects training examples, not inference inputs)

**Configuration** (`configs/model/dpparaphrase.yaml`):
```yaml
paraphrase_model: models/gpt2-paraphraser
model_type: seq2seq
max_length: 256
num_beams: 4
```

**Implementation**: `methods/dp/_dpparaphrase.py`

**DP-SGD Training**:
- Clip gradients per example: `g_i = g_i / max(1, ||g_i|| / C)`
- Add Gaussian noise: `g_avg = (Σ g_i) / batch_size + N(0, σ^2 C^2 / batch_size^2)`
- Privacy analysis via Renyi DP or moments accountant

**Strengths**:
- One-time privacy cost during training
- Fast inference (standard seq2seq forward pass)
- Semantic preservation through paraphrase objective

**Limitations**:
- Requires paraphrase training data (expensive to obtain)
- Privacy applies to training set, not inference inputs (different threat model)
- May not anonymize PII if not seen during training

**Use Case**: When training data privacy is concern, inference speed matters, and paraphrase data available

### DPBART

**Type**: Sequence-to-sequence rewriting with DP training

**Mechanism**:
1. Fine-tune BART on text-to-anonymized-text pairs using DP-SGD
2. At inference: standard BART generation from input to output
3. Privacy budget consumed during fine-tuning

**Privacy Guarantee**: Same as DPParaphrase (training set privacy)

**Configuration** (`configs/model/dpbart.yaml`):
```yaml
bart_model: facebook/bart-base
max_length: 512
dp_training:
  epsilon: 8.0
  delta: 1e-5
  max_grad_norm: 1.0
```

**Implementation**: `methods/dp/_dpbart.py`

**Strengths**:
- Powerful pretrained model (BART)
- Handles long contexts better than GPT-2
- Can learn complex anonymization patterns

**Limitations**:
- Requires paired training data (original + anonymized)
- Expensive fine-tuning with DP-SGD
- Privacy for training set, not inference

**Use Case**: When paired anonymization data exists, training resources available, and inference privacy not primary concern

## K-Anonymity Methods

These methods ensure **k-anonymity**: each record is indistinguishable from at least k-1 other records in the dataset based on quasi-identifiers.

### PETRE (Privacy Enhancement Using Text Re-Identification)

**Type**: Risk-guided token generalization

**Mechanism**:
1. Train TRI model to predict author from text (re-identification attack)
2. For each token, compute importance score: change in TRI confidence when token masked
3. Iteratively mask tokens in descending importance order
4. Stop when TRI cannot re-identify author within top-k predictions

**K-Anonymity Guarantee**: Output text maps to k or more individuals with similar probability

**Configuration** (`configs/model/petre.yaml`):
```yaml
tri_model_path: models/tri_pipelines/tab/20231025_143022
explainer: greedy  # or shap, uniform
chunking:
  max_length: 512
  strategy: truncate
```

**Implementation**: `methods/k_anon/_petre.py`

**Risk Scoring**:
```python
def token_risk_score(text, token_idx, tri_model):
    original_prob = tri_model.predict(text)[true_author]
    masked_text = mask_token(text, token_idx)
    masked_prob = tri_model.predict(masked_text)[true_author]
    return original_prob - masked_prob  # Drop in confidence
```

**Strengths**:
- Empirically driven (removes what TRI uses for re-identification)
- Adaptive to dataset characteristics
- Provides interpretability (which tokens are risky)

**Limitations**:
- Requires training TRI model (needs author labels)
- Iterative masking is slow (O(n) TRI inferences per text)
- No formal privacy guarantee (heuristic k-anonymity)

**Use Case**: When re-identification is threat model, dataset has author metadata, and interpretability valued

## Method Comparison

### Privacy Guarantees

| Method      | Guarantee Type | Strength           | Inference Privacy |
|-------------|----------------|--------------------|-------------------|
| Presidio    | None           | N/A                | No                |
| SpaCy       | None           | N/A                | No                |
| DPMLM       | ε-DP           | Per-token ε/n      | Yes               |
| DPPrompt    | ε-DP           | Total ε            | Yes               |
| DPParaphrase| ε-DP           | Training set only  | No                |
| DPBART      | ε-DP           | Training set only  | No                |
| PETRE       | k-Anonymity    | Empirical          | Depends on TRI    |

### Computational Cost

| Method      | Training       | Inference Time | GPU Memory |
|-------------|----------------|----------------|------------|
| Presidio    | None           | ~10ms/doc      | None       |
| SpaCy       | None           | ~50ms/doc      | None       |
| DPMLM       | None           | ~100ms/doc     | 2GB        |
| DPPrompt    | None           | ~500ms/doc     | 4GB        |
| DPParaphrase| DP-SGD (days)  | ~200ms/doc     | 4GB        |
| DPBART      | DP-SGD (days)  | ~300ms/doc     | 6GB        |
| PETRE       | TRI (hours)    | ~2s/doc        | 2GB        |

### Utility Preservation

*To be measured in experiments section*

Expected ranking (high to low utility):
1. DPParaphrase, DPBART (semantic preservation)
2. DPPrompt (fluent generation)
3. DPMLM (local perturbation)
4. PETRE (selective masking)
5. SpaCy, Presidio (redaction)
6. Manual (complete redaction)

## Implementation Details

### Filtering Strategies

**Purpose**: Determine which tokens to anonymize

**Implementations**:
- `AllSelector`: Process all tokens (for methods without PII detector)
- `PIIOnlySelector`: Only perturb tokens predicted as PII by detector

**Integration**: Only DPMlmAnonymizer implements `set_filtering_strategy()` method

### Scoring Strategies

**Purpose**: Allocate privacy budget or masking priority

**Implementations**:
- `UniformExplainer`: Equal importance for all tokens
- `GreedyExplainer`: TRI-based importance (re-identification risk)
- `ShapExplainer`: Shapley values for model predictions

**Integration**: DPMlmAnonymizer and PetreAnonymizer have `set_scoring_strategy()` method

### Chunking Strategies

**Purpose**: Handle texts exceeding model context limits

**Implementations**:
- `TruncateChunker`: Keep first N tokens only
- `SlidingWindowChunker`: Overlapping windows, merge predictions

**Configuration**: Specified per component (PII detector, TRI model, main model)

### Registry Pattern

All methods register capabilities:

```python
register_anonymizer(
    name="dpmlm",
    anonymizer_class=DPMlmAnonymizer,
    capabilities=MethodCapabilities(
        requires_epsilon=True,
        requires_k=False,
        must_use_dataset=False,
        can_use_filtering=True,
        can_use_scoring=True,
    )
)
```

This enables static validation and routing in `model.py`.

## Extending with New Methods

To add a new anonymization method:

1. **Implement Anonymizer** (inherit from SimpleAnonymizer, DPAnonymizer, or KAnonymizer):
```python
from dp.methods.dp import DPAnonymizer

class MyMethodAnonymizer(DPAnonymizer):
    def grid_anonymize(self, text: str, epsilon: List[float], **kwargs):
        results = []
        for eps in epsilon:
            anonymized_text = ...  # Your implementation
            results.append(AnonymizationResult(
                text=anonymized_text,
                metadata={"epsilon": eps}
            ))
        return results
```

2. **Register in `methods/registry.py`**:
```python
from dp.methods.dp._mymethod import MyMethodAnonymizer

DP_MODEL_REGISTRY["mymethod"] = MyMethodAnonymizer
MODEL_REGISTRY["mymethod"] = MyMethodAnonymizer
```

3. **Add Capabilities in `methods/constants.py`**:
```python
MODEL_CAPABILITIES["mymethod"] = ModelCapabilities(
    requires_epsilon=True,
    can_use_filtering=False,
)
```

4. **Create Config**: `configs/model/mymethod.yaml`

5. **Add Tests**: Verify anonymization and privacy guarantees
