# Architecture

## System Design

### Core Abstractions

#### Anonymizer

Base class for all privacy-preserving text transformations:

```python
class Anonymizer(ABC):
    @abstractmethod
    def anonymize(self, text: str, *args, **kwargs) -> AnonymizationResult:
        pass
    
    @abstractmethod
    def anonymize_from_dataset(self, idx: int, *args, **kwargs) -> AnonymizationResult:
        pass
    
    def builder(self) -> AnonymizationBuilder:
        return AnonymizationBuilder(self)
```

**Design Rationale**: 
- Single responsibility: Transform text with privacy guarantees
- Two paths: Direct text input vs. dataset index (for methods requiring context)
- Builder access: Enables complex multi-parameter execution flows

#### AnonymizationResult

Structured output from anonymization:

```python
@dataclass
class AnonymizationResult:
    text: str                           # Anonymized text
    spans: Optional[List] = None        # Modified entity spans
    metadata: Optional[dict] = None     # Method-specific info (ε, k, perturbed tokens)
```

#### DatasetRecord

Unified representation across datasets:

```python
@dataclass
class DatasetRecord:
    text: str                           # Original text
    uid: str                            # Unique identifier
    name: str                           # Entity name (for TRI)
    spans: Optional[List[TextAnnotation]] = None
    utilities: Optional[dict] = None    # Downstream task labels
    metadata: Optional[dict] = None
```

### Method Hierarchy

```
Anonymizer (abstract base)
├── SimpleAnonymizer
│   ├── PresidioAnonymizer
│   ├── SpacyAnonymizer
│   ├── ManualAnonymizer
│   └── BaroudAnonymizer
├── DPAnonymizer
│   ├── DPMlmAnonymizer
│   ├── DPPromptAnonymizer
│   ├── DPParaphraseAnonymizer
│   └── DPBartAnonymizer
└── KAnonAnonymizer
    └── PetreAnonymizer
```

**Inheritance Strategy**:
- `SimpleAnonymizer`: No formal privacy guarantees, operates on single texts
- `DPAnonymizer`: ε-DP guarantees, supports grid search over privacy budgets
- `KAnonAnonymizer`: k-anonymity guarantees, requires dataset for indistinguishability sets

### Builder Pattern

Handles complex parameter combinations:

```python
class AnonymizationBuilder:
    def with_texts(self, texts: List[str]) -> Self
    def with_indices(self, indices: List[int]) -> Self
    def with_epsilons(self, epsilons: List[float]) -> Self
    def with_ks(self, ks: List[int]) -> Self
    def anonymize(self, **kwargs) -> List[AnonymizationResult]
```

**Execution Flow**:
1. Set input (texts or indices)
2. Set parameters (ε for DP, k for k-anon)
3. Call `anonymize()` with runtime config
4. Returns nested results: `[text][parameter_value] -> AnonymizationResult`

### Capability System

Methods declare capabilities via registry:

```python
@dataclass
class ModelCapabilities:
    must_use_dataset: bool = False              # Needs full dataset context
    requires_epsilon: bool = False              # DP method
    requires_k: bool = False                    # K-anonymity method
    must_use_non_uniform_explainer: bool = False  # Requires explainer (PETRE)
    can_use_annotations: bool = False           # Can leverage external annotations
    can_use_scoring: bool = False               # Supports importance scoring
    can_use_filtering: bool = False             # Supports PII-only processing
    supports_batch_predict: bool = False        # Supports batch prediction
```

**Routing Logic**: `model.py` inspects capabilities to determine execution path:
- Simple methods: Direct text-by-text processing
- DP methods: Grid search over ε values with texts
- K-anon methods: Dataset-aware processing with k values

### Component Integration

#### PII Detector

Token classification for entity recognition:

```python
class PIIDetector:
    def train(self, epochs, batch_size, use_nervaluate=True)
    def predict(self, records: List[DatasetRecord]) -> List[DatasetRecord]
    def evaluate(self, records, use_nervaluate=True) -> Dict[str, Any]
```

**Integration Points**:
- **Filtering Strategy**: DPMLM has `set_filtering_strategy()` to use PIIOnlySelector with detector
- **Annotation Source**: PETRE can leverage predicted spans for guided anonymization

#### TRI Detector

Re-identification attack model:

```python
class TRIDetector:
    def train(self, pretraining_epochs, finetuning_epochs)
    def predict(self, record: DatasetRecord) -> Dict[str, float]  # name -> probability
    def evaluate(self, records) -> Dict[str, Any]
```

**Integration Points**:
- **Scoring Strategy**: PETRE and DPMLM have `set_scoring_strategy()` to use GreedyExplainer with TRI
- **Privacy Evaluation**: Measures rank degradation after anonymization

#### Explainer

Importance scoring for privacy budget allocation:

```python
class Explainer(ABC):
    @abstractmethod
    def explain(self, text: str, tokens: List[str]) -> np.ndarray
```

**Implementations**:
- `UniformExplainer`: Equal privacy budget per token
- `GreedyExplainer`: TRI-based importance scores
- `ShapExplainer`: Shapley value attribution

### Data Pipeline

#### Loaders

Dataset-specific adapters:

```python
class DatasetAdapter(ABC):
    @abstractmethod
    def iter_records(self) -> Iterable[DatasetRecord]
    
    @abstractmethod
    def __len__(self) -> int
```

**Implementations**:
- `TabDatasetAdapter`: Legal documents with nested annotations
- `TrustpilotDatasetAdapter`: Review text with star ratings
- `DbBioDatasetAdapter`: Biomedical abstracts with HuggingFace format

#### Preprocessing

`data.py` provides:
- Train/validation/test splitting
- Annotation format conversion
- Metadata extraction
- Quality filtering

### Chunking Strategy

Handle texts exceeding model context limits:

```python
class Chunker(ABC):
    @abstractmethod
    def chunk(self, text: str) -> List[Chunk]
```

**Implementations**:
- `TruncateChunker`: Keep first N tokens
- `SlidingWindowChunker`: Overlapping windows with span aggregation
- `TokenAwareChunker`: Respects token boundaries

**Aggregation**: `SpanMergeAggregator` combines predictions from overlapping chunks

### Configuration Management

Three-level config hierarchy:

1. **Model Config** (`configs/model/*.yaml`):
   - Algorithm hyperparameters
   - Paths to pretrained components (PII detector, explainer)
   - Chunking settings per component

2. **Runtime Config** (`configs/runtime/*.yaml`):
   - Execution parameters (ε, k)
   - Batch sizes, device allocation

3. **CLI Arguments**:
   - Override config values
   - Specify input/output paths

**Loading Order**: CLI args > Runtime config > Model config > Defaults

### Output Handlers

Flexible result serialization:

```python
class OutputHandler(ABC):
    @abstractmethod
    def output(self, result: AnonymizationResult, idx: Optional[int], **kwargs)
```

**Implementations**:
- `PrintOutputHandler`: Console display
- `JsonlOutputHandler`: Timestamped JSONL files for batch processing

### Error Handling

**Fail Fast Principle**: No silent errors, explicit validation

**Device Resolution**: Automatic GPU selection (CUDA > MPS > CPU)

**Type Safety**: Runtime type validation for configuration values

## Execution Flow

### Single Text Anonymization

```
model.py --texts "..." --model dpmlm --runtime_in configs/runtime/dp.yaml
    ↓
Load model config from configs/model/dpmlm.yaml
    ↓
Initialize DPMlmAnonymizer with config
    ↓
Set filtering strategy (PIIOnlySelector with PII detector)
    ↓
Set scoring strategy (GreedyExplainer with TRI detector)
    ↓
Builder: with_texts([text]).with_epsilons([1.0, 5.0, 10.0])
    ↓
For each text, for each ε:
    - Detect PII spans (if filtering enabled)
    - Score token importance (if scoring enabled)
    - Apply DPMLM perturbation with budget allocation
    - Return AnonymizationResult
    ↓
Output results via handler
```

### Dataset Anonymization

```
model.py --data tab --data_in data.json --model petre --runtime_in configs/runtime/k_anon.yaml
    ↓
Load dataset via TabDatasetAdapter
    ↓
Load model config from configs/model/petre.yaml
    ↓
Initialize PetreAnonymizer with dataset_records
    ↓
Load annotations if specified (for TRI training context)
    ↓
Set scoring strategy (GreedyExplainer - required for PETRE)
    ↓
Builder: with_indices([0,1,2]).with_ks([2,5,10])
    ↓
For each index, for each k:
    - Compute TRI risk scores for all tokens
    - Iteratively mask highest-risk tokens until k-anon satisfied
    - Return AnonymizationResult
    ↓
Output results via handler
```

### Training Pipeline

**PII Detector**:
```
pii.py --mode train --dataset tab --epochs 5 --use-nervaluate
    ↓
Load train/validation/test splits
    ↓
Initialize PIIDetector with labels from training data
    ↓
Create PIIDataset (tokenize, align labels with BPE)
    ↓
Train with Trainer (nervaluate metrics for best model selection)
    ↓
Evaluate on test set with strict/partial/exact modes
    ↓
Save model checkpoint
```

**TRI Detector**:
```
tri.py --mode train --dataset tab --use-pretraining --annotation-folder outputs/
    ↓
Load dataset with UIDs and names
    ↓
Initialize TRIDetector
    ↓
Optional: MLM pretraining on unlabeled text
    ↓
Load annotations from multiple sources (Presidio, SpaCy, etc.)
    ↓
Create TRIDataset (text + author label)
    ↓
Fine-tune classifier on re-identification task
    ↓
Evaluate: measure ranking accuracy, privacy scores
    ↓
Save model checkpoint
```

## Design Decisions

### Why Builder Pattern?

Grid search over parameters (ε, k) requires N×M executions. Builder pattern:
- Separates parameter setting from execution
- Enables batch processing optimizations
- Provides clean API: `builder.with_epsilons([...]).anonymize()`

### Why Capability System?

Different methods have incompatible requirements:
- DPMLM: operates on texts, supports filtering/scoring
- PETRE: requires full dataset, mandatory explainer
- Presidio: simple text-to-text, no parameters

ModelCapabilities enable **static validation** before execution. Defined in `methods/constants.py`.

### Why Three Config Levels?

- **Model config**: Algorithm-specific, version-controlled
- **Runtime config**: Experiment-specific, shareable
- **CLI args**: One-off overrides, interactive use

This separation enables reproducible experiments while maintaining flexibility.

### Why Plug-and-Play Components?

Research requires rapid experimentation:
- Swap PII detectors without changing anonymization code
- Try different explainers for budget allocation
- Add new anonymization methods by implementing one interface

Loose coupling via strategy pattern enables this modularity.
