# API Reference

## Core Interfaces

### Anonymizer

Base class for all anonymization methods.

```python
from dp.methods.anonymizer import Anonymizer, AnonymizationResult
from abc import ABC, abstractmethod

class Anonymizer(ABC):
    def __init__(self, *args, **kwargs):
        """
        Initialize anonymizer.
        
        Args:
            *args: Positional arguments
            **kwargs: Keyword arguments (stored for builder)
        """
        self._init_args = args
        self._init_kwargs = kwargs
        self._model_name = kwargs.get('model', None)
    
    @abstractmethod
    def anonymize(self, text: str, *args, **kwargs) -> AnonymizationResult:
        """
        Anonymize a single text.
        
        Args:
            text: Input text to anonymize
            **kwargs: Method-specific parameters (epsilon, k, etc.)
            
        Returns:
            AnonymizationResult with anonymized text and metadata
        """
        raise NotImplementedError()
    
    @abstractmethod
    def anonymize_from_dataset(self, idx: int, *args, **kwargs) -> AnonymizationResult:
        """
        Anonymize a text from the loaded dataset by index.
        
        Args:
            idx: Index of the text in the dataset
            **kwargs: Method-specific parameters
            
        Returns:
            AnonymizationResult with anonymized text and metadata
        """
        raise NotImplementedError()
    
    def builder(self) -> AnonymizationBuilder:
        """
        Get a builder for batch anonymization.
        
        Returns:
            AnonymizationBuilder instance
        """
        from dp.methods.builder import AnonymizationBuilder
        return AnonymizationBuilder(self, self._model_name)
```

### AnonymizationResult

Structured output from anonymization.

```python
from dataclasses import dataclass
from typing import Optional, List, Dict, Any

@dataclass
class AnonymizationResult:
    """
    Result of anonymization operation.
    
    Attributes:
        text: Anonymized text
        spans: Modified entity spans (optional)
        metadata: Method-specific information
    """
    text: str
    spans: Optional[List[TextAnnotation]] = None
    metadata: Optional[Dict[str, Any]] = None
```

### AnonymizationBuilder

Builder for batch anonymization with parameter grids.

```python
class AnonymizationBuilder:
    def __init__(self, anonymizer: Anonymizer):
        """
        Initialize builder with an anonymizer.
        
        Args:
            anonymizer: Anonymizer instance
        """
        self.anonymizer = anonymizer
        self.texts = None
        self.indices = None
        self.epsilons = []
        self.ks = []
    
    def with_texts(self, texts: List[str]) -> Self:
        """
        Set texts to anonymize.
        
        Args:
            texts: List of text strings
            
        Returns:
            Self for chaining
        """
        self.texts = texts
        return self
    
    def with_indices(self, indices: List[int]) -> Self:
        """
        Set dataset indices to anonymize.
        
        Args:
            indices: List of integer indices
            
        Returns:
            Self for chaining
        """
        self.indices = indices
        return self
    
    def with_epsilons(self, epsilons: List[float]) -> Self:
        """
        Set privacy budgets for DP methods.
        
        Args:
            epsilons: List of epsilon values
            
        Returns:
            Self for chaining
        """
        self.epsilons = epsilons
        return self
    
    def with_ks(self, ks: List[int]) -> Self:
        """
        Set k values for k-anonymity methods.
        
        Args:
            ks: List of k values
            
        Returns:
            Self for chaining
        """
        self.ks = ks
        return self
    
    def anonymize(self, **kwargs) -> List[List[AnonymizationResult]]:
        """
        Execute anonymization for all combinations.
        
        Args:
            **kwargs: Additional runtime parameters
            
        Returns:
            Nested list: results[text_idx][param_idx]
        """
        pass
```

### DatasetRecord

Unified data structure across datasets.

```python
from dataclasses import dataclass

@dataclass
class DatasetRecord:
    """
    Unified representation of a dataset record.
    
    Attributes:
        text: Original text content
        uid: Unique identifier
        name: Entity name (for TRI, typically author)
        spans: List of annotated entity spans
        utilities: Task-specific labels (sentiment, category, etc.)
        metadata: Additional information
    """
    text: str
    uid: str
    name: str
    spans: Optional[List[TextAnnotation]] = None
    utilities: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None
```

### TextAnnotation

Character-level span annotation.

```python
@dataclass
class TextAnnotation:
    """
    Annotation for a text span.
    
    Attributes:
        start: Character offset (inclusive)
        end: Character offset (exclusive)
        label: Entity type (PERSON, ORG, LOC, etc.)
        text: Text content of the span
    """
    start: int
    end: int
    label: str
    text: str
```

## Dataset Adapters

### DatasetAdapter

Base class for dataset loaders.

```python
from abc import ABC, abstractmethod
from typing import Iterable

class DatasetAdapter(ABC):
    @abstractmethod
    def iter_records(self) -> Iterable[DatasetRecord]:
        """
        Iterate over dataset records.
        
        Yields:
            DatasetRecord instances
        """
        pass
    
    @abstractmethod
    def __len__(self) -> int:
        """
        Get total number of records.
        
        Returns:
            Number of records
        """
        pass
```

### get_adapter

Factory function for dataset adapters.

```python
def get_adapter(name: str, **kwargs) -> DatasetAdapter:
    """
    Get dataset adapter by name.
    
    Args:
        name: Dataset name (tab, trustpilot, db_bio)
        **kwargs: Adapter-specific arguments (data_in, max_records, etc.)
        
    Returns:
        DatasetAdapter instance
        
    Example:
        >>> adapter = get_adapter("tab", data_in="data/TAB/splitted/train.json")
        >>> for record in adapter.iter_records():
        ...     print(record.text)
    """
    pass
```

## PII Detection

### PIIDetector

Token classification model for PII detection.

```python
class PIIDetector:
    def __init__(
        self,
        model_name: str = "roberta-base",
        labels: Optional[List[str]] = None,
        max_length: int = 512,
        device: str = "auto",
        use_chunking: bool = False
    ):
        """
        Initialize PII detector.
        
        Args:
            model_name: HuggingFace model name or path to checkpoint
            labels: List of entity labels (auto-detected if None)
            max_length: Maximum sequence length
            device: Device for inference (cuda, mps, cpu, or auto)
            use_chunking: Enable chunking for long texts
        """
        pass
    
    def train(
        self,
        train_records: List[DatasetRecord],
        val_records: List[DatasetRecord],
        epochs: int = 5,
        batch_size: int = 8,
        use_nervaluate: bool = True,
        evaluation_mode: str = "partial",
        metric_mode: str = "recall"
    ):
        """
        Train PII detector on annotated data.
        
        Args:
            train_records: Training records with spans
            val_records: Validation records with spans
            epochs: Number of training epochs
            batch_size: Batch size for training
            use_nervaluate: Use nervaluate metrics (recommended)
            evaluation_mode: Evaluation mode (strict, partial, exact)
            metric_mode: Metric for best model selection (precision, recall, f1)
        """
        pass
    
    def predict(
        self,
        records: List[DatasetRecord]
    ) -> List[DatasetRecord]:
        """
        Predict PII spans for records.
        
        Args:
            records: Records with text (spans will be predicted)
            
        Returns:
            Records with predicted spans
        """
        pass
    
    def evaluate(
        self,
        records: List[DatasetRecord],
        use_nervaluate: bool = True,
        evaluation_mode: str = "partial"
    ) -> Dict[str, Any]:
        """
        Evaluate PII detector on labeled data.
        
        Args:
            records: Records with gold spans
            use_nervaluate: Use nervaluate metrics
            evaluation_mode: Evaluation mode
            
        Returns:
            Dictionary with precision, recall, F1 per entity type and overall
        """
        pass
```

## TRI Detection

### TRIDetector

Re-identification attack model.

```python
class TRIDetector:
    def __init__(
        self,
        dataset_name: Optional[str] = None,
        model_name: str = "distilbert-base-uncased",
        max_length: int = 512,
        device: str = "auto",
        use_chunking: bool = False
    ):
        """
        Initialize TRI detector.
        
        Args:
            dataset_name: Dataset name (for saving models)
            model_name: HuggingFace model name or path
            max_length: Maximum sequence length
            device: Device for inference
            use_chunking: Enable chunking for long texts
        """
        pass
    
    def train(
        self,
        train_records: List[DatasetRecord],
        val_records: List[DatasetRecord],
        pretraining_epochs: int = 0,
        finetuning_epochs: int = 10,
        batch_size: int = 8,
        pretraining_batch_size: int = 8,
        use_pretraining: bool = False,
        annotation_datasets: Optional[Dict[str, List[DatasetRecord]]] = None,
        best_metric_dataset: Optional[str] = None
    ):
        """
        Train TRI model for authorship attribution.
        
        Args:
            train_records: Training records with name field
            val_records: Validation records with name field
            pretraining_epochs: MLM pretraining epochs (optional)
            finetuning_epochs: Classification fine-tuning epochs
            batch_size: Batch size for finetuning
            pretraining_batch_size: Batch size for pretraining
            use_pretraining: Enable MLM pretraining phase
            annotation_datasets: Dict of anonymized datasets for evaluation
            best_metric_dataset: Which dataset to use for best model selection
        """
        pass
    
    def predict(
        self,
        record: DatasetRecord
    ) -> Dict[str, float]:
        """
        Predict author probabilities for a text.
        
        Args:
            record: Record with text
            
        Returns:
            Dictionary mapping author names to probabilities
        """
        pass
    
    def evaluate(
        self,
        records: List[DatasetRecord]
    ) -> Dict[str, Any]:
        """
        Evaluate TRI model on labeled data.
        
        Args:
            records: Records with name field
            
        Returns:
            Dictionary with top-k accuracy, MRR, rank statistics
        """
        pass
```

## Strategies

### Selector

Base class for token selection strategies (in `utils/selector/base.py`).

```python
from abc import ABC, abstractmethod

class Selector(ABC):
    @abstractmethod
    def select(
        self,
        text: str,
        tokens: List[str]
    ) -> np.ndarray:
        """
        Select which tokens to anonymize.
        
        Args:
            text: Original text
            tokens: Tokenized text
            
        Returns:
            Boolean mask (True = anonymize this token)
        """
        pass
```

**Implementations** (in `utils/selector/`):
- `AllSelector`: Select all tokens
- `PIIOnlySelector`: Select only tokens predicted as PII by detector

**Note**: Only `DPMlmAnonymizer` has `set_filtering_strategy()` method to use selectors.

### Explainer

Base class for token importance scoring (in `utils/explainer/base.py`).

```python
from abc import ABC, abstractmethod

class Explainer(ABC):
    @abstractmethod
    def explain(
        self,
        text: str,
        tokens: List[str]
    ) -> np.ndarray:
        """
        Score token importance for privacy.
        
        Args:
            text: Original text
            tokens: Tokenized text
            
        Returns:
            Importance scores (higher = more important to anonymize)
        """
        pass
```

**Implementations** (in `utils/explainer/`):
- `UniformExplainer`: Equal importance for all tokens
- `GreedyExplainer`: TRI-based importance (drop in confidence when masked)
- `ShapExplainer`: Shapley values for model predictions

**Note**: Only `DPMlmAnonymizer` and `PetreAnonymizer` have `set_scoring_strategy()` method to use explainers.

## Method Registry

### Model Registry

Methods are registered in `methods/registry.py` through dictionaries:

```python
from dp.methods.registry import MODEL_REGISTRY, SIMPLE_MODEL_REGISTRY, DP_MODEL_REGISTRY, K_ANON_MODEL_REGISTRY

MODEL_REGISTRY: Dict[str, Type[Anonymizer]] = {
    "spacy": SpacyAnonymizer,
    "presidio": PresidioAnonymizer,
    "manual": ManualAnonymizer,
    "baroud": BaroudAnonymizer,
    "petre": PetreAnonymizer,
    "dpbart": DPBartAnonymizer,
    "dpparaphrase": DPParaphraseAnonymizer,
    "dpprompt": DPPromptAnonymizer,
    "dpmlm": DPMlmAnonymizer,
}
```

To add a new method, add it to the appropriate registry dict.

### ModelCapabilities

Declare method requirements and features in `methods/constants.py`.

```python
from dataclasses import dataclass

@dataclass
class ModelCapabilities:
    """
    Capabilities and requirements of an anonymization method.
    
    Attributes:
        must_use_dataset: Method needs full dataset (e.g., Manual, PETRE)
        requires_epsilon: Method uses epsilon parameter (DP methods)
        requires_k: Method uses k parameter (k-anonymity)
        must_use_non_uniform_explainer: Method requires explainer (PETRE)
        can_use_annotations: Method can leverage external annotations
        can_use_scoring: Method supports importance scoring
        can_use_filtering: Method supports token filtering
        supports_batch_predict: Method supports batch prediction
    """
    must_use_dataset: bool = False
    requires_epsilon: bool = False
    requires_k: bool = False
    must_use_non_uniform_explainer: bool = False
    can_use_annotations: bool = False
    can_use_scoring: bool = False
    can_use_filtering: bool = False
    supports_batch_predict: bool = False

MODEL_CAPABILITIES: Dict[str, ModelCapabilities] = {
    "spacy": ModelCapabilities(),
    "presidio": ModelCapabilities(),
    "manual": ModelCapabilities(must_use_dataset=True),
    "baroud": ModelCapabilities(supports_batch_predict=True),
    "petre": ModelCapabilities(
        must_use_dataset=True,
        requires_k=True,
        must_use_non_uniform_explainer=True,
        can_use_annotations=True,
        can_use_scoring=True,
    ),
    "dpmlm": ModelCapabilities(
        requires_epsilon=True,
        can_use_filtering=True,
        can_use_scoring=True,
    ),
    "dpbart": ModelCapabilities(requires_epsilon=True),
    "dpparaphrase": ModelCapabilities(requires_epsilon=True),
    "dpprompt": ModelCapabilities(requires_epsilon=True),
}

def get_capabilities(model_name: str) -> ModelCapabilities:
    """Get capabilities for a model."""
    if model_name not in MODEL_CAPABILITIES:
        raise ValueError(f"Unknown model: {model_name}")
    return MODEL_CAPABILITIES[model_name]
```

## CLI Tools

### model.py

Main anonymization script.

```bash
python3 model.py \
    --data tab \
    --data_in data/TAB/splitted/test.json \
    --model dpmlm \
    --model_in configs/model/dpmlm.yaml \
    --runtime_in configs/runtime/dp.yaml \
    --max_records 10 \
    --output_path outputs/tab/dpmlm
```

**Arguments**:
- `--data`: Dataset name (tab, trustpilot, db_bio)
- `--data_in`: Path to data file
- `--model`: Method name (dpmlm, presidio, petre, etc.)
- `--model_in`: Path to model config YAML
- `--runtime_in`: Path to runtime config YAML
- `--max_records`: Limit number of texts
- `--output_path`: Output directory for results

### pii.py

PII detector training and evaluation.

```bash
# Train
python3 pii.py \
    --dataset tab \
    --data-path data/TAB/splitted \
    --mode train \
    --epochs 5 \
    --batch-size 8 \
    --use-nervaluate \
    --evaluation-mode partial

# Evaluate
python3 pii.py \
    --dataset tab \
    --data-path data/TAB/splitted \
    --mode evaluate \
    --model-path models/pii_detectors/tab/20231025_143022 \
    --use-nervaluate

# Predict
python3 pii.py \
    --dataset tab \
    --data-path data/TAB/splitted \
    --mode predict \
    --model-path models/pii_detectors/tab/20231025_143022
```

**Arguments**:
- `--dataset`: Dataset name
- `--data-path`: Path to train/val/test splits
- `--mode`: Operation mode (train, evaluate, predict)
- `--model-path`: Path to model checkpoint (for evaluate/predict)
- `--epochs`: Training epochs
- `--batch-size`: Batch size
- `--use-nervaluate`: Enable nervaluate metrics
- `--evaluation-mode`: Nervaluate mode (strict, partial, exact)

### tri.py

TRI model training and evaluation.

```bash
# Train
python3 tri.py \
    --dataset tab \
    --data-path data/TAB/splitted \
    --mode train \
    --use-pretraining \
    --pretraining-epochs 3 \
    --finetuning-epochs 10 \
    --annotation-folder outputs/tab/presidio \
    --best-metric-dataset presidio

# Evaluate
python3 tri.py \
    --dataset tab \
    --data-path data/TAB/splitted \
    --mode evaluate \
    --model-path models/tri_pipelines/tab/20231025_143022 \
    --annotation-folder outputs/tab/presidio
```

**Arguments**:
- `--dataset`: Dataset name
- `--data-path`: Path to data
- `--mode`: Operation mode (train, evaluate, predict)
- `--model-path`: Path to checkpoint
- `--use-pretraining`: Enable MLM pretraining
- `--pretraining-epochs`: MLM epochs
- `--finetuning-epochs`: Classification epochs
- `--annotation-folder`: Path to anonymized texts for TRI training
- `--best-metric-dataset`: Which annotation source to prioritize for model selection

### data.py

Dataset preprocessing.

```bash
python3 data.py \
    --data tab \
    --data_in data/TAB/tab.json \
    --max_records 1000
```

**Arguments**:
- `--data`: Dataset name (required)
- `--data_in`: Input data path (required)
- `--max_records`: Limit dataset size (optional)

## Configuration Files

### Model Config

Example: `configs/model/dpmlm.yaml`

```yaml
mlm_model: bert-base-uncased
temperature: 1.0
pii_detector:
  model_path: models/pii_detectors/tab/20231025_143022
  chunking:
    max_length: 512
    strategy: truncate
filtering_strategy: pii_only
```

### Runtime Config

Example: `configs/runtime/dp.yaml`

```yaml
epsilons: [0.1, 1.0, 5.0, 10.0]
batch_size: 16
device: auto
```

## Type Hints

All functions use type hints for clarity:

```python
from typing import List, Dict, Optional, Any, Iterable

def process_records(
    records: List[DatasetRecord],
    model: Anonymizer,
    epsilon: Optional[float] = None
) -> List[AnonymizationResult]:
    """Process records with anonymization."""
    pass
```

## Error Handling

**Fail Fast**: Errors are raised immediately, no silent failures

**Type Validation**: Runtime checks for config types

**Device Resolution**: Automatic fallback (CUDA → MPS → CPU)

```python
def get_device(device: str = "auto") -> str:
    if device == "auto":
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return device
```
