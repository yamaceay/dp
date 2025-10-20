# Examples

## Quick Start

### Simple Anonymization

Anonymize a single text with Presidio:

```bash
python3 model.py \
    --texts "John Doe visited New York on March 15, 2024." \
    --model presidio \
    --model_in configs/model/presidio.yaml
```

Output:
```
[PERSON] visited [LOCATION] on [DATE].
```

### Dataset Anonymization

Anonymize TAB dataset with SpaCy:

```bash
python3 model.py \
    --data tab \
    --data_in data/TAB/splitted/test.json \
    --model spacy \
    --model_in configs/model/spacy.yaml \
    --max_records 10
```

Results saved to timestamped JSONL file in current directory.

### Differential Privacy

Anonymize with DPMLM at different privacy levels:

```bash
python3 model.py \
    --data tab \
    --data_in data/TAB/splitted/test.json \
    --model dpmlm \
    --model_in configs/model/dpmlm.yaml \
    --runtime_in configs/runtime/dp.yaml \
    --max_records 10
```

Runtime config (`configs/runtime/dp.yaml`):
```yaml
epsilons: [0.1, 1.0, 5.0, 10.0]
```

Output contains results for each epsilon value.

## Training Pipelines

### Train PII Detector

Train on TAB dataset with nervaluate metrics:

```bash
python3 pii.py \
    --dataset tab \
    --data-path data/TAB/splitted \
    --mode train \
    --epochs 5 \
    --batch-size 8 \
    --use-nervaluate \
    --evaluation-mode partial \
    --metric-mode recall
```

Model saved to `models/pii_detectors/tab/<timestamp>/`.

### Evaluate PII Detector

Evaluate trained model on test set:

```bash
python3 pii.py \
    --dataset tab \
    --data-path data/TAB/splitted \
    --mode evaluate \
    --model-path models/pii_detectors/tab/20231025_143022 \
    --use-nervaluate
```

Output shows strict, partial, and exact F1 scores.

### Train TRI Model

Train re-identification model with pretraining:

```bash
python3 tri.py \
    --dataset tab \
    --data-path data/TAB/splitted \
    --mode train \
    --use-pretraining \
    --pretraining-epochs 3 \
    --finetuning-epochs 10 \
    --batch-size 8 \
    --annotation-folder outputs/tab/presidio \
    --best-metric-dataset presidio
```

- Pretrain with MLM on unlabeled text (3 epochs)
- Fine-tune classifier on authorship (10 epochs)
- Use Presidio-anonymized texts as negative examples
- Select best model based on Presidio metrics

### Evaluate TRI Model

Measure re-identification risk:

```bash
python3 tri.py \
    --dataset tab \
    --data-path data/TAB/splitted \
    --mode evaluate \
    --model-path models/tri_pipelines/tab/20231025_143022 \
    --annotation-folder outputs/tab
```

Output shows rank drop and confidence drop for each anonymization method in folder.

## Advanced Usage

### Using PII Detector with DPMLM

Configure DPMLM to use trained PII detector for filtering:

`configs/model/dpmlm.yaml`:
```yaml
mlm_model: bert-base-uncased
temperature: 1.0
pii_detector:
  model_path: models/pii_detectors/tab/20231025_143022
  chunking:
    max_length: 512
    strategy: truncate
filtering_strategy: pii_only  # Only perturb detected PII
```

Run:
```bash
python3 model.py \
    --data tab \
    --data_in data/TAB/splitted/test.json \
    --model dpmlm \
    --model_in configs/model/dpmlm.yaml \
    --runtime_in configs/runtime/dp.yaml
```

DPMLM will:
1. Detect PII spans with trained detector
2. Only perturb tokens within PII spans
3. Leave non-PII text unchanged

### Using TRI Model with PETRE

Configure PETRE to use TRI model for risk scoring:

`configs/model/petre.yaml`:
```yaml
tri_model_path: models/tri_pipelines/tab/20231025_143022
explainer: greedy  # TRI-based importance scoring
chunking:
  max_length: 512
  strategy: truncate
```

Run:
```bash
python3 model.py \
    --data tab \
    --data_in data/TAB/splitted/test.json \
    --model petre \
    --model_in configs/model/petre.yaml \
    --runtime_in configs/runtime/k_anon.yaml
```

PETRE will:
1. Compute TRI risk for each token (drop in confidence when masked)
2. Iteratively mask highest-risk tokens
3. Stop when k-anonymity achieved (author not in top-k predictions)

### Multi-Parameter Grid Search

Test multiple epsilon and k values:

`configs/runtime/grid.yaml`:
```yaml
epsilons: [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
ks: [2, 3, 5, 10, 20]
```

Run DPMLM:
```bash
python3 model.py \
    --data tab \
    --data_in data/TAB/splitted/test.json \
    --model dpmlm \
    --model_in configs/model/dpmlm.yaml \
    --runtime_in configs/runtime/grid.yaml
```

Output contains results for all 6 epsilon values per text.

Run PETRE:
```bash
python3 model.py \
    --data tab \
    --data_in data/TAB/splitted/test.json \
    --model petre \
    --model_in configs/model/petre.yaml \
    --runtime_in configs/runtime/grid.yaml
```

Output contains results for all 5 k values per text.

## Programmatic Usage

### Python API

```python
from dp.loaders import get_adapter
from dp.methods import get_anonymizer
from dp.methods.builder import AnonymizationBuilder

# Load dataset
adapter = get_adapter("tab", data_in="data/TAB/splitted/test.json", max_records=10)
records = list(adapter.iter_records())

# Load anonymizer
anonymizer = get_anonymizer(
    name="dpmlm",
    config_path="configs/model/dpmlm.yaml"
)

# Build and execute
builder = anonymizer.builder()
results = builder.with_texts([r.text for r in records]) \
                 .with_epsilons([1.0, 5.0, 10.0]) \
                 .anonymize()

# Process results
for text_idx, text_results in enumerate(results):
    print(f"Text {text_idx}:")
    for eps_idx, result in enumerate(text_results):
        epsilon = [1.0, 5.0, 10.0][eps_idx]
        print(f"  ε={epsilon}: {result.text}")
        print(f"    Metadata: {result.metadata}")
```

### Training PII Detector

```python
from dp.utils.pii_detector import PIIDetector
from dp.loaders import get_adapter

```bash
# Load data
train_adapter = get_adapter("tab", data_in="data/TAB/splitted/train.json")
val_adapter = get_adapter("tab", data_in="data/TAB/splitted/dev.json")
test_adapter = get_adapter("tab", data_in="data/TAB/splitted/test.json")

train_records = list(train_adapter.iter_records())
val_records = list(val_adapter.iter_records())
test_records = list(test_adapter.iter_records())

# Initialize detector
detector = PIIDetector(
    model_name="roberta-base",
    device="cuda"
)

# Train
detector.train(
    train_records=train_records,
    val_records=val_records,
    epochs=5,
    batch_size=8,
    use_nervaluate=True,
    evaluation_mode="partial"
)

# Evaluate
metrics = detector.evaluate(
    records=test_records,
    use_nervaluate=True
)
print(f"Test F1: {metrics['partial_f1']:.3f}")

# Predict
predicted = detector.predict(records=test_records)
for record in predicted:
    print(f"Text: {record.text[:100]}...")
    print(f"Spans: {record.spans}")
```

### Training TRI Model

```python
from dp.utils.tri_detector import TRIDetector
from dp.loaders import get_adapter

# Load data (separate files)
train_adapter = get_adapter("tab", data_in="data/TAB/splitted/train.json")
val_adapter = get_adapter("tab", data_in="data/TAB/splitted/dev.json")
train_records = list(train_adapter.iter_records())
val_records = list(val_adapter.iter_records())

# Initialize detector
tri = TRIDetector(
    dataset_name="tab",
    model_name="distilbert-base-uncased",
    device="cuda"
)

# Train with pretraining
tri.train(
    train_records=train_records,
    val_records=val_records,
    use_pretraining=True,
    pretraining_epochs=3,
    finetuning_epochs=10,
    batch_size=8
)

# Predict
test_record = val_records[0]
probabilities = tri.predict(test_record)

# Show top-5 predictions
sorted_preds = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
print(f"True author: {test_record.name}")
print("Top-5 predictions:")
for rank, (author, prob) in enumerate(sorted_preds[:5], 1):
    marker = "✓" if author == test_record.name else " "
    print(f"  {rank}. {author}: {prob:.3f} {marker}")
```

### Custom Anonymizer

```python
from dp.methods.anonymizer import Anonymizer, AnonymizationResult
from dp.methods.registry import register_anonymizer, MethodCapabilities

from dp.methods.simple import SimpleAnonymizer

class MyAnonymizer(SimpleAnonymizer):
    def __init__(self, replacement_text: str = "[REDACTED]", **kwargs):
        super().__init__(**kwargs)
        self.replacement_text = replacement_text
    
    def anonymize(self, text: str, **kwargs) -> AnonymizationResult:
        import re
        anonymized = re.sub(r'\b[A-Z][a-z]+\b', self.replacement_text, text)
        return AnonymizationResult(
            text=anonymized,
            metadata={"method": "my_anonymizer"}
        )
    
    def anonymize_from_dataset(self, idx: int, **kwargs) -> AnonymizationResult:
        raise NotImplementedError("This method doesn't use datasets")

# Register in registry.py
from dp.methods.registry import SIMPLE_MODEL_REGISTRY
SIMPLE_MODEL_REGISTRY["my_anonymizer"] = MyAnonymizer

# Also add to MODEL_REGISTRY
from dp.methods.registry import MODEL_REGISTRY
MODEL_REGISTRY["my_anonymizer"] = MyAnonymizer

# Add capabilities in constants.py
from dp.methods.constants import MODEL_CAPABILITIES, ModelCapabilities
MODEL_CAPABILITIES["my_anonymizer"] = ModelCapabilities()

# Use
anonymizer = MyAnonymizer(replacement_text="[NAME]", model="my_anonymizer")
result = anonymizer.anonymize("John Doe went to Paris on Monday.")
print(result.text)  # [NAME] [NAME] went to [NAME] on [NAME].
```

## Common Workflows

### Experiment: Privacy-Utility Tradeoff

```bash
# 1. Train PII detector
python3 pii.py --dataset tab --data-path data/TAB/splitted --mode train --epochs 5 --use-nervaluate

# 2. Anonymize with different epsilons
python3 model.py \
    --data tab --data_in data/TAB/splitted/test.json \
    --model dpmlm --model_in configs/model/dpmlm.yaml \
    --runtime_in configs/runtime/dp.yaml

# 3. Train TRI on original texts
python3 tri.py --dataset tab --data-path data/TAB/splitted --mode train --finetuning-epochs 10

# 4. Evaluate TRI on anonymized texts
python3 tri.py --dataset tab --data-path data/TAB/splitted --mode evaluate \
    --model-path models/tri_pipelines/tab/<timestamp> \
    --annotation-folder outputs/tab/dpmlm

# 5. Compute downstream task utility (manual: train classifier on anonymized, test on original)
```

### Experiment: Method Comparison

```bash
# Anonymize with all methods
for method in presidio spacy dpmlm dpprompt petre; do
    python3 model.py \
        --data tab --data_in data/TAB/splitted/test.json \
        --model $method \
        --model_in configs/model/$method.yaml \
        --output_path outputs/tab/$method
done

# Compare TRI metrics
python3 tri.py --dataset tab --data-path data/TAB/splitted --mode evaluate \
    --model-path models/tri_pipelines/tab/<timestamp> \
    --annotation-folder outputs/tab

# Aggregate results
python3 experiments/aggregate_results.py --input outputs/tab --output results.csv
```

### Cross-Dataset Evaluation

```bash
# Train PII detector on TAB
python3 pii.py --dataset tab --data-path data/TAB/splitted --mode train --epochs 5

# Evaluate on Trustpilot (zero-shot)
python3 pii.py --dataset trustpilot --data-path data/trustpilot/www.amazon.com \
    --mode evaluate --model-path models/pii_detectors/tab/<timestamp>

# Compare domain-specific vs. general detector
python3 pii.py --dataset trustpilot --data-path data/trustpilot/www.amazon.com \
    --mode train --epochs 5  # Train Trustpilot-specific

python3 pii.py --dataset trustpilot --data-path data/trustpilot/www.amazon.com \
    --mode evaluate --model-path models/pii_detectors/trustpilot/<timestamp>
```

## Troubleshooting

### Out of Memory

Reduce batch size or use chunking:

```yaml
# In model config
chunking:
  max_length: 256  # Reduce from 512
  strategy: truncate  # Or sliding_window
```

### Slow Inference

Use CPU-friendly methods for large-scale processing:

```bash
# Fast baselines
python3 model.py --model presidio ...  # ~10ms per doc
python3 model.py --model spacy ...     # ~50ms per doc

# Avoid for large datasets
python3 model.py --model petre ...     # ~2s per doc (iterative masking)
```

### Poor PII Detection

Increase training epochs or use domain-specific models:

```bash
python3 pii.py --dataset tab --mode train --epochs 10 --learning-rate 3e-5
```

Check nervaluate metrics for specific entity types:

```
Entity F1 scores:
  PERSON: 0.92
  ORG: 0.85
  LOC: 0.88
  DATE: 0.95  # Easy to detect
  DEM: 0.62   # Hard (need more training)
```

### TRI Model Not Distinguishing Authors

Ensure sufficient texts per author (minimum 10):

```python
from collections import Counter
adapter = get_adapter("tab", data_in="data/TAB/splitted/train.json")
author_counts = Counter(r.name for r in adapter.iter_records())
print(f"Authors with <10 texts: {sum(1 for c in author_counts.values() if c < 10)}")
```

Filter low-frequency authors in preprocessing:

```python
min_texts_per_author = 10
filtered_records = [r for r in records if author_counts[r.name] >= min_texts_per_author]
```

## Performance Tips

### Use GPU

Specify device in config:

```yaml
device: cuda  # or mps for Apple Silicon
```

Or auto-detect:

```yaml
device: auto  # CUDA > MPS > CPU
```

### Batch Processing

Increase batch size for throughput:

```yaml
batch_size: 32  # Default: 8
```

Trade-off: Higher memory usage.

### Caching Models

Models are loaded once per execution. For repeated calls, use programmatic API:

```python
# Bad: Loads model every time
for text in texts:
    os.system(f'python3 model.py --texts "{text}" --model dpmlm ...')

# Good: Load once, reuse
anonymizer = get_anonymizer("dpmlm", config_path="...")
for text in texts:
    result = anonymizer.anonymize(text, epsilon=1.0)
```

### Parallel Processing

Use builder for parallel anonymization:

```python
builder = anonymizer.builder()
results = builder.with_texts(texts).with_epsilons([1.0]).anonymize()
# Processes texts in batches
```
