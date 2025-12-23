# model.py Simplification & Vulnerability Analysis

## Complexity Issues & Vulnerabilities

### 1. **Excessive Conditional Branching in `stream_anonymization()` (Lines 242-340)**
**Vulnerability**: Multiple nested if-else chains with repeated code patterns
- 4 separate conditional branches (`requires_k`, `must_use_dataset`, `requires_epsilon`, else)
- Each branch duplicates validation and output logic
- **Risk**: Maintenance bugs, inconsistent error handling, logic divergence

**Lines**:
```python
if capabilities.requires_k:
    # Block A: stream handling
    
elif capabilities.must_use_dataset:
    # Block B: dataset handling
    
elif capabilities.requires_epsilon:
    # Block C: epsilon handling
    
else:
    # Block D: generic handling with nested parameter logic
```

**Problem**: Block D contains another nested if-else for parameter handling (lines 300-337)

---

### 2. **Configuration Extraction Fragmentation (Lines 407-422)**
**Vulnerability**: Scattered extraction of nested dictionary keys
- `explainer_block` extraction spans 7 lines with manual key checks
- `nested_risk_path` renamed inconsistently
- **Risk**: Silent failures if keys missing, no validation

```python
if isinstance(explainer_block, dict):
    explainer_name = explainer_block.get("name")
    explainer_path = explainer_block.get("tri_pipeline")
    risk_temperature = explainer_block.get("risk_temperature")
    nested_risk_path = explainer_block.get("risk_scores")  # Renamed to risk_path
```

---

### 3. **Selector Instantiation Logic (Lines 439-456)**
**Vulnerability**: Nested if-elif with implicit state management
- 3 conditional branches with `None` type checking
- `PIIOnlyUnit` requires `pii_annotator_path`, `ByRiskUnit` requires `risk_tolerance`
- **Risk**: Missing validation on intermediate values, unclear dependency ordering

```python
if type_of_selector is not None and type_of_selector == "pii_only":
    # Requires pii_annotator_path validation
elif type_of_selector is not None and type_of_selector == "by_risk":
    # Requires risk_tolerance validation
```

---

### 4. **Explainer Instantiation Complexity (Lines 458-475)**
**Vulnerability**: Cascading conditions with implicit fallback
- 5 nested condition checks
- Default behavior ("uniform" explainer) not enforced consistently
- **Risk**: State confusion if conditions partially satisfied

```python
if explainability is None:
    explainability = "uniform"  # Default

if capabilities.must_use_non_uniform_explainer:
    if explainability == "uniform":
        raise ValueError(...)  # Contradicts default

if explainability == "uniform":
    explainer = UniformExplainer()
elif explainer_path is not None:
    if explainability == "greedy":
        # ...
    elif explainability == "shap":
        # ...
```

---

### 5. **Risk Path Resolution (Lines 476-483)**
**Vulnerability**: Two-stage fallback with unclear priority
- Model config path checked, then runtime path used as fallback
- **Risk**: Silent overrides, implicit precedence rules

```python
if risk_path is None and runtime_risk_path is not None:
    risk_path = runtime_risk_path
```

---

### 6. **Selection Logic Fragmentation (Lines 506-536)**
**Vulnerability**: Split validation across multiple conditions
- Mutually exclusive checks spread over 30 lines
- Requires all three branches to understand valid states

```python
if texts_arg is not None and indices_arg is not None:
    raise ValueError(...)

if capabilities.must_use_dataset:
    if texts_arg is not None:
        raise ValueError(...)
    # Build dataset selection

else:
    # Build text selection
```

---

### 7. **Parameter Vector Initialization (Lines 538-560)**
**Vulnerability**: Separate initialization blocks with repeated patterns
- 4 separate if-blocks for different parameter types
- Each uses `.with_*()` builder pattern with different defaults
- **Risk**: Inconsistent default handling

```python
if args.model in pii_confidence_models and runtime_bundle.pii_confidence_values:
    builder.with_pii_confidences(...)

if args.model in risk_tolerance_models and runtime_bundle.risk_tolerance_values:
    builder.with_risk_tolerances(...)

if capabilities.requires_k:
    ks = runtime_bundle.k_values or [5]  # Default applied here

if capabilities.requires_epsilon:
    epsilons = runtime_bundle.epsilon_values or [100.0]  # Default applied here
```

---

### 8. **Output Handler Instantiation (Lines 493-497)**
**Vulnerability**: Registry lookup with implicit fallback
- Missing handler type falls back to "print" without logging
- **Risk**: Silent degradation, difficult debugging

```python
output_handler_cls = OUTPUT_HANDLER_REGISTRY.get(args.output, OUTPUT_HANDLER_REGISTRY["print"])

if args.output in ["jsonl"]:
    output_handler = output_handler_cls(timestamp=batch_timestamp)
else:
    output_handler = output_handler_cls()
```

---

## Recommended Simplifications

### Strategy 1: Extract Config Builders
Create helper functions to consolidate config extraction:

```python
def extract_explainer_config(model_config: dict) -> Dict[str, Any]:
    block = model_config.pop("explainer", {})
    if not isinstance(block, dict):
        return {}
    return {
        "name": block.get("name"),
        "path": block.get("tri_pipeline"),
        "temperature": block.get("risk_temperature"),
        "risk_scores": block.get("risk_scores"),
    }

def extract_token_selector_config(model_config: dict) -> Dict[str, Any]:
    return model_config.get("token_selection", {})

def extract_chunking_config(model_config: dict, kind: str) -> Dict[str, Any]:
    return model_config.get(f"{kind}_chunking", {})
```

### Strategy 2: Unify Stream Processing
Replace 4-branch stream_anonymization with single dispatch:

```python
def stream_anonymization(
    *,
    anonymizer: Anonymizer,
    builder: AnonymizationBuilder,
    capabilities,
    **kwargs
) -> Tuple[int, float]:
    run_start = time.time()
    processed = 0
    
    handler = StreamHandler.for_capability(capabilities)
    for record in handler.stream(builder, anonymizer, **kwargs):
        output_result(record, **kwargs)
        processed += 1
    
    elapsed = time.time() - run_start
    return processed, elapsed
```

### Strategy 3: Registry-Based Selector Creation
Replace conditional selector instantiation:

```python
SELECTOR_REGISTRY = {
    "pii_only": PIIOnlyUnit.from_config,
    "by_risk": ByRiskUnit.from_config,
    None: AllUnit,
}

selector_type = token_selection_config.get("name")
selector_cls = SELECTOR_REGISTRY.get(selector_type)
if selector_cls is None:
    raise ValueError(f"Unknown selector: {selector_type}")
selector = selector_cls(token_selection_config)
```

### Strategy 4: Consolidate Parameter Initialization
Single loop for builder parameters:

```python
PARAMETER_MAPPINGS = {
    "pii_confidence": ("pii_confidence_models", "pii_confidence_values", "with_pii_confidences"),
    "risk_tolerance": ("risk_tolerance_models", "risk_tolerance_values", "with_risk_tolerances"),
}

for param_key, (model_set, values_attr, builder_method) in PARAMETER_MAPPINGS.items():
    if args.model in eval(model_set) and getattr(runtime_bundle, values_attr):
        values = getattr(runtime_bundle, values_attr)
        getattr(builder, builder_method)(values)
```

### Strategy 5: Standardize Config Merging
Replace two-stage risk_path resolution:

```python
def resolve_config_priority(model_config: dict, runtime_config: dict, key: str) -> Optional[Any]:
    return model_config.get(key) or runtime_config.get(key)
```

---

## High-Risk Areas Summary

| Issue | Severity | Lines | Impact |
|-------|----------|-------|--------|
| 4-way stream branching | HIGH | 242-340 | Logic divergence, inconsistent error handling |
| Nested parameter logic | HIGH | 300-337 | Silent failures, parameter conflicts |
| Selector instantiation | MEDIUM | 439-456 | Missing validation, implicit dependencies |
| Explainer cascading | MEDIUM | 458-475 | State confusion, contradictory defaults |
| Output handler fallback | MEDIUM | 493-497 | Silent degradation |
| Parameter initialization | MEDIUM | 538-560 | Inconsistent defaults |
| Risk path fallback | LOW | 476-483 | Implicit precedence |
| Selection branching | LOW | 506-536 | Fragmented validation |

---

## Next Steps

1. **Priority 1**: Refactor `stream_anonymization()` using dispatch pattern
2. **Priority 2**: Consolidate parameter initialization in single loop
3. **Priority 3**: Extract config builders into helper functions
4. **Priority 4**: Add comprehensive logging for fallback paths
