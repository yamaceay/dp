# PETRE Refactoring - Incremental k-Anonymization

## Problem Statement
The original `_petre.py` violated the incremental computation principle by precomputing all k values before streaming results. This broke the lazy evaluation pattern established in `_baroud.py` and `_risk.py`.

**Original Flow (BROKEN):**
```
For each k_value:
    call _ensure_annotations_for_k for all k values FIRST
Then:
    loop through k values and yield
```

**Issue:** Full computation happened upfront, not during iteration.

---

## Solution: Incremental Computation Inside Loop

### Core Changes

#### 1. Simplified `_ensure_annotations_for_k(target_k, state, starting_spans)` 
**Line Changes:** 100+ lines → 68 lines (32% reduction)

- Removed state caching infrastructure (`_annotation_history`, `_k_processed`, `_base_annotations_for`)
- Changed from class-level history tracking to pure function: takes `starting_spans`, returns masked `spans`
- Incremental approach: computes ONLY for current k, accepts previous k's spans as input
- No side effects: just returns new spans list

**Signature:**
```python
def _ensure_annotations_for_k(self, target_k: int, state: RecordState, starting_spans: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    # Compute incrementally until rank >= target_k
    # Return new spans (all masked PII for this k)
```

**Pattern:**
```python
spans: List[Tuple[int, int]] = list(starting_spans)
while True:
    rank = self._rank_from_probs(current_probs, state.label)
    if rank >= target_k:
        break
    # Find and mask next best token...
    spans.extend(new_spans)
    current_probs = self._evaluate_state(state, spans)
return spans
```

#### 2. Refactored `anonymize_from_dataset()` for Incremental Streaming
**Line Changes:** 38 lines → 43 lines (matches Baroud/Risk pattern)

**New Flow:**
```python
sorted_k_values = sorted(k_params.values(), reverse=True)  # Strictest to weakest
ledger = TokenLedger(text, state.term_spans)
all_result_spans = []
masked_token_indices = set()
current_spans = []  # Spans from previous k

for k_value in sorted_k_values:
    new_spans = _ensure_annotations_for_k(k_value, state, current_spans)
    
    # Mask only NEW tokens not masked before
    for token_idx in state.term_spans:
        if span in new_spans and token_idx not in masked_token_indices:
            ledger.replace(token_idx, mask)
            masked_token_indices.add(token_idx)
    
    current_spans = new_spans
    yield AnonymizationResult(text=ledger.render_offsets(text), ...)
```

**Key Properties:**
- ✅ Single ledger persists across all k iterations
- ✅ Spans accumulate incrementally 
- ✅ Only NEW tokens masked per iteration (via `masked_token_indices`)
- ✅ Intermediate results yielded per k value
- ✅ Current computation input from previous k via `current_spans`

---

## Consistency Across Methods

### Before Refactoring
| Aspect | Baroud | Risk | PETRE |
|--------|--------|------|-------|
| Ledger outside loop | ✅ | ✅ | ❌ (precomputed all k) |
| Spans accumulate | ✅ | ✅ | ❌ (stored in _annotation_history) |
| Compute inside loop | ✅ | ✅ | ❌ (called before loop) |
| Incremental tracking | ✅ (all_result_spans) | ✅ (masked_so_far) | ❌ (_k_processed) |

### After Refactoring  
| Aspect | Baroud | Risk | PETRE |
|--------|--------|------|-------|
| Ledger outside loop | ✅ | ✅ | ✅ |
| Spans accumulate | ✅ | ✅ | ✅ |
| Compute inside loop | ✅ | ✅ | ✅ |
| Incremental tracking | ✅ | ✅ | ✅ |
| Pass prev result | ✅ (prev_threshold) | ✅ (masked_so_far) | ✅ (current_spans) |

---

## Guidelines.md Compliance

✅ **Self-Explanatory Code:** Variable names (`current_spans`, `masked_token_indices`, `all_result_spans`) explain intent without comments

✅ **Type Strictness:** All types preserved (`List[Tuple[int, int]]`, `Set[int]`, `Iterator`)

✅ **Single Source of Truth:** One `_ensure_annotations_for_k` method, no caching duplicates

✅ **Minimal Changes:** Removed unnecessary state tracking, kept core algorithm identical

✅ **Top-Down Control:** Flow goes: sorted k values → ensure annotations → mask tokens → yield result

---

## Breaking Changes

⚠️ **API Change:** `_ensure_annotations_for_k` signature changed from class-level caching to pure function:
```python
# Old
def _ensure_annotations_for_k(self, target_k: int, indices: List[int]) -> None
    # Side effects: updates self._annotation_history, self._k_processed

# New
def _ensure_annotations_for_k(self, target_k: int, state: RecordState, starting_spans: List[Tuple[int, int]]) -> List[Tuple[int, int]]
    # Pure function: no side effects, returns new spans
```

⚠️ **Removed Attributes:**
- `self._annotation_history` - no longer needed
- `self._k_processed` - incremental tracking built into loop
- `self._base_annotations_for(k)` calls - now computed per-iteration

---

## Performance Impact

**Before:** O(n × k) - precomputed all k values for all records before streaming

**After:** O(1) startup, O(k) per-record streaming - lazy computation only for requested k values

**Memory:** Reduced - no accumulation of `_annotation_history` dict for all k values

---

## Testing Checklist

- [ ] `anonymize_from_dataset(idx)` yields intermediate results for each k
- [ ] Spans accumulate correctly across k values (k=2 ⊆ k=3 ⊆ k=5)
- [ ] Token masking is incremental (no duplicate masking)
- [ ] Text rendering matches input text length after masking
- [ ] Metadata correctly tracks k and perturbed_tokens per iteration
- [ ] Consistency: Results match Baroud/Risk pattern

---

## Files Modified

- `dp/methods/_petre.py`
  - `_ensure_annotations_for_k()` - simplified from 100+ lines to 68 lines
  - `anonymize_from_dataset()` - refactored for incremental streaming (43 lines)
  - Removed state caching infrastructure

---

## Next Steps

1. Run unit tests for PETRE anonymization
2. Verify incremental k-anonymity guarantee (rank should increase monotonically)
3. Profile performance vs original implementation
4. Update any documentation referencing `_annotation_history`
