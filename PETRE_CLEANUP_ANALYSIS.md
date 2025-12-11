# PETRE Bloat Analysis: What TokenLedger Makes Unnecessary

## Summary
PETRE has ~750 lines with 30+ methods, many of which are annotation/span manipulation utilities that **TokenLedger already handles better**. Here's what's redundant:

---

## Tier 1: Direct Redundancy (Can Delete Immediately)

### 1. `_normalize_annotation_list()` (Lines 311-350, 40 lines)
**Purpose**: Convert various input formats to TextAnnotation objects

**Why it's unnecessary:**
- Takes Iterable and normalizes to `List[TextAnnotation]`
- Just copies properties between TextAnnotation objects
- Creates validation overhead
- **TokenLedger doesn't need annotations** - it works with raw `List[Tuple[int, int]]` spans directly

**Who calls it:**
- `set_annotations()` line 481

**What to do instead:**
```python
# Current: converts annotations to normalized list
normalized = self._normalize_annotation_list(raw_items)

# Better: just extract spans directly
spans = [(int(ann.start), int(ann.end)) if isinstance(ann, TextAnnotation) else (int(ann[0]), int(ann[1])) 
         for ann in raw_items if valid(ann)]
```

**Delete candidate: YES** ✂️

---

### 2. `_clone_annotation_dict()` (Lines 352-368, 17 lines)
**Purpose**: Deep copy of annotation dictionaries

**Why it's unnecessary:**
- Only used during setup (lines 483-484)
- Creates unnecessary memory copies
- TokenLedger stores spans as simple tuples - no cloning needed
- Could just store one reference if needed at all

**Who calls it:**
- `set_annotations()` lines 483-484 (2 calls)

**What to do instead:**
```python
# Current: clones dict and nested annotations
self._starting_annotations = self._clone_annotation_dict(aligned)

# Better: just reference what we need
self._starting_annotations = {uid: [(ann.start, ann.end) for ann in anns] 
                              for uid, anns in aligned.items()}
```

**Delete candidate: YES** ✂️

---

### 3. `_align_annotations()` (Lines 281-308, 28 lines)
**Purpose**: Expand and deduplicate annotation spans

**Why it's problematic:**
- Expands character spans to token boundaries using `_expand_span_to_tokens()`
- Deduplicates overlapping spans
- Sorts annotations by position
- **TokenLedger already handles span boundaries** via `render_offsets()`
- **No need to pre-align** if we just pass raw spans to TokenLedger

**Who calls it:**
- `set_annotations()` line 482
- `_ensure_annotations_for_k()` (old version, now gone)

**What to do instead:**
```python
# Current: aligns every annotation before storing
aligned[uid] = self._align_annotations(state, normalized)

# Better: pass raw spans to TokenLedger during rendering
# Alignment happens naturally when TokenLedger renders
spans = [(ann.start, ann.end) for ann in raw_items]
```

**Delete candidate: YES (if we refactor how spans are used)** ⚠️

---

### 4. `_expand_span_to_tokens()` (Lines 251-279, 29 lines)
**Purpose**: Map character spans to token boundaries

**Why it's unnecessary:**
- Only used by `_align_annotations()`
- Tries to expand a span to nearest token boundaries
- **TokenLedger already has token positions** - just use those directly
- If span is already on token boundary, this does nothing
- If span is mid-token, TokenLedger handles it in `render_offsets()`

**Who calls it:**
- `_align_annotations()` line 291 only

**Delete candidate: YES (if we remove _align_annotations)** ✂️

---

### 5. `_apply_spans_to_sentence()` (Lines 494-516, 23 lines)
**Purpose**: Mask spans within a sentence

**Why it's unnecessary:**
- Manually builds masked text by string slicing
- **TokenLedger does this via `replace()`**
- Redundant work

**Who calls it:**
- `_evaluate_state()` line 539

**What to do instead:**
```python
# Current: manual string manipulation
relevant = [s for s in spans if sent_start < s[0] and s[1] < sent_end]
for start, end in relevant:
    segment = segment[:start] + self.mask_text + segment[end:]

# Better: use TokenLedger inside sentence bounds
sentence_ledger = TokenLedger(sentence_text, [(s[0] - sent_start, s[1] - sent_start) for s in relevant_spans])
for idx, span in enumerate(relevant_spans):
    sentence_ledger.replace(idx, self.mask_text)
return sentence_ledger.render_offsets(sentence_text)
```

**Delete candidate: YES** ✂️

---

## Tier 2: Problematic/Inefficient (Should Simplify)

### 6. `_apply_spans_to_text()` (Lines 517-536, 20 lines)
**Status**: ALREADY REFACTORED to use TokenLedger ✅

**Current implementation (lines 517-528):**
```python
def _apply_spans_to_text(self, text: str, spans: List[Tuple[int, int]]) -> str:
    if not spans:
        return text
    ledger = TokenLedger(text, tuple(sorted(set(spans))))
    span_set = set(spans)
    for idx, entry in enumerate(ledger.iter_entries()):
        if (entry.start, entry.end) in span_set:
            ledger.replace(idx, self.mask_text)
    return ledger.render_offsets(text)
```

This is already good! Keep it as reference.

---

### 7. `_build_terms_to_ignore()` (Lines 171-195, 25 lines)
**Purpose**: Build set of terms to skip during masking

**Issues:**
- Loads stopwords and marks to ignore
- Creates normalized versions (lowercase)
- Fine-grained logic for handling annotations
- **Not used by the refactored `_ensure_annotations_for_k()`** anymore
- Only called during setup

**Who calls it:**
- `add_dataset_records()` line 82
- `set_annotations()` line 485

**Verdict:**
- Keep for now, but mark for review
- Used by `_should_ignore()` which is still active

**Delete candidate: MAYBE** ❓

---

### 8. `_should_ignore()` (Lines 610-619, 10 lines)
**Purpose**: Check if a term should be skipped

**Issues:**
- Uses `_terms_to_ignore` set and regex pattern
- Called during token selection in `_ensure_annotations_for_k()`
- Reasonable logic, but only skips terms already in cached set
- Could be simplified to just check if token is punctuation

**Who calls it:**
- `_ensure_annotations_for_k()` line 665

**Verdict:**
- Could be 1-liner: `return not token_text.strip() or token_text in self._terms_to_ignore`
- Keep logic but simplify

**Delete candidate: NO, simplify** ⚙️

---

## Tier 3: Necessary Infrastructure (Keep With Review)

### 9. Record State Management
**Methods:**
- `_build_label_mappings()` - Lines 197-208 (12 lines)
- `_build_record_states()` - Lines 209-250 (42 lines)

**Status:** NECESSARY
- Creates RecordState dataclass with tokenization
- Splits text into sentences and tokens
- Maps term texts to indices
- All needed for `_ensure_annotations_for_k()` to work

**Delete candidate: NO** ✅

---

### 10. Scoring Infrastructure
**Methods:**
- `_token_scores_for_state()` - Lines 533-583 (51 lines)
- `_ordered_token_indices_for_state()` - Lines 585-591 (7 lines)
- `_parse_label()` - Lines 521-524 (4 lines)

**Status:** NECESSARY
- Loads pre-computed risk scores or generates via explainer
- Caches scores per record
- Critical for choosing which tokens to mask

**Delete candidate: NO** ✅

---

### 11. Pipeline & Device Management
**Methods:**
- `_resolve_device()` - Lines 375-381 (7 lines)
- `_pipeline_device()` - Lines 383-388 (6 lines)
- `_load_tri_pipeline()` - Lines 519-531 (13 lines)

**Status:** NECESSARY
- Loads TRI classification pipeline
- Handles device selection (CUDA/MPS/CPU)
- Essential for `_evaluate_state()`

**Delete candidate: NO** ✅

---

### 12. Evaluation Logic
**Methods:**
- `_evaluate_state()` - Lines 555-583 (29 lines)
- `_rank_from_probs()` - Lines 596-604 (9 lines)

**Status:** NECESSARY
- Runs TRI model on masked text to check k-anonymity
- Returns rank of true author's label
- Core k-anonymity guarantee check

**Delete candidate: NO** ✅

---

### 13. Candidate Span Selection
**Methods:**
- `_expand_candidate_spans()` - Lines 590-608 (19 lines)
- `_span_overlaps_existing()` - Lines 608-614 (7 lines)

**Status:** NECESSARY
- Finds all instances of a token to mask (if `mask_all_instances`)
- Checks overlap constraints
- Core masking logic

**Delete candidate: NO** ✅

---

## Summary Table: Delete vs Keep

| Method | Lines | Status | Reason |
|--------|-------|--------|--------|
| `_normalize_annotation_list()` | 40 | ✂️ DELETE | TokenLedger handles spans directly |
| `_clone_annotation_dict()` | 17 | ✂️ DELETE | Unnecessary deep copy |
| `_align_annotations()` | 28 | ✂️ DELETE | TokenLedger handles boundaries |
| `_expand_span_to_tokens()` | 29 | ✂️ DELETE | Only used by _align_annotations |
| `_apply_spans_to_sentence()` | 23 | ✂️ DELETE | Replace with TokenLedger |
| `_build_terms_to_ignore()` | 25 | ❓ MAYBE | Only used during setup |
| `_should_ignore()` | 10 | ⚙️ SIMPLIFY | Can be 1-liner |
| All record/scoring/pipeline/eval | 200+ | ✅ KEEP | Core functionality |

**Total deletable: 180 lines (24% of file)**

---

## Refactoring Path

### Phase 1: Delete Annotation Manipulation (180 lines)
```
Current: annotations → _normalize_annotation_list → _align_annotations → _expand_span_to_tokens
         ↓
         stored in self._starting_annotations, self.annotations

New: Just store spans as List[Tuple[int, int]] per record
     Skip all alignment - TokenLedger handles it during render_offsets()
```

### Phase 2: Simplify Span Masking
```
Current: _apply_spans_to_sentence() uses manual string manipulation
New: Use TokenLedger.replace() for sentence rendering too
```

### Phase 3: Consolidate Setup Logic
```
Current: set_annotations() does:
  - normalize → align → clone (3 operations, 85+ lines)
  
New: set_annotations() does:
  - Extract spans, store them, done (5 lines)
```

### Result
**Before:** 753 lines, 30+ methods
**After:** ~570 lines, 20 methods
**Reduction:** 24%, cleaner, matches TokenLedger pattern

---

## Questions for User

1. Should we keep `_terms_to_ignore` / `_should_ignore()` or is it dead code?
2. Should setup phase handle `_normalize_annotation_list()` at all, or just assume spans are valid?
3. Do we actually need to store `_starting_annotations` or can we derive it from `state.term_spans`?
4. Should `_apply_spans_to_sentence()` be replaced entirely, or keep for `_evaluate_state()`?
