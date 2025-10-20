# Data

## Datasets

### TAB (Text Anonymization Benchmark)

**Domain**: Legal case documents from European Court of Human Rights

**Size**: 1,268 documents
- Train: 1,014 documents
- Validation: 127 documents
- Test: 127 documents

**Entity Types** (8 categories):
- `PERSON`: Individual names, aliases
- `ORG`: Organizations, institutions, companies
- `LOC`: Locations, addresses, geographic entities
- `DATE`: Dates, times, temporal expressions
- `CODE`: Case numbers, reference codes, identifiers
- `QUANTITY`: Numerical values with units
- `DEM`: Demographics (age, gender, occupation)
- `MISC`: Miscellaneous sensitive information

**Annotation Quality**: Professional annotators with legal expertise

**Format**: JSON with nested span annotations
```json
{
  "uid": "001-58258",
  "text": "John Doe was born on 15 March 1985...",
  "name": "001-58258",
  "spans": [
    {"start": 0, "end": 8, "label": "PERSON"},
    {"start": 22, "end": 36, "label": "DATE"}
  ]
}
```

**Usage**:
```bash
python3 data.py --data tab --data_in data/TAB/tab.json
```

**Characteristics**:
- Long documents (mean: 2,341 tokens, max: 15,000+)
- High entity density (avg 47 entities per document)
- Complex nested structures (quotes, parentheticals)
- Multi-lingual names and places

### Trustpilot Reviews

**Domain**: Customer reviews from e-commerce and service companies

**Companies** (6 domains):
- Amazon (www.amazon.com)
- Audible (www.audible.co.uk)
- HSBC (www.hsbc.co.uk)
- HelloFresh (hellofresh.co.uk)
- FlixBus (flixbus.co.uk)
- BackMarket (backmarket.co.uk)

**Size**: Varies by company (hundreds to thousands per domain)

**Entity Types**: Implicit PII in reviews
- Names in complaint narratives
- Locations in delivery descriptions
- Dates of service events
- Phone numbers, emails in complaints

**Format**: CSV with metadata
```csv
text,stars,category,date
"Terrible service at London branch...",1,retail,2024-01-15
```

**Usage**:
```bash
python3 data.py --data trustpilot --data_in data/trustpilot/www.amazon.com/train.json
```

**Characteristics**:
- Short texts (mean: 87 tokens)
- Informal language, typos, abbreviations
- Sentiment-labeled (1-5 stars)
- Real-world PII distribution (sparse, varied)

### DB-Bio

**Domain**: Biomedical article abstracts from PubMed

**Size**: From HuggingFace datasets

**Entity Types**: Biomedical entities
- Diseases, conditions
- Drugs, treatments
- Anatomical terms
- Procedures
- Clinical measurements

**Format**: HuggingFace Arrow format
```python
{
  "text": "Patient presented with hypertension...",
  "entities": [
    {"start": 23, "end": 35, "label": "DISEASE"}
  ]
}
```

**Usage**:
```bash
python3 data.py --data db_bio --data_in data/db_bio/test/data-00000-of-00001.arrow
```

**Characteristics**:
- Scientific terminology
- Abbreviations and acronyms
- Structured abstracts (Background, Methods, Results)
- Domain-specific entity types

## Data Processing

### Preprocessing Pipeline

**Splitting** (`data.py`):
```python
--data tab                 # Dataset name (required)
--data_in path/to/file     # Input file (required)
--max_records 1000         # Limit dataset size
```

Note: data.py does not have built-in splitting. Use separate train/dev/test files.

**Format Conversion**:
- HuggingFace datasets → unified `DatasetRecord` format
- Character-based spans preserved across tokenization
- Metadata extraction (UIDs, names for TRI)

**Quality Filtering**:
- Remove empty texts
- Validate span boundaries
- Deduplicate overlapping annotations

### Annotation Format

**Character-level Spans**:
```python
@dataclass
class TextAnnotation:
    start: int          # Character offset
    end: int            # Character offset (exclusive)
    label: str          # Entity type
    text: str           # Matched text snippet
```

**Token-level Labels** (for PII detector):
- BIO tagging: B-PERSON, I-PERSON, O
- Aligned with subword tokenization (BERT WordPiece)
- Handled by `_spans_to_token_labels()` in `pii_detector.py`

### Dataset Adapters

Each dataset has a specialized adapter implementing `DatasetAdapter`:

**TabDatasetAdapter**:
- Handles nested JSON structure
- Preserves hierarchical case metadata
- Groups by case UID for TRI training

**TrustpilotDatasetAdapter**:
- Parses CSV per company
- Extracts star ratings for utility evaluation
- Normalizes review text encoding

**DbBioDatasetAdapter**:
- Loads HuggingFace Arrow files
- Maps biomedical entity types to standard labels
- Handles abstract section markers

## Data Statistics

### TAB Dataset

| Split      | Documents | Avg Tokens | Avg Entities | Entity Density |
|------------|-----------|------------|--------------|----------------|
| Train      | 1,014     | 2,341      | 47.3         | 2.02%          |
| Validation | 127       | 2,298      | 46.1         | 2.01%          |
| Test       | 127       | 2,387      | 48.7         | 2.04%          |

**Entity Distribution** (train set):
- PERSON: 31.2%
- DATE: 24.8%
- ORG: 16.5%
- LOC: 12.3%
- CODE: 8.1%
- DEM: 4.2%
- QUANTITY: 1.9%
- MISC: 1.0%

### Trustpilot Dataset

| Company        | Reviews | Avg Tokens | Avg Stars |
|----------------|---------|------------|-----------|
| Amazon         | 2,847   | 94         | 2.3       |
| HSBC           | 1,523   | 78         | 1.8       |
| HelloFresh     | 982     | 103        | 3.6       |
| Audible        | 756     | 67         | 3.9       |
| FlixBus        | 634     | 89         | 2.7       |
| BackMarket     | 491     | 82         | 3.2       |

**Star Distribution** (across all):
- 1 star: 38.2%
- 2 stars: 12.5%
- 3 stars: 14.3%
- 4 stars: 16.7%
- 5 stars: 18.3%

### DB-Bio Dataset

*Statistics to be computed*

## Data Challenges

### Long Document Handling

**Problem**: Legal documents exceed BERT's 512 token limit

**Solutions**:
- `TruncateChunker`: Use first 512 tokens only
- `SlidingWindowChunker`: Process overlapping windows, merge predictions
- Future: Longformer, LED for native long-context processing

### Annotation Consistency

**Problem**: Overlapping entities, inconsistent boundaries

**Solution**: `_deduplicate_annotations()` in `ManualAnonymizer`
- Sort by start position
- Keep first annotation for overlapping spans
- Adjust offsets after redaction

### Token Alignment

**Problem**: Character spans don't align with subword tokens

**Solution**: `_spans_to_token_labels()` in `PIIDetector`
- Tokenize by whitespace for character positions
- Map character ranges to token indices
- Assign BIO labels at token level
- Handle partial token overlaps (assign to majority)

### Class Imbalance

**Problem**: O (non-entity) vastly outnumbers entity tokens

**Metrics**:
- Use F1 instead of accuracy
- Report per-entity-type metrics
- Evaluate with strict/partial/exact matching (nervaluate)

**Training**:
- Class weights in loss function
- Focal loss for hard examples
- Oversampling entity-dense documents

## Utility Labels

### TAB
- **Task**: Document classification by case type
- **Labels**: To be extracted from case metadata
- **Placeholder**: `utilities` field in `DatasetRecord`

### Trustpilot
- **Task**: Sentiment classification (star prediction)
- **Labels**: 1-5 stars (already available)
- **Multiclass**: Can treat as regression or 5-class classification

### DB-Bio
- **Task**: Biomedical NER, relation extraction
- **Labels**: Entity types in annotations
- **Placeholder**: Future expansion for relation pairs

## Data Privacy Considerations

**Ethical Review**: All datasets are publicly available or appropriately licensed

**TAB**: Court documents are public records (ECtHR)

**Trustpilot**: User reviews are publicly posted on Trustpilot website

**DB-Bio**: Anonymized clinical abstracts from PubMed

**No IRB Required**: No collection of new human subjects data, all secondary analysis of public data

## Extending to New Datasets

To add a new dataset:

1. **Create Adapter** in `loaders/`:
```python
class MyDatasetAdapter(DatasetAdapter):
    def iter_records(self):
        for item in self.load_data():
            yield DatasetRecord(
                text=item["text"],
                uid=item["id"],
                name=item.get("author", ""),
                spans=self.parse_spans(item),
                utilities={"label": item["category"]}
            )
```

2. **Register in `loaders/__init__.py`**:
```python
def get_adapter(data: str, **kwargs):
    adapters = {
        "tab": TabDatasetAdapter,
        "mydataset": MyDatasetAdapter,
    }
    return adapters[data](**kwargs)
```

3. **Add Preprocessing** in `data.py`:
```python
elif args.data == "mydataset":
    adapter = MyDatasetAdapter(...)
    # Custom splitting, filtering
```

4. **Update Configs**: Create `configs/model/*_mydataset.yaml` if needed
