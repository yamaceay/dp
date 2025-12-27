import json
from tqdm import tqdm

from dp.utils.explainer import GreedyExplainer, ShapExplainer
from dp.loaders import get_adapter
from dp.loaders.results import build_dataset_from_results
from dp.utils.splitter import TextSplitter
from dp.utils.memory import clear_memory

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Test RiskAnonymizer")
    parser.add_argument('--data', type=str, default=None, help='Dataset name')
    parser.add_argument('--data_in', type=str, default=None, help='Path to input data file')
    parser.add_argument('--risk_in', type=str, default=None, help='Path to anonymization results JSONL file')
    parser.add_argument('--max_records', type=int, default=None, help='Maximum number of records to load')
    args = parser.parse_args()

    data_kwargs = dict(
        data=args.data, data_in=args.data_in, max_records=args.max_records
    )
    adapter = get_adapter(args.data, **data_kwargs) if args.data and args.data_in else None
    splitter = TextSplitter()

    if not args.data or not args.data_in or not args.risk_in:
        raise ValueError("data, data_in and risk_in are required")
    records = list(get_adapter(args.data, **data_kwargs).iter_records())

    scores = []
    offsets = []
    tokens = []
    with open(args.risk_in, 'r', encoding='utf-8') as f:
        for line, record in zip(f, records):
            risk_record = json.loads(line)
            scores.append(risk_record['scores'])
            offsets.append(risk_record['offsets'])
            tokens_for_record = [record.text[offset[0]:offset[1]] for offset in offsets[-1]]
            tokens.append(tokens_for_record)

    for record, record_scores, record_offsets, record_tokens in tqdm(zip(records, scores, offsets, tokens), desc="Processing records", total=len(records)):
        print(f"Record UID: {record.uid}")
        for token, offset, score in zip(record_tokens, record_offsets, record_scores):
            print(f"Token: '{token}' | Offset: {offset} | Score: {score}")