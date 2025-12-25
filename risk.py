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
    parser.add_argument('--result_in', type=str, default=None, help='Path to anonymization results JSONL file')
    parser.add_argument('--explainer', type=str, required=True, choices=['greedy', 'shap'], help='Anonymization model name')
    parser.add_argument('--explainer_in', type=str, required=True, help='Path to model config file')
    parser.add_argument('--max_records', type=int, default=None, help='Maximum number of records to load')
    parser.add_argument('--save_to_jsonl', type=str, default=None, help='Path to save output JSONL file')
    parser.add_argument('--starting_index', type=int, default=0, help='Starting index for processing records')
    parser.add_argument('--sort_by', type=str, choices=['scores', 'offsets'], default='offsets', help='Whether to sort tokens by score or offset in output')
    args = parser.parse_args()

    data_kwargs = dict(
        data=args.data, data_in=args.data_in, max_records=args.max_records
    )
    adapter = get_adapter(args.data, **data_kwargs) if args.data and args.data_in else None
    if args.explainer == 'greedy':
        explainer = GreedyExplainer(model_name=args.explainer_in)
    elif args.explainer == 'shap':
        explainer = ShapExplainer(model_name=args.explainer_in)
    splitter = TextSplitter()
    if args.result_in:
        if not args.data or not args.data_in:
            raise ValueError("data and data_in are required to load result_in")
        original_records = list(get_adapter(args.data, **data_kwargs).iter_records())
        records, _ = build_dataset_from_results(args.result_in, original_records)
    else:
        if adapter is None:
            raise ValueError("data and data_in are required when result_in is not provided")
        records = list(adapter.iter_records())
    records = records[args.starting_index:]
    for record in tqdm(records, desc="Explaining records"):
        tokens = []
        offsets = []
        for start, end, token in splitter.tokenize_with_spans(record.text):
            tokens.append(token)
            offsets.append((start, end))
        scores = explainer.explain(record.text, offsets)
        clear_memory()
        if args.sort_by == 'scores':
            sorted_indices = sorted(range(len(offsets)), key=lambda i: scores[i], reverse=True)
            offsets = [offsets[i] for i in sorted_indices]
            scores = [scores[i] for i in sorted_indices]
        if args.save_to_jsonl:
            with open(args.save_to_jsonl, 'a', encoding='utf-8') as f:
                output_record = {
                    'uid': record.uid,
                    'offsets': offsets,
                    'scores': scores.tolist() if hasattr(scores, 'tolist') else scores,
                }
                f.write(json.dumps(output_record) + '\n')
        else:
            print(f"Record UID: {record.uid}")
            for (start, end), score in zip(offsets, scores):
                print(f"  Offset: ({start}, {end}), Score: {score}")
