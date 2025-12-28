import json
from tqdm import tqdm

from dp.utils.explainer.base import load_tri_label_mapping
from dp.utils.explainer import ShapExplainer
from dp.loaders import get_adapter
from dp.loaders.results import build_dataset_from_results, load_result_records
from dp.utils.splitter import TextSplitter
from dp.utils.memory import clear_memory
from dp.utils.token_edits import map_result_offset_to_original

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Test RiskAnonymizer")
    parser.add_argument('--data', type=str, default=None, help='Dataset name')
    parser.add_argument('--data_in', type=str, default=None, help='Path to input data file')
    parser.add_argument('--result_in', type=str, default=None, help='Path to anonymization results JSONL file')
    parser.add_argument('--explainer', type=str, required=True, choices=['shap'], help='Anonymization model name')
    parser.add_argument('--explainer_in', type=str, required=True, help='Path to model config file')
    parser.add_argument('--max_records', type=int, default=None, help='Maximum number of records to load')
    parser.add_argument('--save_to_jsonl', type=str, default=None, help='Path to save output JSONL file')
    parser.add_argument('--starting_index', type=int, default=0, help='Starting index for processing records')
    parser.add_argument('--sort_by', type=str, choices=['scores', 'offsets'], default='offsets', help='Whether to sort tokens by score or offset in output')
    parser.add_argument('--offset_mode', type=str, choices=['original', 'result'], default='original', help='Output offsets in original or result text coordinates')
    args = parser.parse_args()

    data_kwargs = dict(
        data=args.data, data_in=args.data_in
    )
    adapter = get_adapter(args.data, **data_kwargs) if args.data and args.data_in else None
    if args.explainer == 'shap':
        explainer = ShapExplainer(model_name=args.explainer_in)
    splitter = TextSplitter()
    
    token_edits_by_idx = {}
    if args.result_in:
        if not args.data or not args.data_in:
            raise ValueError("data and data_in are required to load result_in")
        original_records = list(get_adapter(args.data, **data_kwargs).iter_records())
        records, _ = build_dataset_from_results(args.result_in, original_records)
        result_records = load_result_records(args.result_in)
        result_texts = [rr.text for rr in result_records]
        for i, rr in enumerate(result_records):
            edits = [te.to_dict() for te in rr.annotations.token_edits]
            if edits:
                token_edits_by_idx[i] = edits
    else:
        if adapter is None:
            raise ValueError("data and data_in are required when result_in is not provided")
        records = list(adapter.iter_records())
        result_texts = [r.text for r in records]
    
    records = records[args.starting_index:args.starting_index + (args.max_records or len(records) - args.starting_index)]
    result_texts = result_texts[args.starting_index:args.starting_index + (args.max_records or len(result_texts) - args.starting_index)]

    mapping, _ = load_tri_label_mapping(explainer)

    for idx, (record, text) in enumerate(tqdm(zip(records, result_texts), desc="Explaining records", total=len(records))):
        global_idx = args.starting_index + idx
        tokens = []
        offsets = []
        for start, end, token in splitter.tokenize_with_spans(text):
            tokens.append(token)
            offsets.append((start, end))
        target_label_id = mapping.get(record.name)
        target_label = f"LABEL_{target_label_id}"
        scores = explainer.explain(text, offsets, target_label=target_label)
        clear_memory()
        
        output_offsets = offsets
        if args.offset_mode == 'original' and global_idx in token_edits_by_idx:
            edits = token_edits_by_idx[global_idx]
            output_offsets = [map_result_offset_to_original(s, e, edits) for s, e in offsets]
        
        if args.sort_by == 'scores':
            sorted_indices = sorted(range(len(output_offsets)), key=lambda i: scores[i], reverse=True)
            output_offsets = [output_offsets[i] for i in sorted_indices]
            scores = [scores[i] for i in sorted_indices]
        if args.save_to_jsonl:
            with open(args.save_to_jsonl, 'a', encoding='utf-8') as f:
                output_record = {
                    'uid': record.uid,
                    'offsets': output_offsets,
                    'scores': scores.tolist() if hasattr(scores, 'tolist') else scores,
                }
                f.write(json.dumps(output_record) + '\n')
        else:
            print(f"Record UID: {record.uid}")
            for (start, end), score in zip(output_offsets, scores):
                print(f"  Offset: ({start}, {end}), Score: {score}")
