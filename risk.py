import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple, List
import yaml
from tqdm import tqdm

from dp.utils.explainer.base import load_tri_label_mapping
from dp.utils.explainer import ShapExplainer, ShapType
from dp.loaders import get_adapter
from dp.loaders.results import build_dataset_from_results, load_result_records
from dp.utils.splitter import TextSplitter
from dp.utils.memory import clear_memory
from dp.utils.token_edits import map_result_offset_to_original
from dp.utils.tasking import resolve_task_id, apply_task_template

def _load_config(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {}
    p = Path(path).expanduser()
    data = yaml.safe_load(p.read_text(encoding='utf-8'))
    return data if isinstance(data, dict) else {}

def _merge_args(config: Dict[str, Any], args: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(config or {})
    for k, v in args.items():
        if v is not None:
            merged[k] = v
    return merged

def _resolve_params(params: Dict[str, Any]) -> Dict[str, Any]:
    dataset = params.get('data') or params.get('dataset')
    data_in = params.get('data_in')
    split = params.get('split')
    result_in = params.get('result_in')
    explainer_name = params.get('explainer') or 'shap'
    explainer_in = params.get('explainer_in') or params.get('tri_pipeline')
    save_to_jsonl = params.get('save_to_jsonl') or params.get('risk_out')
    max_records = params.get('max_records')
    starting_index = int(params.get('starting_index', 0))
    sort_by = params.get('sort_by') or 'offsets'
    offset_mode = params.get('offset_mode') or 'original'
    task_id = params.get('task_id')
    if task_id is not None and (not isinstance(task_id, int) or task_id < 0):
        raise ValueError('task_id must be a non-negative integer when provided')
    return {
        'data': dataset,
        'data_in': data_in,
        'split': split,
        'result_in': result_in,
        'explainer': explainer_name,
        'explainer_in': explainer_in,
        'save_to_jsonl': save_to_jsonl,
        'max_records': max_records,
        'starting_index': starting_index,
        'sort_by': sort_by,
        'offset_mode': offset_mode,
        'task_id': task_id,
    }

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description='Risk precomputation')
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--data', type=str, default=None)
    parser.add_argument('--data_in', type=str, default=None)
    parser.add_argument('--split', type=str, default=None)
    parser.add_argument('--result_in', type=str, default=None)
    parser.add_argument('--explainer', type=str, choices=['shap', 'shap_permutation'], default=None)
    parser.add_argument('--explainer_in', type=str, default=None)
    parser.add_argument('--max_records', type=int, default=None)
    parser.add_argument('--save_to_jsonl', type=str, default=None)
    parser.add_argument('--starting_index', type=int, default=None)
    parser.add_argument('--sort_by', type=str, choices=['scores', 'offsets'], default=None)
    parser.add_argument('--offset_mode', type=str, choices=['original', 'result'], default=None)
    parser.add_argument('--task_id', type=int, default=None, help='Task id for task-aware path templates')
    raw = parser.parse_args()

    cfg = _load_config(getattr(raw, 'config', None))
    params = _merge_args(cfg, vars(raw))
    resolved = _resolve_params(params)
    task_id = resolve_task_id(resolved.get('task_id'))
    for key in ('data_in', 'split', 'result_in', 'explainer_in', 'save_to_jsonl'):
        resolved[key] = apply_task_template(resolved.get(key), task_id)

    adapter = get_adapter(resolved['data'], data=resolved['data'], data_in=resolved['data_in'], split=resolved['split']) if resolved['data'] and resolved['data_in'] else None
    explainer_type = ShapType(resolved['explainer'])
    explainer = ShapExplainer(
        model_name=resolved['explainer_in'],
        explainer_type=explainer_type,
    )
    splitter = TextSplitter()

    token_edits_by_idx: Dict[int, List[Dict[str, Any]]] = {}
    if resolved['result_in']:
        if not resolved['data'] or not resolved['data_in']:
            raise ValueError('data and data_in are required to load result_in')
        original_records = list(get_adapter(resolved['data'], data=resolved['data'], data_in=resolved['data_in'], split=resolved['split']).iter_records())
        records, _ = build_dataset_from_results(resolved['result_in'], original_records)
        result_records = load_result_records(resolved['result_in'])
        result_texts = [rr.text for rr in result_records]
        for i, rr in enumerate(result_records):
            edits = [te.to_dict() for te in rr.annotations.token_edits]
            if edits:
                token_edits_by_idx[i] = edits
        if resolved['offset_mode'] == 'original' and not token_edits_by_idx:
            print('WARNING: --offset_mode=original but no token_edits found in result file.', file=sys.stderr)
            print('  Offsets will be in result-text coordinates, not original-text.', file=sys.stderr)
    else:
        if adapter is None:
            raise ValueError('data and data_in are required when result_in is not provided')
        records = list(adapter.iter_records())
        result_texts = [r.text for r in records]

    max_records = resolved['max_records'] or len(records) - resolved['starting_index']
    start = resolved['starting_index']
    records = records[start:start + max_records]
    result_texts = result_texts[start:start + max_records]

    mapping, _ = load_tri_label_mapping(explainer)
    if resolved['save_to_jsonl']:
        output_path = Path(resolved['save_to_jsonl'])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text('', encoding='utf-8')

    for idx, (record, text) in enumerate(tqdm(zip(records, result_texts), desc='Explaining records', total=len(records))):
        global_idx = start + idx
        offsets: List[Tuple[int, int]] = []
        for s, e, _ in splitter.tokenize_with_spans(text):
            offsets.append((s, e))
        target_label_id = mapping.get(record.name)
        target_label = f'LABEL_{target_label_id}'
        scores = explainer.explain(text, offsets, target_label=target_label)
        clear_memory()

        output_offsets = offsets
        if resolved['offset_mode'] == 'original' and global_idx in token_edits_by_idx:
            edits = token_edits_by_idx[global_idx]
            output_offsets = [map_result_offset_to_original(s, e, edits) for s, e in offsets]

        if resolved['sort_by'] == 'scores':
            order = sorted(range(len(output_offsets)), key=lambda i: scores[i], reverse=True)
            output_offsets = [output_offsets[i] for i in order]
            scores = [scores[i] for i in order]

        if resolved['save_to_jsonl']:
            with open(resolved['save_to_jsonl'], 'a', encoding='utf-8') as f:
                f.write(json.dumps({'uid': record.uid, 'offsets': output_offsets, 'scores': scores.tolist() if hasattr(scores, 'tolist') else scores}) + '\n')
        else:
            print(f'Record UID: {record.uid}')
            for (s, e), score in zip(output_offsets, scores):
                print(f"  Offset: ({s}, {e}), Score: {score}, Token: '{text[s:e]}'")

if __name__ == '__main__':
    main()
