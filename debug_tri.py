import json
from tqdm import tqdm

from dp.loaders import get_adapter
from dp.loaders.results import build_dataset_from_results
from dp.tri.with_bk import TRIDetectorWithBK

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Test RiskAnonymizer")
    parser.add_argument('--data', type=str, default=None, help='Dataset name')
    parser.add_argument('--data_in', type=str, default=None, help='Path to input data file')
    parser.add_argument('--result_in', type=str, default=None, help='Path to anonymization results JSONL file')
    parser.add_argument('--pipeline_in', type=str, default=None, help='Path to anonymization pipeline JSON file')
    parser.add_argument('--risk_in', type=str, default=None, help='Path to anonymization risks JSONL file')
    parser.add_argument('--start', type=int, default=0, help='Start index for records to process')
    parser.add_argument('--end', type=int, default=None, help='End index for records to process')
    parser.add_argument('--step', type=int, default=1, help='Step size for records to process')
    parser.add_argument('--max_records', type=int, default=None, help='Maximum number of records to load')
    parser.add_argument('--full_record', action='store_true', help='Whether to display full record information')
    parser.add_argument('--save_to_jsonl', type=str, default=None, help='Path to save the output records as JSONL')
    args = parser.parse_args()

    data_kwargs = dict(
        data=args.data, data_in=args.data_in
    )

    if not args.data or not args.data_in or not args.risk_in or not args.result_in:
        raise ValueError("data, data_in, risk_in and result_in are required")
    original_records = list(get_adapter(args.data, **data_kwargs).iter_records())
    args.end = args.end or len(original_records)

    records, indices = build_dataset_from_results(args.result_in, original_records)

    i=0
    with open(args.result_in, 'r', encoding='utf-8') as f:
        for line in f:
            records[i].text = json.loads(line)['text']
            i+=1

    scores = []
    offsets = []
    tokens = []
    with open(args.risk_in, 'r', encoding='utf-8') as f:
        for line in f:
            risk_record = json.loads(line)
            record = None
            for r in records:
                if r.uid == risk_record['uid']:
                    record = r
                    break
            scores.append(risk_record['scores'])
            offsets.append(risk_record['offsets'])
            tokens.append([record.text[offset[0]:offset[1]] for offset in offsets[-1]])

    records = records[args.start:args.end:args.step][:args.max_records or len(records)]
    scores = scores[args.start:args.end:args.step][:args.max_records or len(scores)]
    offsets = offsets[args.start:args.end:args.step][:args.max_records or len(offsets)]
    tokens = tokens[args.start:args.end:args.step][:args.max_records or len(tokens)]

    if not args.pipeline_in:
        raise ValueError("pipeline_in is required")
    detector = TRIDetectorWithBK(dataset_name=args.data)
    detector.load(args.pipeline_in)
    predictions = detector.predict(records)

    ranks = []
    for record in records:
        prediction = predictions.get(record.uid, {})
        ordered = sorted(prediction.items(), key=lambda item: item[1], reverse=True)
        rank = None
        for position, (candidate, _) in enumerate(ordered, start=1):
            if candidate == record.name:
                rank = position
                break
        ranks.append(rank)

    j = 0
    f = None
    if args.save_to_jsonl:
        f = open(args.save_to_jsonl, 'w', encoding='utf-8')
    for i in range(args.start, args.end, args.step):
        if args.max_records and j >= args.max_records:
            break
        if not args.save_to_jsonl:
            print(f"Record UID: {records[i].uid} | Evaluated Rank: {ranks[i]}")
            if args.full_record:
                for token, offset, score in zip(tokens[j], offsets[j], scores[j]):
                    print(f"Token: '{token}' | Offset: {offset} | Score: {score}")
        else:
            f.write(json.dumps({"uid": records[i].uid, "rank": ranks[i]}) + '\n')
            for token, offset, score in zip(tokens[j], offsets[j], scores[j]):
                f.write(json.dumps({"uid": records[i].uid, "token": token, "offset": offset, "score": score}) + '\n')
        j += 1