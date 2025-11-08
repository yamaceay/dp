import argparse
from pathlib import Path
from datetime import datetime

from dp.tri import get_tri_detector
from dp.tri.loaders import get_attacker_adapter, ATTACKER_ADAPTER_REGISTRY

available_datasets = list(ATTACKER_ADAPTER_REGISTRY.keys())

def main():
    parser = argparse.ArgumentParser(description="Train and evaluate TRI model for re-identification")
    parser.add_argument("--dataset", type=str, default="tab", choices=available_datasets, 
                        help="Dataset name")
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to dataset file")
    parser.add_argument("--model_name", type=str, default="distilbert-base-uncased",
                        help="Base model name")
    parser.add_argument("--max_records", type=int, default=None,
                        help="Maximum number of records to use")
    parser.add_argument("--finetuning_epochs", type=int, default=15,
                        help="Number of finetuning epochs")
    parser.add_argument("--pretraining_epochs", type=int, default=3,
                        help="Number of pretraining epochs")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for training")
    parser.add_argument("--pretraining_batch_size", type=int, default=8,
                        help="Batch size for pretraining")
    parser.add_argument("--use_pretraining", action="store_true",
                        help="Use MLM pretraining before finetuning")
    parser.add_argument("--per_step", type=int, default=None,
                        help="Evaluation frequency in steps (default: per epoch)")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "evaluate", "predict"],
                        help="Mode: train, evaluate, or predict")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to an existing TRI model checkpoint (optional for train/evaluate/predict)")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device to use (auto, cuda, mps, cpu)")
    parser.add_argument("--attacker_extensions", type=str, default=None,
                        help="Optional JSONL file with precomputed attacker extensions (BK + summary)")
    parser.add_argument("--early_stop_threshold", type=float, default=None,
                        help="Minimum accuracy threshold across all eval datasets to stop training early (0-100)")
    
    args = parser.parse_args()
    
    print(f"Loading {args.dataset} dataset from {args.data_path}...")
    
    adapter = get_attacker_adapter(args.dataset, data=args.dataset, data_in=args.data_path, max_records=args.max_records)
    if args.attacker_extensions:
        adapter.load_cache_from_jsonl(args.attacker_extensions)
    
    records = list(adapter.iter_records())
    print(f"✓ Loaded {len(records)} records")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = args.model_path or f"models/tri_pipelines/{args.dataset}/{timestamp}"
    
    tri = get_tri_detector("bk", dataset_name=args.dataset, model_name=args.model_name, max_length=512, device=args.device)

    if args.mode == "train":
        print(f"\nInitializing TRI detector for {args.dataset} (bk)...")
        if args.model_path:
            model_path = Path(args.model_path)
            if not model_path.exists():
                raise ValueError(f"Model path not found: {model_path}")
            print(f"\nLoading weights from {model_path}...")
            tri.load(str(model_path))
        
        tri.setup(records=records)
        
        print(f"\nFinetuning for {args.finetuning_epochs} epochs...")
        tri.train(
            epochs=args.finetuning_epochs,
            batch_size=args.batch_size,
            output_dir=model_path,
            use_pretraining=args.use_pretraining,
            pretraining_epochs=args.pretraining_epochs,
            early_stop_threshold=args.early_stop_threshold,
            per_step=args.per_step,
        )
        
        print(f"\n✓ Model saved to {model_path}")
        
    elif args.mode == "evaluate":
        print(f"\nLoading model from {model_path}...")
        tri.load(model_path)
        
        tri.setup(records=records)
        
        print(f"\nEvaluating on test records...")
        results = tri.evaluate(tri.eval_records)
        print(f"✓ Results: {results}")
        
    elif args.mode == "predict":
        print(f"\nLoading model from {model_path}...")
        tri.load(model_path)
        
        tri.setup(records=records)
        
        print("\nPredicting on sample records...")
        sample_records = tri.eval_records[:5] if len(tri.eval_records) >= 5 else tri.eval_records
        predictions = tri.predict(sample_records)
        
        print("\n✓ Sample predictions:")
        for uid, probs in list(predictions.items())[:5]:
            top_label = max(probs.items(), key=lambda x: x[1])
            print(f"  Record {uid}: {top_label[0]} ({top_label[1]:.2%})")

if __name__ == "__main__":
    main()
