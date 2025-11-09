# import argparse
# from datetime import datetime
# from pathlib import Path
# import random
# from typing import List

# from dp.loaders import ADAPTER_REGISTRY, DatasetRecord, get_adapter
# from dp.methods.simple._spacy import SpacyAnonymizer
# from dp.tri import get_tri_detector

# available_datasets = list(ADAPTER_REGISTRY.keys())


# def main():
#     parser = argparse.ArgumentParser(description="Train and evaluate TRI model with explicit data splits")
#     parser.add_argument("--dataset", type=str, default="tab", choices=available_datasets)
#     parser.add_argument("--data_path", type=str, required=True)
#     parser.add_argument("--model_name", type=str, default="distilbert-base-uncased")
#     parser.add_argument("--max_records", type=int, default=None)
#     parser.add_argument("--finetuning_epochs", type=int, default=15)
#     parser.add_argument("--pretraining_epochs", type=int, default=3)
#     parser.add_argument("--batch_size", type=int, default=16)
#     parser.add_argument("--pretraining_batch_size", type=int, default=8)
#     parser.add_argument("--use_pretraining", action="store_true")
#     parser.add_argument("--per_step", type=int, default=None)
#     parser.add_argument("--mode", type=str, default="train", choices=["train", "evaluate", "predict"])
#     parser.add_argument("--model_path", type=str, default=None)
#     parser.add_argument("--device", type=str, default="cpu")
#     parser.add_argument("--early_stop_threshold", type=float, default=None)
#     parser.add_argument("--train_fraction", type=float, default=0.7)
#     parser.add_argument("--val_fraction", type=float, default=0.15)
#     parser.add_argument("--test_fraction", type=float, default=0.15)
#     parser.add_argument("--split_seed", type=int, default=0)
#     parser.add_argument("--no_stratify", action="store_true")
#     parser.add_argument("--trustpilot_include_description", action="store_true")
#     parser.add_argument("--trustpilot_max_reviews", type=int, default=None)
#     args = parser.parse_args()

#     print(f"Loading {args.dataset} dataset from {args.data_path}...")
#     adapter = get_adapter(args.dataset, data=args.dataset, data_in=args.data_path, max_records=args.max_records)
#     records = list(adapter.iter_records())
#     if not records:
#         raise ValueError("Dataset returned no records")
#     if args.dataset == "trustpilot":
#         records = build_trustpilot_review_records(
#             records,
#             include_description=args.trustpilot_include_description,
#             max_reviews=args.trustpilot_max_reviews,
#             seed=args.split_seed,
#         )
#     print(f"✓ Loaded {len(records)} records")

#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     model_path = args.model_path or f"models/tri_pipelines/{args.dataset}/{timestamp}"
#     tri = get_tri_detector("split", dataset_name=args.dataset, model_name=args.model_name, max_length=512, device=args.device)

#     split_kwargs = {
#         "train_fraction": args.train_fraction,
#         "val_fraction": args.val_fraction,
#         "test_fraction": args.test_fraction,
#         "seed": args.split_seed,
#         "stratified": not args.no_stratify,
#     }

#     if args.mode == "train":
#         print("\nInitializing TRI detector for split workflow...")
#         if args.model_path:
#             existing = Path(args.model_path)
#             if not existing.exists():
#                 raise ValueError(f"Model path not found: {existing}")
#             print(f"\nLoading weights from {existing}...")
#             tri.load(str(existing))
#             model_path = str(existing)
#         tri.setup(records=records, **split_kwargs)
#         print(f"Splits: train={len(tri.train_records)} val={len(tri.eval_records)} test={len(tri.test_records)}")
#         print(f"\nFinetuning for {args.finetuning_epochs} epochs...")
#         tri.train(
#             epochs=args.finetuning_epochs,
#             batch_size=args.batch_size,
#             output_dir=model_path,
#             use_pretraining=args.use_pretraining,
#             pretraining_epochs=args.pretraining_epochs,
#             early_stop_threshold=args.early_stop_threshold,
#             per_step=args.per_step,
#         )
#         print(f"\nEvaluating on test split ({len(tri.test_records)} records)...")
#         test_results = tri.evaluate(tri.test_records)
#         print(f"✓ Test accuracy: {test_results['accuracy']:.4f} ({test_results['correct']}/{test_results['total']})")
#         print(f"\n✓ Model saved to {model_path}")
#     elif args.mode == "evaluate":
#         print(f"\nLoading model from {model_path}...")
#         tri.load(model_path)
#         tri.setup(records=records, **split_kwargs)
#         print(f"Evaluating on test split ({len(tri.test_records)} records)...")
#         results = tri.evaluate(tri.test_records)
#         print(f"✓ Results: {results}")
#     else:
#         print(f"\nLoading model from {model_path}...")
#         tri.load(model_path)
#         tri.setup(records=records, **split_kwargs)
#         sample = tri.test_records[:5] if len(tri.test_records) >= 5 else tri.test_records
#         predictions = tri.predict(sample)
#         print("\n✓ Sample predictions:")
#         for uid, probs in predictions.items():
#             label, score = max(probs.items(), key=lambda item: item[1])
#             print(f"  Record {uid}: {label} ({score:.2%})")


# def build_trustpilot_review_records(
#     companies: List[DatasetRecord],
#     include_description: bool,
#     max_reviews: int | None,
#     seed: int,
# ) -> list[DatasetRecord]:
#     masker = SpacyAnonymizer("spacy") if include_description else None
#     rng = random.Random(seed)
#     expanded: list[DatasetRecord] = []
#     for company in companies:
#         metadata_records = company.metadata.get("records") if company.metadata else None
#         if not metadata_records:
#             continue
#         reviews = list(metadata_records)
#         rng.shuffle(reviews)
#         if max_reviews is not None:
#             reviews = reviews[:max_reviews]
#         if not reviews:
#             continue
#         description = company.text
#         if include_description and masker:
#             description = masker.anonymize(description, labels=["ORG", "PERSON"]).text
#         base_metadata = {k: v for k, v in (company.metadata or {}).items() if k != "records"}
#         for idx, review in enumerate(reviews):
#             review_text = review.get("text")
#             if not review_text:
#                 continue
#             parts = []
#             if include_description and description:
#                 parts.append(description)
#             parts.append(review_text)
#             text = "\n\n".join(parts)
#             metadata = {
#                 **base_metadata,
#                 "review_index": idx,
#                 "stars": review.get("stars"),
#             }
#             expanded.append(
#                 DatasetRecord(
#                     text=text,
#                     uid=f"{company.uid}_{idx}",
#                     name=company.name,
#                     spans=company.spans,
#                     metadata=metadata,
#                 )
#             )
#     if not expanded:
#         raise ValueError("Trustpilot reviews could not be expanded from company records")
#     return expanded


# if __name__ == "__main__":
#     main()
