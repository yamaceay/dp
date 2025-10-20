import argparse
from typing import Dict, Any, List
from pathlib import Path
from dp.loaders.base import load_data_from_config
from dp.experiments.reidentification import ReidentificationRiskExperiment
from dp.methods.constants import get_capabilities, requires_dataset
from dp.methods.registry import is_k_anonymizer, is_dp_anonymizer, MODEL_REGISTRY


def add_data_args(parser: argparse.ArgumentParser) -> List[str]:
    parser.add_argument("--data", type=str, required=True, help="Dataset name")
    parser.add_argument("--data_in", type=str, default=None, help="Data config path")
    parser.add_argument("--max_records", type=int, default=None, help="Max records")
    return ["data", "data_in", "max_records"]


def add_model_args(parser: argparse.ArgumentParser) -> List[str]:
    parser.add_argument("--model", type=str, required=True, help="Model name")
    parser.add_argument("--model_in", type=str, required=True, help="Model config path")
    return ["model", "model_in"]


def add_experiment_args(parser: argparse.ArgumentParser) -> List[str]:
    parser.add_argument("--tri_model_path", type=str, default=None, help="Pre-trained TRI model path")
    parser.add_argument("--tri_max_length", type=int, default=512, help="TRI max sequence length")
    parser.add_argument("--tri_device", type=str, default="auto", help="TRI device")
    parser.add_argument("--parameters", type=str, required=True, help="Comma-separated parameter values")
    parser.add_argument("--output", type=str, default="print", help="Output format")
    return ["tri_model_path", "tri_max_length", "tri_device", "parameters", "output"]


def load_config(config_path: str) -> Dict[str, Any]:
    import yaml
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def parse_parameters(param_str: str, param_type: str = "float") -> List[Any]:
    values = param_str.split(",")
    if param_type == "int":
        return [int(v.strip()) for v in values]
    elif param_type == "float":
        return [float(v.strip()) for v in values]
    else:
        return [v.strip() for v in values]


def main():
    parser = argparse.ArgumentParser(description="Run reidentification risk experiment")
    
    data_keys = add_data_args(parser)
    model_keys = add_model_args(parser)
    experiment_keys = add_experiment_args(parser)
    
    args = parser.parse_args()
    
    data_kwargs = {k: getattr(args, k) for k in data_keys}
    model_kwargs = {k: getattr(args, k) for k in model_keys}
    experiment_kwargs = {k: getattr(args, k) for k in experiment_keys}
    
    dataset = load_data_from_config(args.data, data_kwargs.get("data_in"), data_kwargs.get("max_records"))
    
    train_split = dataset.get("train", None)
    test_split = dataset.get("test", None)
    
    if train_split is None or test_split is None:
        raise ValueError("Dataset must have train and test splits")
    
    model_config = load_config(args.model_in)
    
    model_class = MODEL_REGISTRY[args.model]
    param_values_str = experiment_kwargs["parameters"]
    
    if is_k_anonymizer(model_class):
        parameter_values = parse_parameters(param_values_str, "int")
    elif is_dp_anonymizer(model_class):
        parameter_values = parse_parameters(param_values_str, "float")
    else:
        parameter_values = parse_parameters(param_values_str, "str")
    
    experiment = ReidentificationRiskExperiment(
        dataset_records=test_split,
        train_records=train_split,
        model_class_name=args.model,
        model_config=model_config,
        tri_model_path=experiment_kwargs.get("tri_model_path"),
        tri_max_length=experiment_kwargs.get("tri_max_length", 512),
        tri_device=experiment_kwargs.get("tri_device", "auto"),
        dataset_name=args.data,
    )
    
    result = experiment.execute(parameter_values=parameter_values)
    
    output_format = experiment_kwargs.get("output", "print")
    
    if output_format == "print":
        print("\n" + "="*80)
        print(f"Re-identification Risk Experiment Results")
        print("="*80)
        print(f"Model: {args.model}")
        print(f"Dataset: {args.data}")
        print(f"Number of records: {len(test_split)}")
        print(f"Parameters tested: {parameter_values}")
        print(f"\nAverage privacy score: {result.metrics['average_privacy_score']:.4f}")
        print("\nResults per parameter:")
        for res in result.metrics["results_per_parameter"]:
            print(f"  Parameter {res['parameter']}: privacy_score={res['privacy_score']:.4f}")
        print("="*80 + "\n")
    elif output_format == "json":
        import json
        output_data = {
            "model": args.model,
            "dataset": args.data,
            "num_records": len(test_split),
            "parameters": parameter_values,
            "average_privacy_score": result.metrics["average_privacy_score"],
            "results": result.metrics["results_per_parameter"],
        }
        print(json.dumps(output_data, indent=2))


if __name__ == "__main__":
    main()
