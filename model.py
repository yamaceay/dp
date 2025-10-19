from typing import Optional, List, Tuple
import argparse
import yaml

from dp.methods.anonymizer import Anonymizer
from dp.methods.registry import MODEL_REGISTRY
from dp.loaders import ADAPTER_REGISTRY, DatasetRecord

from dp.utils.pii_detector import PIIDetector
from dp.utils.selector.pii_only_selector import PIIOnlySelector
from dp.utils.explainer import UniformExplainer, GreedyExplainer, ShapExplainer
from dp.utils.chunking import TruncateChunker, SlidingWindowChunker, TokenAwareChunker

available_models = list(MODEL_REGISTRY.keys())
available_datasets = list(ADAPTER_REGISTRY.keys())

def add_data_args(parser: argparse.ArgumentParser) -> Tuple[argparse.ArgumentParser, List[str]]:
    parser.add_argument('--data', type=str, required=True, choices=available_datasets, help='Dataset name (trustpilot, tab, db_bio)')
    parser.add_argument('--data_in', type=str, required=True, help='Path to input data file or directory')
    parser.add_argument('--max_records', type=int, default=1, help='Maximum number of records to load')
    return parser, ['data', 'data_in', 'max_records']

def add_model_args(parser: argparse.ArgumentParser) -> Tuple[argparse.ArgumentParser, List[str]]:
    parser.add_argument('--model', type=str, required=True, choices=available_models, help='Anonymization model/method to evaluate')
    parser.add_argument('--model_in', type=str, default=None, help='Path to the method configuration')
    return parser, ['model', 'model_in']

def add_runtime_args(parser: argparse.ArgumentParser) -> Tuple[argparse.ArgumentParser, List[str]]:
    parser.add_argument('--runtime_in', type=str, default=None, help='Path to the runtime configuration')
    parser.add_argument('--text', type=str, default="Kurt Edward Fishback is an American photographer noted for his portraits of other artists and photographers. Kurt was born in Sacramento, CA in 1942. Son of photographer Glen Fishback and namesake of photographer Edward Weston, he was exposed to art photography at an early age as his father's friends included Edward Weston, Ansel Adams and Wynn Bullock. Kurt studied art at Sacramento City College, SFAI, Cornell University and UC Davis where he received his Master of Fine Arts Degree studying with Robert Arneson, Roy DeForest, William Wiley and Manuel Neri. Ceramic Sculpture was the first medium that gained him high visibility in the Art World. Kurt took up photography in 1962 when he asked his Father to teach him. After finishing graduate work and teaching fine art media at several colleges, Kurt was asked to teach at his father's school of photography in Sacramento. The series of artist portraits which now number over 250 were begun in 1979. Since 1963 Kurt has been involved in many solo and group exhibitions including; SFMOMA, and Crocker Art Museum. His work is represented in many public, private and corporate collections including; SFMOMA, SFAI, and Museum of Contemporary Crafts, New York, NY. Today, Kurt lives in Sacramento, California with his wife Cassandra Reeves. He exhibits at galleries and museums, teaches photography at American River College, and has published several books including a book of portraits of California artists entitled, Art in Residence: West Coast Artists in Their Space (see illustration). The book includes portraits of 74 artists, including Ansel Adams, Wayne Thiebaud, Judy Chicago, Brett Weston, and Jock Sturges. Other artist portraits made by Kurt include Cornell Capa, André Kertész, Mary Ellen Mark, Chuck Close and Robert Mapplethorpe. Kurt is represented by Appel Photography Gallery in Sacramento, CA and The Camera Obscura Gallery in Denver, CO.", help='Text to anonymize')
    parser.add_argument('--idx', type=int, default=0, help='Index of the record to anonymize (for datasets with multiple records)')
    return parser, ['runtime_in', 'text', 'idx']

def load_config(sth_in: Optional[str]) -> dict:
    config = {}
    if sth_in is not None:
        with open(sth_in, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    return config

def load_data(data_kwargs: Optional[dict]) -> List[DatasetRecord]:
    data = data_kwargs.get("data")
    adapter = ADAPTER_REGISTRY.get(data)
    if adapter is None:
        raise ValueError(f"Adapter '{data}' not found.")

    dataset = adapter(**data_kwargs)
    return dataset

def load_model(model_config: Optional[dict], model_kwargs: Optional[dict], data_kwargs: Optional[dict], dataset: Optional[List[DatasetRecord]] = None) -> Anonymizer:
    model = model_kwargs.get("model")
    model_cls = MODEL_REGISTRY.get(model)
    if model_cls is None:
        raise ValueError(f"Model '{model}' not found.")

    if model in ["petre"]:
        if dataset is None:
            raise ValueError("PETRE requires dataset to be loaded")
        model_instance = model_cls(dataset_records=list(dataset.iter_records()), **model_config, **model_kwargs)
    else:
        model_instance = model_cls(**model_config, **model_kwargs, **data_kwargs)
    
    return model_instance

def use_idx(model_config: dict, data_kwargs: dict, length: int) -> bool:
    if "requires_idx" not in model_config or not model_config["requires_idx"]:
        return False
    length = min(length, data_kwargs.get("max_records", 0))
    idx = data_kwargs.get("idx", 0)
    if idx < 0 or idx >= length:
        raise ValueError(f"Index {idx} is out of bounds for dataset of length {length}.")
    return True
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load and Evaluate Anonymization Models")
    parser, data_keys = add_data_args(parser)
    parser, model_keys = add_model_args(parser)
    parser, runtime_keys = add_runtime_args(parser)
    
    args = parser.parse_args()
    data_kwargs = {k: getattr(args, k) for k in data_keys}
    model_kwargs = {k: getattr(args, k) for k in model_keys}
    runtime_kwargs = {k: getattr(args, k) for k in runtime_keys}

    dataset = load_data(data_kwargs)

    model_config = load_config(args.model_in)
    
    pii_chunking = model_config.get("pii_chunking", {})
    tri_chunking = model_config.get("tri_chunking", {})
    dpmlm_chunking = model_config.get("dpmlm_chunking", {})

    if args.model in ["petre"]:
        starting_annotations = None
        starting_annotations_path = model_config.get("starting_annotations", None)
        if starting_annotations_path:
            import json
            with open(starting_annotations_path, 'r', encoding='utf-8') as f:
                starting_annotations = json.load(f)
        model_config["starting_annotations"] = starting_annotations
    
    model = load_model(model_config, model_kwargs, data_kwargs, dataset)

    if args.model in ["dpmlm"]:
        pii_annotator_path = model_config.get("pii_annotator", None)
        threshold = model_config.get("pii_threshold", None)
        pii_use_chunking = pii_chunking.get("enabled", False)
        if pii_annotator_path is not None:
            pii_annotator = PIIDetector(model_name=pii_annotator_path, use_chunking=pii_use_chunking)
            if pii_use_chunking:
                pii_max_length = pii_chunking.get("max_length", 512)
                pii_overlap = pii_chunking.get("overlap", 50)
                pii_annotator.chunker = SlidingWindowChunker(max_length=pii_max_length, overlap=pii_overlap)
            selector = PIIOnlySelector(pii_detector=pii_annotator, threshold=threshold)
            model.set_filtering_strategy(selector)

    if args.model in ["dpmlm", "petre"]:
        explainer_path = model_config.get("explainer_path", None)
        
        explainability = model_config.get("explainability", None)
        tri_use_chunking = tri_chunking.get("enabled", False)
        
        if args.model == "petre":
            if explainability is None or explainability == "uniform":
                raise ValueError("PETRE requires explainability to be 'greedy' or 'shap', not 'uniform'")
            if explainer_path is None:
                raise ValueError("PETRE requires explainer_path to be set")
        
        if explainability is None:
            explainability = "uniform"
        
        if explainability == "uniform":
            explainer = UniformExplainer()
            model.set_scoring_strategy(explainer)
        elif explainer_path is not None:
            if explainability == "greedy":
                explainer = GreedyExplainer(model_name=explainer_path, use_chunking=tri_use_chunking)
            elif explainability == "shap":
                explainer = ShapExplainer(model_name=explainer_path, use_chunking=tri_use_chunking)
            else:
                raise ValueError(f"Unknown explainability method: {explainability}")
            if tri_use_chunking:
                tri_max_length = tri_chunking.get("max_length", 512)
                explainer.tri_detector.chunker = TruncateChunker(max_length=tri_max_length)
            model.set_scoring_strategy(explainer)

    runtime_config = load_config(args.runtime_in)
    if use_idx(model_config, data_kwargs, len(dataset)):
        result = model.anonymize_from_dataset(idx=args.idx, **runtime_config)
    else:
        result = model.anonymize(text=args.text, **runtime_config)

    print("Anonymized Text:", result.text)
    print("Annotations:", result.spans)
    print("Metadata:", result.metadata)
    