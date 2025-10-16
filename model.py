from typing import Optional
import yaml
import argparse

from data import load_data, add_data_args

from methods.anonymizer import Anonymizer
from methods.registry import MODEL_REGISTRY

available_models = list(MODEL_REGISTRY.keys())

def add_model_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument('--model', type=str, required=True, choices=available_models, help='Anonymization model/method to evaluate')
    parser.add_argument('--model_in', type=str, default=None, help='Path to the method configuration')
    return parser

def add_runtime_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument('--runtime_in', type=str, default=None, help='Path to the runtime configuration')
    parser.add_argument('--text', type=str, default="Kurt Edward Fishback is an American photographer noted for his portraits of other artists and photographers. Kurt was born in Sacramento, CA in 1942. Son of photographer Glen Fishback and namesake of photographer Edward Weston, he was exposed to art photography at an early age as his father's friends included Edward Weston, Ansel Adams and Wynn Bullock. Kurt studied art at Sacramento City College, SFAI, Cornell University and UC Davis where he received his Master of Fine Arts Degree studying with Robert Arneson, Roy DeForest, William Wiley and Manuel Neri. Ceramic Sculpture was the first medium that gained him high visibility in the Art World. Kurt took up photography in 1962 when he asked his Father to teach him. After finishing graduate work and teaching fine art media at several colleges, Kurt was asked to teach at his father's school of photography in Sacramento. The series of artist portraits which now number over 250 were begun in 1979. Since 1963 Kurt has been involved in many solo and group exhibitions including; SFMOMA, and Crocker Art Museum. His work is represented in many public, private and corporate collections including; SFMOMA, SFAI, and Museum of Contemporary Crafts, New York, NY. Today, Kurt lives in Sacramento, California with his wife Cassandra Reeves. He exhibits at galleries and museums, teaches photography at American River College, and has published several books including a book of portraits of California artists entitled, Art in Residence: West Coast Artists in Their Space (see illustration). The book includes portraits of 74 artists, including Ansel Adams, Wayne Thiebaud, Judy Chicago, Brett Weston, and Jock Sturges. Other artist portraits made by Kurt include Cornell Capa, André Kertész, Mary Ellen Mark, Chuck Close and Robert Mapplethorpe. Kurt is represented by Appel Photography Gallery in Sacramento, CA and The Camera Obscura Gallery in Denver, CO.", help='Text to anonymize')
    return parser

def load_config(sth_in: Optional[str]) -> dict:
    config = {}
    if sth_in is not None:
        with open(sth_in, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    return config

def load_model(model: str, config: Optional[dict]) -> Anonymizer:
    model_cls = MODEL_REGISTRY.get(model)
    if model_cls is None:
        raise ValueError(f"Model '{model}' not found.")
    model = model_cls(**config)
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load and Evaluate Anonymization Models")
    parser = add_data_args(parser)
    parser = add_model_args(parser)
    parser = add_runtime_args(parser)
    args = parser.parse_args()

    dataset = load_data(args.data, args.data_in, args.max_records)

    model_config = load_config(args.model_in)
    runtime_config = load_config(args.runtime_in)

    model = load_model(args.model, model_config)

    result = model.anonymize(args.text, **runtime_config)
    
    print("Anonymized Text:", result.text)
    print("Metadata:", result.metadata)
    