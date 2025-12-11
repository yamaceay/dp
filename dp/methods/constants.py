from dataclasses import dataclass
from typing import Dict, List, Union

PII_CLASSIFIER_MODEL_LIST: List[str] = ["baroud"]
RISK_MASKER_MODEL_LIST: List[str] = ["risk"]

@dataclass
class ModelCapabilities:
    can_work_token_level: bool = True

    must_use_dataset: bool = False
    can_use_dataset: bool = False
    must_use_pii_selector: bool = False
    can_use_pii_selector: bool = False
    must_use_risk_selector: bool = False
    can_use_risk_selector: bool = False
    must_use_k_selector: bool = False
    can_use_k_selector: bool = False
    must_use_annotations: bool = False
    can_use_annotations: bool = False
    must_use_scoring: bool = False
    can_use_scoring: bool = False

@dataclass
class MultiParams:
    def values(self):
        pass

@dataclass
class KParams(MultiParams):
    ks: List[int]
    name: str = "k"

    def values(self):
        return sorted(self.ks)
    
@dataclass
class RhoParams(MultiParams):
    rhos: List[float]
    name: str = "rho"

    def values(self):
        return reversed(sorted(self.rhos))
    
@dataclass
class LambdaParams(MultiParams):
    lambdas: List[float]
    name: str = "lambda"

    def values(self):
        return reversed(sorted(self.lambdas))

@dataclass
class SingleParam:
    def value(self):
        pass

@dataclass
class EpsilonParam(SingleParam):
    epsilon: float
    name: str = "epsilon"

    def value(self):
        return self.epsilon

Buckets = List[Union[MultiParams, SingleParam]]