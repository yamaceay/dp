from dataclasses import dataclass
from typing import Dict, List, Union, Any
from itertools import product

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

class BucketDict(dict):
    def encode(self) -> str:
        parts: List[str] = []
        for key, val in self.items():
            if key == "epsilon":
                num = int(val)
                parts.append(f"epsilon={num:04d}")
            elif key in ("rho", "lambda"):
                num = int(round(float(val) * 100))
                parts.append(f"{key}={num:03d}")
            elif key in ("k", "ks"):
                values: List[int] = val if isinstance(val, list) else [int(val)]
                encoded = ",".join(f"{v:03d}" for v in values)
                parts.append(f"{key}={encoded}")
            else:
                parts.append(f"{key}={val}")
        return "?" + "&".join(parts) if parts else ""

def buckets_to_dicts(buckets: Buckets) -> List[BucketDict]:
    if not buckets:
        return [BucketDict()]

    value_sets: List[Dict[str, List[Any]]] = []
    singletons: Dict[str, Any] = {}

    for b in buckets:
        if isinstance(b, KParams):
            value_sets.append({b.name: list(b.values())})
        elif isinstance(b, RhoParams):
            value_sets.append({b.name: list(b.values())})
        elif isinstance(b, LambdaParams):
            value_sets.append({b.name: list(b.values())})
        elif isinstance(b, EpsilonParam):
            singletons[b.name] = b.value()

    combos: List[BucketDict] = []
    if value_sets:
        keys = [list(d.keys())[0] for d in value_sets]
        value_lists = [list(d.values())[0] for d in value_sets]
        for values in product(*value_lists):
            hp = BucketDict()
            hp.update(singletons)
            for key, val in zip(keys, values):
                hp[key] = val
            combos.append(hp)
    else:
        hp = BucketDict()
        hp.update(singletons)
        combos.append(hp)

    return combos