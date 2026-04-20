from __future__ import annotations

import re
from pathlib import Path
from typing import Any


def parse_params_from_key(key: str) -> tuple[str, dict[str, Any]]:
    params: dict[str, Any] = {}
    method = key
    is_dpmlm_shap_no_delim = False
    if "?" in key:
        method, params_str = key.split("?", 1)
        for param in params_str.split("&"):
            if "=" not in param:
                raise ValueError(f"Unexpected hyperparameter format: {param}")
            name, value = param.split("=", 1)
            if method == "dpmlm_shap" and name in {"k", "rho", "lambda"}:
                is_dpmlm_shap_no_delim = True
            if name in {"epsilon", "k"}:
                value = int(value)
            elif name in {"rho", "lambda"}:
                value = float(value) / 100.0
            params[name] = value
    params = dict(sorted(params.items(), key=lambda item: item[0]))
    if not is_dpmlm_shap_no_delim:
        if method == "dpmlm_shap":
            method = "dpmlm_shap_no_presidio"
        elif method == "dpmlm_shap_masked":
            method = "dpmlm_shap"
    return method, params


def normalize_source_key(source_path: str, dataset: str) -> str:
    name = Path(source_path).name
    if name.endswith(".jsonl"):
        name = name[:-6]
    key = re.sub(rf"^\d{{8}}_\d{{6}}_{re.escape(dataset)}_", "", name)
    method, sep, params = key.partition("?")
    method = re.sub(r"_eps_[0-9]{3}$", "", method)
    method = re.sub(r"(?:_k|_risk|_pii)$", "", method)
    return method + ((sep + params) if sep else "")


def parse_method_and_params_from_source_path(source_path: str, dataset: str) -> tuple[str, dict[str, Any]]:
    return parse_params_from_key(normalize_source_key(source_path, dataset))