from __future__ import annotations

from dp.tri.loaders.tab import TabAttackerDatasetAdapter
from dp.tri.loaders.tab_longformer import TabLongformerAttackerDatasetAdapter
from dp.tri.loaders.reddit import RedditAttackerDatasetAdapter
from dp.tri.loaders.trustpilot import TrustpilotAttackerDatasetAdapter
from dp.tri.loaders.base import (
    AttackerDatasetAdapter, 
    RewriterProtocol, 
    save_attacker_extensions_jsonl, 
    load_attacker_extensions_jsonl,
)

ATTACKER_ADAPTER_REGISTRY: dict[str, type[AttackerDatasetAdapter]] = {
    "tab": TabAttackerDatasetAdapter,
    "tab_longformer": TabLongformerAttackerDatasetAdapter,
    "reddit": RedditAttackerDatasetAdapter,
    "trustpilot": TrustpilotAttackerDatasetAdapter,
}

def get_attacker_adapter(name: str, **kwargs) -> AttackerDatasetAdapter:
    """Instantiate an attacker dataset adapter by name."""
    key = (name or "").lower()
    if key not in ATTACKER_ADAPTER_REGISTRY:
        raise ValueError(
            f"Unknown dataset adapter '{name}'. "
            f"Available adapters: {sorted(ATTACKER_ADAPTER_REGISTRY.keys())}"
        )
    adapter_cls = ATTACKER_ADAPTER_REGISTRY[key]
    return adapter_cls(**kwargs)

__all__ = [
    "RewriterProtocol",
    "AttackerDatasetAdapter",
    "TabAttackerDatasetAdapter",
    "TabLongformerAttackerDatasetAdapter",
    "RedditAttackerDatasetAdapter",
    "TrustpilotAttackerDatasetAdapter",
    "save_attacker_extensions_jsonl",
    "load_attacker_extensions_jsonl",
]
