from __future__ import annotations

from dp.tri.loaders.tab import TabAttackerDatasetAdapter
from dp.tri.loaders.dbbio import DBBioAttackerDatasetAdapter
from dp.tri.loaders.yelp import YelpAttackerDatasetAdapter
from dp.tri.loaders._ratbench import RatBenchAttackerDatasetAdapter, RatBenchNoBartAttackerDatasetAdapter
from dp.tri.loaders.base import (
    AttackerDatasetRecord,
    AttackerDatasetAdapter,
    RewriterProtocol,
    save_attacker_extensions_jsonl,
    load_attacker_extensions_jsonl,
)

ATTACKER_ADAPTER_REGISTRY: dict[str, type[AttackerDatasetAdapter]] = {
    "tab": TabAttackerDatasetAdapter,
    "db_bio": DBBioAttackerDatasetAdapter,
    "yelp": YelpAttackerDatasetAdapter,
    "rat_bench": RatBenchAttackerDatasetAdapter,
    "rat_bench_nobart": RatBenchNoBartAttackerDatasetAdapter,
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
    "AttackerDatasetRecord",
    "RewriterProtocol",
    "AttackerDatasetAdapter",
    "save_attacker_extensions_jsonl",
    "load_attacker_extensions_jsonl",
]
