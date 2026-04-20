from typing import Any, Dict, List

from dp.loaders.base import DatasetRecord

def functional_results(source: Dict[Any, Dict[Any, int]]) -> bool:
    maps_to_one_target = True
    maps_to_multiple_targets = True

    for _, tgt_dict in source.items():
        if len(tgt_dict) > 1:
            maps_to_one_target = False
        if len(tgt_dict) == 0:
            maps_to_multiple_targets = False

    return {
        "maps_to_one_target": maps_to_one_target,
        "maps_to_multiple_targets": maps_to_multiple_targets,
    }

def is_function(source: Dict[Any, Dict[Any, int]], target: Dict[Any, Dict[Any, int]]) -> bool:
    functional_results_source = functional_results(source)
    functional_results_target = functional_results(target)
    return functional_results_source["maps_to_one_target"] and functional_results_target["maps_to_multiple_targets"]

class FunctionalRegistry:
    def __init__(self) -> None:
        self.functionals: List[Functional] = []
        self._functionals_set: set = set()

    def add_functional(self, functional: 'Functional') -> None:
        self.functionals.append(functional)
        self._functionals_set.add((functional.source, functional.target))
    
    def is_functional_registered(self, functional: 'Functional') -> bool:
        return (functional.source, functional.target) in self._functionals_set

    def get_functionals(self) -> List['Functional']:
        return self.functionals

class Functional:
    def __init__(self, source: str, target: str) -> None:
        self.source = source
        self.target = target
        self.by_source: Dict[Any, Dict[Any, int]] = {}
    
    def add(self, by_source: Dict[Any, Dict[Any, int]]) -> None:
        self.by_source = by_source

    def __repr__(self) -> str:
        result = f"{self.source} -> {self.target}\n"
        result += "-"*(len(result)-1) + "\n"
        if not self.by_source:
            return result
        for src_key, tgt_dict in self.by_source.items():
            tgt_key = next(iter(tgt_dict))
            result += f"  {src_key} -> {tgt_key}\n"
        return result
    
    def show(self, head_n: int = 10, tail_n: int = 10) -> str:
        result = f"{self.source} -> {self.target}\n"
        result += "-"*(len(result)-1) + "\n"
        if not self.by_source:
            return result
        total_items = len(self.by_source)
        if total_items <= head_n + tail_n:
            for src_key, tgt_dict in self.by_source.items():
                tgt_key = next(iter(tgt_dict))
                cnt = tgt_dict[tgt_key]
                result += f"  {src_key} -> {tgt_key} : {cnt}\n"
        else:
            items = list(self.by_source.items())
            for idx, (src_key, tgt_dict) in enumerate(items):
                if idx < head_n or idx >= total_items - tail_n:
                    tgt_key = next(iter(tgt_dict))
                    cnt = tgt_dict[tgt_key]
                    result += f"  {src_key} -> {tgt_key} : {cnt}\n"
                elif idx == head_n:
                    result += f"  ... ({total_items - head_n - tail_n} more items) ...\n"
        return result

class FunctionalAnalysis:
    def __init__(self, records: List[DatasetRecord], value_getters, exclude_keys: List[str] = []) -> None:
        self.data: Dict[str, List[Any]] = {}
        for record in records:
            for key, getter in value_getters.items():
                self.data.setdefault(key, []).append(getter(record))
        self.data_keys = list(self.data.keys())
        self.exclude_keys = exclude_keys
        if self.exclude_keys:
            self.data_keys = [k for k in self.data_keys if k not in self.exclude_keys]
        self.functional_mappings = []

    def analyze(self) -> None:
        for i in range(len(self.data_keys)):
            for j in range(i + 1, len(self.data_keys)):
                by_source, by_target = {}, {}
                source, target = self.data_keys[i], self.data_keys[j]
                if len(self.data[source]) != len(self.data[target]):
                    raise ValueError(f"Source and target data length mismatch: {source} ({len(self.data[source])}) vs {target} ({len(self.data[target])})")
                for k in range(len(self.data[source])):
                    src_value = self.data[source][k]
                    tgt_value = self.data[target][k]
                    if src_value not in by_source:
                        by_source[src_value] = {}
                    if tgt_value not in by_source[src_value]:
                        by_source[src_value][tgt_value] = 0
                    by_source[src_value][tgt_value] += 1
                    if tgt_value not in by_target:
                        by_target[tgt_value] = {}
                    if src_value not in by_target[tgt_value]:
                        by_target[tgt_value][src_value] = 0
                    by_target[tgt_value][src_value] += 1
                if is_function(by_source, by_target):
                    functional = Functional(source, target)
                    functional.add(by_source)
                    self.functional_mappings.append(functional)

    def dag(self) -> List[Functional]:
        functional_registry = FunctionalRegistry()
        for functional in self.functional_mappings:
            functional_registry.add_functional(functional)
        for _ in range(len(self.functional_mappings)):
            all_registered = True
            for f1 in functional_registry.get_functionals():
                for f2 in functional_registry.get_functionals():
                    if f1.target == f2.source:
                        new_by_source = {}
                        for src_key, mid_dict in f1.by_source.items():
                            for mid_key, count in mid_dict.items():
                                if mid_key in f2.by_source:
                                    for tgt_key, tgt_count in f2.by_source[mid_key].items():
                                        if src_key not in new_by_source:
                                            new_by_source[src_key] = {}
                                        if tgt_key not in new_by_source[src_key]:
                                            new_by_source[src_key][tgt_key] = 0
                                        new_by_source[src_key][tgt_key] += count * tgt_count
                        new_functional = Functional(f1.source, f2.target)
                        new_functional.add(new_by_source)
                        if not functional_registry.is_functional_registered(new_functional):
                            functional_registry.add_functional(new_functional)
                            all_registered = False
            if all_registered:
                break
            
        return functional_registry.get_functionals()

    def print_results(self) -> List[Functional]:
        for functional in self.functional_mappings:
            print(functional)