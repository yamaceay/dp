from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

from dp.loaders.reddit import RedditDatasetAdapter
from dp.utils.explainer import ShapExplainer, ShapType
from dp.loaders.derive import get_getter

sort_order = {
    "age": ["age: 18-29", "age: 30-44", "age: 45-59", "age: 60+"],
    "income_level": ["income: low", "income: medium", "income: high"],
    "education": ["education: secondary", "education: studying", "education: bachelor", "education: master", "education: doctorate"],
}

class UtilityShapConfig:
    def __init__(self):
        self.model_path: str = ""
        self.data: str = ""
        self.data_path: str = ""
        self.feature: str = ""
        self.sample_start: int = 0
        self.sample_end: int = 10
        self.sample_step: int = 1
        self.top_k: int = 5
        self.explainer_type: ShapType = ShapType.DEFAULT
    
    def set_model_path(self, path: str) -> UtilityShapConfig:
        self.model_path = path
        return self
    
    def set_data(self, data: str) -> UtilityShapConfig:
        self.data = data
        return self

    def set_data_path(self, path: str) -> UtilityShapConfig:
        self.data_path = path
        return self
    
    def set_feature(self, feature: str) -> UtilityShapConfig:
        self.feature = feature
        return self
    
    def set_sample_start(self, start: int) -> UtilityShapConfig:
        self.sample_start = start
        return self
    
    def set_sample_end(self, end: int) -> UtilityShapConfig:
        self.sample_end = end
        return self
    
    def set_sample_step(self, step: int) -> UtilityShapConfig:
        self.sample_step = step
        return self
    
    def set_top_k(self, top_k: int) -> UtilityShapConfig:
        self.top_k = top_k
        return self
    
    def set_explainer_type(self, explainer_type: ShapType) -> UtilityShapConfig:
        self.explainer_type = explainer_type
        return self
    
    def validate(self) -> None:
        if not self.model_path:
            raise ValueError("model_path is required")
        if not self.data_path:
            raise ValueError("data_path is required")
        if not self.feature:
            raise ValueError("feature is required")
        if not Path(self.model_path).exists():
            raise ValueError(f"model path does not exist: {self.model_path}")
        if not Path(self.data_path).exists():
            raise ValueError(f"data path does not exist: {self.data_path}")


class TokenSpanExtractor:
    def __init__(self, model_path: str):
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    def extract(self, text: str) -> List[Tuple[int, int]]:
        encoding = self.tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
        offsets = encoding["offset_mapping"]
        return [(int(start), int(end)) for start, end in offsets if start < end]


class DatasetSampler:
    def __init__(self, data: str, data_path: str, feature: str):
        self.adapter = RedditDatasetAdapter(data_in=data_path)
        self.feature = feature
        self.getter = get_getter(data, "feature_label")
    
    def sample(self, start: int, end: int, step: int) -> List[Tuple[str, str, Dict]]:
        candidates: List[Tuple[str, str, Dict]] = []
        for record in self.adapter.iter_records():
            if record.metadata.get("feature") != self.feature:
                continue
            text = record.text
            feature_label = self.getter(record)
            if not text or feature_label is None:
                continue
            candidates.append((text, str(feature_label), dict(record.metadata)))
        return candidates[start:end:step]


class ShapAnalyzer:
    def __init__(self, config: UtilityShapConfig):
        self.config = config
        print(f"[init] creating explainer for model={config.model_path}")
        self.explainer = ShapExplainer(
            model_name=config.model_path,
            device=config.tri_device,
            explainer_type=config.explainer_type
        )
        print(f"[init] explainer type={config.explainer_type.value}")
        self.span_extractor = TokenSpanExtractor(config.model_path)
        self.sampler = DatasetSampler(config.data, config.data_path, config.feature)
    
    def get_label_mappings(self) -> Tuple[Dict[str, int], Dict[int, str]]:
        if not self.explainer.pipeline:
            self.explainer._load_pipeline()
        self.explainer._ensure_tri_mapping()
        
        if not self.explainer.label_to_id or all(k.startswith("LABEL_") for k in self.explainer.label_to_id.keys()):
            from transformers import AutoConfig
            config = AutoConfig.from_pretrained(self.config.model_path)
            if hasattr(config, "id2label") and config.id2label:
                self.explainer.id_to_label = dict(config.id2label)
                self.explainer.label_to_id = {label: idx for idx, label in config.id2label.items()}
                print(f"[mappings] loaded from config.id2label: {self.explainer.id_to_label}")
        
        return self.explainer.label_to_id, self.explainer.id_to_label
    
    def analyze(self) -> List[Dict]:
        print(f"[analyze] sampling records [{self.config.sample_start}:{self.config.sample_end}:{self.config.sample_step}] for feature={self.config.feature}")
        samples = self.sampler.sample(self.config.sample_start, self.config.sample_end, self.config.sample_step)
        if not samples:
            raise ValueError(f"no samples found for feature={self.config.feature}")
        print(f"[analyze] found {len(samples)} samples")
        
        label_to_id, id_to_label = self.get_label_mappings()
        print(f"[analyze] label mappings: {len(label_to_id)} labels")
        print(f"[analyze] label_to_id={label_to_id}")
        print(f"[analyze] id_to_label={id_to_label}")
        
        results: List[Dict] = []
        for idx, (text, label, metadata) in enumerate(samples):
            print(f"[{idx+1}/{len(samples)}] analyzing: {text[:50]}...")
            print(f"  text_length={len(text)} label={label}")
            
            spans = self.span_extractor.extract(text)
            print(f"  extracted {len(spans)} token spans")
            if not spans:
                print(f"  no tokens extracted, skipping")
                continue
            
            print(f"  computing shap values for target_label={label}")
            target_label = sort_order.get(self.config.feature).index(label)
            try:
                shap_values = self.explainer.explain(text, spans, target_label=target_label)
                print(f"  shap_values shape={shap_values.shape} min={shap_values.min():.4f} max={shap_values.max():.4f}")
            except Exception as exc:
                print(f"  failed: {exc}")
                continue
            
            tokens = [text[s:e] for s, e in spans]
            token_importance = [
                {"token": tok, "span": span, "importance": float(val)}
                for tok, span, val in zip(tokens, spans, shap_values)
            ]
            token_importance.sort(key=lambda x: abs(x["importance"]), reverse=True)
            
            top_tokens = token_importance[:self.config.top_k]
            results.append({
                "text": text,
                "label": label,
                "metadata": metadata,
                "tokens": token_importance,
                "top_tokens": top_tokens,
            })
            
            print(f"  label={label}, top tokens:")
            for item in top_tokens:
                print(f"    {item['token']!r}: {item['importance']:.4f}")
        
        return results


def main() -> None:
    print("[main] starting utility shap analysis")
    config = UtilityShapConfig()
    config.set_model_path("models/tri_pipelines/reddit/utility_age")
    config.set_data("reddit")
    config.set_data_path("data/reddit/test.jsonl")
    config.set_feature("age")
    config.set_sample_start(0)
    config.set_sample_end(5)
    config.set_sample_step(1)
    config.set_top_k(5)
    config.set_explainer_type(ShapType.DEFAULT)
    print(f"[main] config: feature={config.feature} samples=[{config.sample_start}:{config.sample_end}:{config.sample_step}] top_k={config.top_k}")
    config.validate()
    print(f"[main] config validated")
    
    analyzer = ShapAnalyzer(config)
    results = analyzer.analyze()
    
    print(f"\n[summary] analyzed {len(results)} samples")
    print(f"[summary] feature: {config.feature}")
    print(f"[summary] model: {config.model_path}")
    if results:
        avg_tokens = sum(len(r["tokens"]) for r in results) / len(results)
        print(f"[summary] avg tokens per sample: {avg_tokens:.1f}")


if __name__ == "__main__":
    main()
