from typing import List, Dict, Optional, Tuple
import os
import re
import torch
import numpy as np
from collections import OrderedDict
from transformers import pipeline

from dp.methods.anonymizer import AnonymizationResult
from dp.methods.k_anon import KAnonymizer
from dp.loaders.base import DatasetRecord, TextAnnotation
from dp.loaders.annotations import read_annotations, annotations_to_spans, spans_to_annotations


class PetreAnonymizer(KAnonymizer):
    def __init__(
        self,
        dataset_records: List[DatasetRecord],
        starting_annotations: Optional[Dict[str, List[TextAnnotation]]] = None,
        mask_text: str = "[MASK]",
        device: str = "auto",
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        
        self.dataset_records = dataset_records
        self.mask_text = mask_text
        
        self.device = self._resolve_device(device)
        self.explainer = None
        self.tri_pipeline_path = None
        
        self._build_label_mappings()
        self._initialize_annotations(starting_annotations)
        
        self._k_cache: Dict[int, bool] = {}
        self._annotation_history: Dict[int, Dict[str, List[TextAnnotation]]] = {}
        
        print(f"✓ PetreAnonymizer initialized with {self.num_labels} individuals")
    
    def _resolve_device(self, device: str) -> torch.device:
        if device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        return torch.device(device)
    
    def _build_label_mappings(self):
        names = sorted(set(r.name for r in self.dataset_records))
        self.label_to_name = {idx: name for idx, name in enumerate(names)}
        self.name_to_label = {name: idx for idx, name in self.label_to_name.items()}
        self.num_labels = len(names)
    
    def _load_tri_pipeline(self):
        self.tri_pipeline = pipeline(
            "text-classification",
            model=self.tri_pipeline_path,
            tokenizer=self.tri_pipeline_path,
            device=self.device,
            top_k=self.num_labels,
            max_length=512,
            truncation=True,
        )
    
    def _initialize_annotations(self, starting_annotations: Optional[Dict[str, List[TextAnnotation]]] = None):
        self.annotations: Dict[str, List[TextAnnotation]] = {}
        
        for record in self.dataset_records:
            uid = record.uid
            if starting_annotations and uid in starting_annotations:
                self.annotations[uid] = starting_annotations[uid]
            else:
                self.annotations[uid] = []
    
    def set_scoring_strategy(self, explainer):
        self.explainer = explainer
        self.tri_pipeline_path = explainer.tri_detector.model_name
        self._load_tri_pipeline()
    
    def set_annotations(self, annotations: Dict[str, List[TextAnnotation]]):
        """
        Set annotations from external source (e.g., loaded from file).
        This replaces the current annotations and resets the cache.
        """
        self.annotations = {}
        for record in self.dataset_records:
            uid = record.uid
            if annotations and uid in annotations:
                self.annotations[uid] = annotations[uid]
            else:
                self.annotations[uid] = []
        
        self._k_cache: Dict[int, bool] = {}
        self._annotation_history: Dict[int, Dict[str, List[TextAnnotation]]] = {}
        
        print(f"✓ Set annotations for {len([a for a in self.annotations.values() if a])} records")
    
    def anonymize(self, text: str, *args, **kwargs) -> AnonymizationResult:
        raise NotImplementedError("Use anonymize_from_dataset for PetreAnonymizer.")
    
    def grid_anonymize_from_dataset(self, idx: int, k: List[int], *args, **kwargs) -> List[AnonymizationResult]:
        k_values = sorted(k)
        
        for current_k in k_values:
            if current_k not in self._k_cache:
                self._run_petre_for_k(current_k)
                self._k_cache[current_k] = True
                self._annotation_history[current_k] = {uid: list(annots) for uid, annots in self.annotations.items()}
        
        record = self.dataset_records[idx]
        uid = record.uid
        text = record.text
        
        results = []
        for k_val in k_values:
            annotations = self._annotation_history.get(k_val, {}).get(uid, [])
            
            sorted_annots = sorted(annotations, key=lambda x: x.start, reverse=True)
            anonymized_text = text
            for ann in sorted_annots:
                if 0 <= ann.start < ann.end <= len(anonymized_text):
                    anonymized_text = anonymized_text[:ann.start] + self.mask_text + anonymized_text[ann.end:]
            
            results.append(AnonymizationResult(
                text=anonymized_text,
                spans=annotations,
                metadata={
                    "k": k_val,
                    "perturbed_tokens": len(annotations),
                    "method": "petre",
                    "uid": uid
                }
            ))
        
        return results
    
    def anonymize_from_dataset(self, idx: int, k: int, *args, **kwargs) -> AnonymizationResult:
        return self.grid_anonymize_from_dataset(idx, k=[k], *args, **kwargs)[0]
    
    def _run_petre_for_k(self, k: int):
        ranks = self._evaluate_all()
        
        for idx, record in enumerate(self.dataset_records):
            rank = ranks[idx]
            
            if rank >= k:
                continue
            
            label = self.name_to_label[record.name]
            
            while rank < k:
                splits_probs = self._evaluate_document(idx)
                masked = self._mask_most_disclosive_term(idx, splits_probs, label)
                
                if not masked:
                    break
                
                splits_probs = self._evaluate_document(idx)
                rank = self._get_rank_from_probs(splits_probs, label)
    
    def _evaluate_all(self) -> np.ndarray:
        ranks = np.zeros(len(self.dataset_records))
        
        for idx in range(len(self.dataset_records)):
            splits_probs = self._evaluate_document(idx)
            label = self.name_to_label[self.dataset_records[idx].name]
            ranks[idx] = self._get_rank_from_probs(splits_probs, label)
        
        return ranks
    
    def _evaluate_document(self, idx: int) -> torch.Tensor:
        record = self.dataset_records[idx]
        text = record.text
        uid = record.uid
        
        splits = self._get_splits(text)
        splits_probs = torch.zeros((len(splits), self.num_labels))
        
        for split_idx, (split_start, split_end) in enumerate(splits):
            split_text = text[split_start:split_end]
            annotated_text = self._apply_annotations_to_split(text, uid, split_start, split_end)
            probs = self._evaluate_text(annotated_text)
            splits_probs[split_idx, :] = probs
        
        return splits_probs
    
    def _get_splits(self, text: str) -> List[Tuple[int, int]]:
        pattern = re.compile(r'[.!?]+\s+')
        splits = []
        start = 0
        
        for match in pattern.finditer(text):
            end = match.end()
            splits.append((start, end))
            start = end
        
        if start < len(text):
            splits.append((start, len(text)))
        
        if not splits:
            splits.append((0, len(text)))
        
        return splits
    
    def _apply_annotations_to_split(self, text: str, uid: str, split_start: int, split_end: int) -> str:
        annotations = self.annotations.get(uid, [])
        
        split_text = text[split_start:split_end]
        offset = split_start
        
        relevant_annotations = []
        for ann in annotations:
            if split_start <= ann.start < split_end or split_start < ann.end <= split_end:
                rel_start = max(0, ann.start - offset)
                rel_end = min(len(split_text), ann.end - offset)
                relevant_annotations.append((rel_start, rel_end))
        
        masked_text = split_text
        for start, end in sorted(relevant_annotations, key=lambda x: x[0], reverse=True):
            if 0 <= start < end <= len(masked_text):
                masked_text = masked_text[:start] + self.mask_text + masked_text[end:]
        
        return masked_text
    
    def _evaluate_text(self, text: str) -> torch.Tensor:
        results = self.tri_pipeline([text])[0]
        probs = torch.zeros(self.num_labels)
        
        for pred in results:
            label_str = pred["label"]
            label = int(label_str.split("_")[1]) if "_" in label_str else int(label_str)
            score = float(pred["score"])
            
            if 0 <= label < self.num_labels:
                probs[label] = score
        
        return probs
    
    def _get_rank_from_probs(self, splits_probs: torch.Tensor, label: int) -> int:
        aggregated_probs = splits_probs.mean(dim=0)
        sorted_idxs = torch.argsort(aggregated_probs, descending=True)
        rank_idx = torch.where(sorted_idxs == label)[0].item()
        return rank_idx + 1
    
    def _mask_most_disclosive_term(self, idx: int, splits_probs: torch.Tensor, label: int) -> bool:
        record = self.dataset_records[idx]
        text = record.text
        uid = record.uid
        
        splits = self._get_splits(text)
        individual_probs = splits_probs[:, label]
        sorted_splits_idxs = torch.argsort(individual_probs, descending=True)
        
        for split_idx in sorted_splits_idxs:
            split_start, split_end = splits[split_idx]
            split_text = text[split_start:split_end]
            
            annotated_split = self._apply_annotations_to_split(text, uid, split_start, split_end)
            original_terms = self._get_terms(split_text)
            annotated_terms = self._get_terms(annotated_split)
            
            if not annotated_terms:
                continue
            
            term_weights = self._compute_term_weights(annotated_split, annotated_terms, label)
            most_disclosive_idx = self._get_most_disclosive_term_idx(term_weights)
            
            if most_disclosive_idx >= 0:
                annotated_term_text = annotated_terms[most_disclosive_idx][2]
                
                for orig_start, orig_end, orig_text in original_terms:
                    if orig_text == annotated_term_text:
                        global_start = split_start + orig_start
                        global_end = split_start + orig_end
                        self._add_annotation(uid, global_start, global_end)
                        return True
        
        return False
    
    def _get_terms(self, text: str) -> List[Tuple[int, int, str]]:
        pattern = re.compile(r'\b\w+\b')
        terms = []
        
        for match in pattern.finditer(text):
            if match.group() != self.mask_text.strip('[]'):
                start, end = match.span()
                term_text = match.group()
                terms.append((start, end, term_text))
        
        return terms
    
    def _compute_term_weights(self, text: str, terms: List[Tuple[int, int, str]], label: int) -> np.ndarray:
        if self.explainer is not None:
            tokens = [term_text for _, _, term_text in terms]
            return self.explainer.explain(text, tokens=tokens, target_label=f"LABEL_{label}")
        
        return self._greedy_explainability(text, terms, label)
    
    def _greedy_explainability(self, text: str, terms: List[Tuple[int, int, str]], label: int) -> np.ndarray:
        base_probs = self._evaluate_text(text)
        base_prob = base_probs[label].item()
        
        term_weights = np.zeros(len(terms))
        
        for term_idx, (start, end, _) in enumerate(terms):
            masked_text = text[:start] + self.mask_text + text[end:]
            masked_probs = self._evaluate_text(masked_text)
            masked_prob = masked_probs[label].item()
            term_weights[term_idx] = base_prob - masked_prob
        
        return term_weights
    
    def _get_most_disclosive_term_idx(self, term_weights: np.ndarray) -> int:
        if len(term_weights) == 0:
            return -1
        
        sorted_idxs = np.argsort(term_weights)[::-1]
        
        for idx in sorted_idxs:
            if term_weights[idx] > 0:
                return int(idx)
        
        return -1
    
    def _add_annotation(self, uid: str, start: int, end: int):
        if uid not in self.annotations:
            self.annotations[uid] = []
        
        record = next((r for r in self.dataset_records if r.uid == uid), None)
        text = record.text if record else ""
        
        self.annotations[uid].append(TextAnnotation(
            start=start,
            end=end,
            text=text[start:end] if text and end <= len(text) else None,
            replacement=self.mask_text,
            annotator="petre"
        ))
    
    def get_annotations(self) -> Dict[str, List[TextAnnotation]]:
        return {uid: list(annots) for uid, annots in self.annotations.items()}
    
    def get_annotation_history(self) -> Dict[int, Dict[str, List[TextAnnotation]]]:
        return {k: {uid: list(annots) for uid, annots in hist.items()} for k, hist in self._annotation_history.items()}