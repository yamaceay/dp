"""Grammatical Correctness (GC) divergence metric.

Implements WG (Word Grammaticality) and SG (Sentence Grammaticality) from:
  Cano & Habernal (2025) — "Differentially-private text generation degrades
  output language quality", arXiv 2509.11176.

Scored via a RoBERTa classifier fine-tuned on CoLA (Corpus of Linguistic
Acceptability) rather than language_tool_python, which requires a local Java
runtime (unavailable on cluster containers). Per sentence, GC is the model's
p(acceptable):
  SG  = mean over sentences of p(acceptable)_s                 [unweighted]
  WG  = word-count-weighted mean over sentences of p(acceptable)_s

Requires: pip install torch transformers

Usage as a divergence metric:
  "similarity" = GC score of the anonymised text (1 = perfect grammar).
  "divergence" = 1 - GC  (grammatical unacceptability rate).

The original texts' GC is computed as a baseline and stored in metadata.
"""

from __future__ import annotations

import re
from statistics import mean
from typing import Any, Dict, List, Optional, Sequence

from dp.experiments.divergence.base import DivergenceMetric, TextDivergenceExperiment
from dp.experiments import ExperimentResult


def _word_count(text: str) -> int:
    return max(len(text.split()), 1)


def _split_sentences(text: str) -> List[str]:
    try:
        import nltk
        try:
            sentences = nltk.sent_tokenize(text)
        except LookupError:
            nltk.download("punkt_tab", quiet=True)
            sentences = nltk.sent_tokenize(text)
        return [s for s in sentences if s.strip()]
    except ImportError:
        pass
    # fallback: split on sentence-ending punctuation
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p for p in parts if p.strip()] or [text]


COLA_MODEL = "cointegrated/roberta-large-cola-krishna2020"


class GrammaticalCorrectnessMetric(DivergenceMetric):
    """WG or SG grammatical correctness using a CoLA-tuned acceptability classifier."""

    def __init__(self, variant: str = "wg", language: str = "en-US") -> None:
        if variant not in ("wg", "sg"):
            raise ValueError(f"variant must be 'wg' or 'sg', got '{variant}'")
        super().__init__(f"grammar_{variant}")
        self.variant = variant
        self.language = language
        self._tokenizer: Any = None
        self._model: Any = None

    def _get_model(self) -> Any:
        if self._model is None:
            try:
                import torch  # noqa: F401
                from transformers import AutoModelForSequenceClassification, AutoTokenizer
            except ImportError as exc:
                raise ImportError("pip install torch transformers") from exc
            self._tokenizer = AutoTokenizer.from_pretrained(COLA_MODEL)
            self._model = AutoModelForSequenceClassification.from_pretrained(COLA_MODEL)
            self._model.eval()
        return self._tokenizer, self._model

    def clone(self) -> "GrammaticalCorrectnessMetric":
        return GrammaticalCorrectnessMetric(variant=self.variant, language=self.language)

    def _acceptability(self, text: str) -> float:
        import torch

        tokenizer, model = self._get_model()
        inputs = tokenizer(text, return_tensors="pt", truncation=True)
        with torch.no_grad():
            logits = model(**inputs).logits
        # LABEL_0 = acceptable (verified empirically, see grammar_poc.py)
        return float(torch.softmax(logits, dim=-1)[0, 0])

    def _gc_wg(self, text: str) -> float:
        sentences = _split_sentences(text)
        if not sentences:
            return 0.0
        weights = [_word_count(s) for s in sentences]
        scores = [self._acceptability(s) for s in sentences]
        total_weight = sum(weights)
        return sum(s * w for s, w in zip(scores, weights)) / total_weight

    def _gc_sg(self, text: str) -> float:
        sentences = _split_sentences(text)
        scores = [self._acceptability(s) for s in sentences]
        return mean(scores) if scores else 0.0

    def gc(self, text: str) -> float:
        if self.variant == "sg":
            return self._gc_sg(text)
        return self._gc_wg(text)

    def similarities(self, references: Sequence[str], candidates: Sequence[str]) -> List[float]:
        # references are ignored — GC is an intrinsic quality metric
        return [self.gc(c) for c in candidates]

    def metadata(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "variant": self.variant,
            "language": self.language,
            "model": COLA_MODEL,
            "definition": "GC = mean p(acceptable) over sentences (CoLA classifier)",
        }

    def cleanup(self) -> None:
        self._tokenizer = None
        self._model = None


class GrammaticalCorrectnessDivergence(TextDivergenceExperiment):
    """Divergence experiment measuring grammatical correctness degradation.

    For each annotation source, computes GC (WG or SG) on anonymised texts.
    Also computes baseline GC on original texts and stores it in the report.
    divergence = 1 - GC_anonymised  (grammatical error rate).
    """

    def __init__(self, variant: str = "wg", language: str = "en-US") -> None:
        super().__init__(GrammaticalCorrectnessMetric(variant=variant, language=language))
        self._variant = variant
        self._language = language

    def run(self, **kwargs: Any) -> ExperimentResult:
        if not self.metric:
            raise RuntimeError("setup must be completed before run")

        total_records = len(self.original_texts)
        original_keys = list(self.original_texts.keys())
        original_texts_list = [self.original_texts[k] for k in original_keys]

        # Baseline GC on original texts
        original_gc_values = [self.metric.gc(t) for t in original_texts_list]
        original_gc_mean = mean(original_gc_values) if original_gc_values else 0.0

        evaluations: Dict[str, Dict[str, Any]] = {}
        divergence_means: List[float] = []

        for name, payload in self.evaluation_datasets.items():
            texts: Dict[str, str] = payload["texts"]
            total: int = payload["total"]
            matched_keys = [k for k in self.original_texts if k in texts]

            if not matched_keys:
                evaluations[name] = {
                    "similarity": {},
                    "divergence": {},
                    "summary": None,
                    "matched": 0,
                    "total": total,
                    "missing": total_records,
                }
                continue

            candidates = [texts[k] for k in matched_keys]
            gc_values = [self.metric.gc(c) for c in candidates]
            error_rates = [1.0 - v for v in gc_values]

            similarity_map = {k: gc_values[i] for i, k in enumerate(matched_keys)}
            divergence_map = {k: error_rates[i] for i, k in enumerate(matched_keys)}
            summary = self._summarize(gc_values, error_rates)
            evaluations[name] = {
                "similarity": similarity_map,
                "divergence": divergence_map,
                "summary": summary,
                "matched": len(matched_keys),
                "total": total,
                "missing": max(total_records - len(matched_keys), 0),
            }
            if summary:
                divergence_means.append(summary["divergence_mean"])

        score_value = mean(divergence_means) if divergence_means else 0.0
        metrics = {
            "records": self.record_info,
            "original": {
                "count": total_records,
                f"gc_{self._variant}_mean": original_gc_mean,
            },
            "evaluations": evaluations,
            "metric": self.metric_metadata.get("name"),
            "metric_metadata": self.metric_metadata,
        }
        return ExperimentResult(score=score_value, metrics=metrics, metadata=dict(self.metric_metadata))
