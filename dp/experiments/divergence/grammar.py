"""Grammatical Correctness (GC) divergence metric.

Implements WG (Word Grammaticality) and SG (Sentence Grammaticality) from:
  Cano & Habernal (2025) — "Differentially-private text generation degrades
  output language quality", arXiv 2509.11176.

  GC  = 1 - (#errors / #tokens)
  WG  = 1 - (total_errors / total_words)          [whole-sample normalisation]
  SG  = mean over sentences of (1 - errors_s / words_s)

Requires: pip install language_tool_python
LanguageTool downloads a local server on first use (~250 MB, Java required).

Usage as a divergence metric:
  "similarity" = GC score of the anonymised text (1 = perfect grammar).
  "divergence" = 1 - GC  (grammatical error rate).

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


class GrammaticalCorrectnessMetric(DivergenceMetric):
    """WG or SG grammatical correctness using LanguageTool."""

    def __init__(self, variant: str = "wg", language: str = "en-US") -> None:
        if variant not in ("wg", "sg"):
            raise ValueError(f"variant must be 'wg' or 'sg', got '{variant}'")
        super().__init__(f"grammar_{variant}")
        self.variant = variant
        self.language = language
        self._tool: Any = None

    def _get_tool(self) -> Any:
        if self._tool is None:
            try:
                import language_tool_python
            except ImportError as exc:
                raise ImportError(
                    "pip install language_tool_python  (requires Java runtime)"
                ) from exc
            self._tool = language_tool_python.LanguageTool(self.language)
        return self._tool

    def clone(self) -> "GrammaticalCorrectnessMetric":
        return GrammaticalCorrectnessMetric(variant=self.variant, language=self.language)

    def _gc_wg(self, text: str) -> float:
        tool = self._get_tool()
        errors = len(tool.check(text))
        return max(0.0, 1.0 - errors / _word_count(text))

    def _gc_sg(self, text: str) -> float:
        tool = self._get_tool()
        sentences = _split_sentences(text)
        scores = []
        for sent in sentences:
            errors = len(tool.check(sent))
            scores.append(max(0.0, 1.0 - errors / _word_count(sent)))
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
            "definition": "GC = 1 - (#grammar_errors / #words)",
        }

    def cleanup(self) -> None:
        if self._tool is not None:
            try:
                self._tool.close()
            except Exception:
                pass
            self._tool = None


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
