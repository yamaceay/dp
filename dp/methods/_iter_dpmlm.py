import math
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import string

from dp.methods._dpmlm import DPMlmAnonymizer
from dp.methods.anonymizer import AnonymizationResult
from dp.methods.constants import Buckets, BucketDict, buckets_to_dicts
from dp.utils.token_ledger import TokenLedger
from dp.utils.risk import _scores_to_inverse_probs
from dp.utils.memory import clear_memory
from dp.loaders.base import TextAnnotation, TextAnnotations, TokenEdit


class IterDPMlmAnonymizer(DPMlmAnonymizer):
    MODEL_NAME = "iter_dpmlm"

    def __init__(self, *args, T: Union[int, float] = math.inf, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if T != math.inf and (not isinstance(T, int) or T < 1):
            raise ValueError("T must be a positive integer or math.inf")
        self.T = T

    def _compute_current_offsets(self, ledger: TokenLedger, n: int) -> List[Tuple[int, int]]:
        result: List[Optional[Tuple[int, int]]] = [None] * n
        sorted_entries = sorted(ledger._entries, key=lambda e: e.start)
        cursor = 0
        gap_idx = 0
        gaps = ledger._gaps

        for entry in sorted_entries:
            while gap_idx < len(gaps) and gaps[gap_idx].start < entry.start:
                cursor += len(gaps[gap_idx].text)
                gap_idx += 1
            if not entry.deleted:
                result[entry.index] = (cursor, cursor + len(entry.text))
                cursor += len(entry.text)
                for addition in entry.additions:
                    cursor += len(addition.text)

        for i in range(n):
            if result[i] is None:
                result[i] = (0, 0)

        return result  # type: ignore

    def _refresh_shap_scores(
        self,
        ledger: TokenLedger,
        original_text: str,
        n: int,
        target_label_str: str,
    ) -> np.ndarray:
        current_text = ledger.render_offsets(original_text)
        current_offsets = self._compute_current_offsets(ledger, n)

        surviving = [(i, current_offsets[i]) for i in range(n) if not ledger.entry(i).deleted]
        if not surviving:
            return np.zeros(n, dtype=float)

        surviving_indices, surviving_offsets = zip(*surviving)
        raw_scores = self._explainer.explain(
            current_text, list(surviving_offsets), target_label=target_label_str
        )

        scores = np.zeros(n, dtype=float)
        for pos, orig_idx in enumerate(surviving_indices):
            scores[orig_idx] = float(raw_scores[pos])
        return scores

    def _print_step_risk_scores(
        self,
        step: int,
        shap_scores: np.ndarray,
        unprocessed: set,
        text: str,
        offsets: List[Tuple[int, int]],
        record_ref: str,
        refreshed: bool,
    ) -> None:
        if not self.verbose:
            return
        top = sorted(unprocessed, key=lambda i: float(shap_scores[i]), reverse=True)[:5]
        refresh_marker = " [refreshed]" if refreshed else ""
        tokens = [f"{text[offsets[i][0]:offsets[i][1]]}({shap_scores[i]:.4f})" for i in top]
        print(
            f"[iter_dpmlm][verbose] record={record_ref} step={step}{refresh_marker} "
            f"remaining={len(unprocessed)} top5={tokens}"
        )

    def _compute_token_epsilon(
        self,
        idx: int,
        unprocessed_by_score: List[int],
        shap_scores: np.ndarray,
        spent_budgets: List[float],
        eps_val: float,
        n: int,
    ) -> float:
        remaining_budget = eps_val * n - sum(spent_budgets)
        if not unprocessed_by_score:
            return max(remaining_budget, 1e-8)
        scores = np.array([float(shap_scores[i]) for i in unprocessed_by_score], dtype=float)
        weights = _scores_to_inverse_probs(scores, temperature=self.risk_temperature)
        idx_pos = unprocessed_by_score.index(idx)
        return float(remaining_budget * weights[idx_pos])

    def _apply_token_iterative(
        self,
        idx: int,
        ledger: TokenLedger,
        original_text: str,
        n: int,
        eps_for_token: float,
        masked_spans: Sequence[TextAnnotation],
        runtime_stats: Dict[str, int],
    ) -> None:
        entry = ledger.entry(idx)

        if self._is_token_masked(entry.start, entry.end, masked_spans):
            ledger.replace(idx, entry.original_text)
            runtime_stats["total"] += 1
            runtime_stats["skipped_masked"] += 1
            return

        current_offsets = self._compute_current_offsets(ledger, n)
        token_start, token_end = current_offsets[idx]
        current_text = ledger.render_offsets(original_text)
        current_token = current_text[token_start:token_end]

        if current_token in string.punctuation:
            ledger.replace(idx, current_token)
            runtime_stats["total"] += 1
            runtime_stats["skipped_punctuation"] += 1
            return

        private_token = self._privatize_token(
            current_text, current_token, (token_start, token_end), eps_for_token
        )

        original_span = entry.original_text
        if len(private_token) == len(original_span):
            private_token = "".join(
                p.upper() if o.isupper() else p.lower()
                for p, o in zip(private_token, original_span)
            )
        elif original_span and original_span[0].isupper():
            private_token = private_token.capitalize()

        ledger.replace(idx, private_token)
        if private_token != current_token:
            runtime_stats["perturbed"] += 1
        else:
            runtime_stats["same_token"] += 1
        runtime_stats["total"] += 1

    def _build_output(
        self,
        hp: BucketDict,
        ledger: TokenLedger,
        original_text: str,
        offsets: List[Tuple[int, int]],
        eps_val: float,
        used_precomputed: bool,
        runtime_stats: Dict[str, int],
        threshold_type: Optional[str],
        threshold: Any,
        extra_meta: Dict[str, Any],
    ) -> Tuple[BucketDict, AnonymizationResult]:
        hp_out = {**hp}
        if threshold is not None and threshold_type in {"k", "rho", "lambda"}:
            hp_out[threshold_type] = threshold

        private_text = ledger.render_offsets(original_text)
        token_edits = [TokenEdit.from_mapping(e) for e in ledger.result_edits_metadata()]

        if used_precomputed:
            explainer_name = "PrecomputedRisk"
        elif self._explainer is not None:
            explainer_name = self._explainer.__class__.__name__
        else:
            explainer_name = "unknown"

        metadata: Dict[str, Any] = {
            "epsilon": eps_val,
            "method": self.MODEL_NAME,
            "model": self.model_checkpoint,
            "threshold": threshold,
            "explainer": explainer_name,
            "T": self.T,
            **runtime_stats,
            **extra_meta,
        }
        metadata["total"] = int(len(offsets))

        return (
            hp_out,
            AnonymizationResult(
                text=private_text,
                annotations=TextAnnotations(spans=[], token_edits=token_edits),
                metadata=metadata,
            ),
        )

    def anonymize_any_text(
        self,
        text: str,
        *args,
        buckets: Buckets = [],
        record_name: Optional[str] = None,
        record_uid: Optional[str] = None,
        masked_spans: Optional[Sequence[TextAnnotation]] = None,
        **kwargs,
    ) -> List[Tuple[BucketDict, AnonymizationResult]]:
        if not text or not text.strip():
            return []
        if masked_spans is None:
            masked_spans = []

        from dp.utils.selector.until_k_selector import UntilKUnit
        from dp.utils.selector.all_selector import AllUnit

        combos = buckets_to_dicts(buckets)
        outputs: List[Tuple[BucketDict, AnonymizationResult]] = []

        precomputed_offsets = self._resolve_precomputed_offsets(record_name, record_uid)
        if precomputed_offsets is not None:
            offsets = precomputed_offsets
        else:
            _, offsets = self._tokenize(text)
        n = len(offsets)

        static_shap_scores: Optional[np.ndarray] = None
        used_precomputed = False
        if math.isinf(self.T):
            static_shap_scores = self._lookup_precomputed_scores(
                text, offsets, record_name, record_uid=record_uid
            )
            used_precomputed = static_shap_scores is not None

        needs_live_shap = static_shap_scores is None
        needs_k_stop = self._unit is not None and isinstance(self._unit, UntilKUnit)

        if needs_live_shap and self._explainer is None:
            raise ValueError(
                "IterDPMlmAnonymizer requires precomputed risk scores (for T=inf) or an explainer"
            )

        target_label_id: Optional[int] = None
        target_label_str: Optional[str] = None
        if needs_live_shap or needs_k_stop:
            target_label_id = self._target_label_id_for_record(record_name)
            target_label_str = f"LABEL_{target_label_id}"

        combos_by_epsilon: Dict[float, List[BucketDict]] = {}
        for hp in combos:
            eps_val = hp.get("epsilon")
            if eps_val is None:
                raise ValueError("IterDPMlmAnonymizer requires epsilon via Buckets (EpsilonParam)")
            combos_by_epsilon.setdefault(float(eps_val), []).append(hp)

        if self._unit is None:
            self._unit = AllUnit()

        for eps_val, eps_combos in combos_by_epsilon.items():
            try:
                runtime_stats: Dict[str, int] = {
                    "perturbed": 0,
                    "total": 0,
                    "added": 0,
                    "deleted": 0,
                    "skipped_punctuation": 0,
                    "skipped_masked": 0,
                    "same_token": 0,
                }
                ledger = TokenLedger(text, offsets)
                spent_budgets: List[float] = []
                processed: set = set()

                if isinstance(self._unit, UntilKUnit):
                    k_vals = sorted(set(int(hp["k"]) for hp in eps_combos if hp.get("k") is not None))
                    if not k_vals:
                        raise ValueError(
                            "IterDPMlmAnonymizer using until_k selector requires KParams buckets"
                        )

                    self._unit.set_thresholds(k_vals, name="k")
                    self._unit.set_target_label(target_label_id)
                    self._unit.set_rank_evaluator(self._make_rank_evaluator())

                    rank_evaluator = self._make_rank_evaluator()
                    current_text = ledger.render_offsets(text)
                    current_rank = rank_evaluator(current_text, target_label_id)

                    shap_scores: Optional[np.ndarray] = static_shap_scores
                    step_count = 0
                    unprocessed: set = set(range(n))

                    for target_k in k_vals:
                        matching_hp = next(
                            (hp for hp in eps_combos if int(hp.get("k", -1)) == target_k),
                            eps_combos[0],
                        )

                        if current_rank >= target_k:
                            outputs.append(self._build_output(
                                hp=matching_hp, ledger=ledger, original_text=text,
                                offsets=offsets, eps_val=eps_val, used_precomputed=used_precomputed,
                                runtime_stats=runtime_stats,
                                threshold_type="k", threshold=target_k,
                                extra_meta={"rank": current_rank, "selector": "until_k",
                                            "processed_count": len(processed)},
                            ))
                            continue

                        while unprocessed and current_rank < target_k:
                            needs_refresh = shap_scores is None or (
                                not math.isinf(self.T) and step_count % int(self.T) == 0
                            )
                            if needs_refresh:
                                shap_scores = self._refresh_shap_scores(
                                    ledger, text, n, target_label_str
                                )

                            record_ref = record_uid or record_name or "<unknown>"
                            self._print_step_risk_scores(
                                step_count, shap_scores, unprocessed, text, offsets,
                                record_ref, needs_refresh,
                            )

                            candidates = sorted(
                                unprocessed,
                                key=lambda i: float(shap_scores[i]),
                                reverse=True,
                            )
                            idx = candidates[0]

                            eps_for_token = self._compute_token_epsilon(
                                idx, candidates, shap_scores, spent_budgets, eps_val, n
                            )

                            self._apply_token_iterative(
                                idx, ledger, text, n, eps_for_token, masked_spans, runtime_stats
                            )
                            spent_budgets.append(eps_for_token)
                            processed.add(idx)
                            unprocessed.discard(idx)
                            step_count += 1

                            if ledger.is_modified(idx):
                                current_text = ledger.render_offsets(text)
                                current_rank = rank_evaluator(current_text, target_label_id)

                        outputs.append(self._build_output(
                            hp=matching_hp, ledger=ledger, original_text=text,
                            offsets=offsets, eps_val=eps_val, used_precomputed=used_precomputed,
                            runtime_stats=runtime_stats,
                            threshold_type="k", threshold=target_k,
                            extra_meta={"rank": current_rank, "selector": "until_k",
                                        "processed_count": len(processed)},
                        ))

                else:
                    shap_scores = static_shap_scores
                    step_count = 0
                    unprocessed_set: set = set(range(n))

                    while unprocessed_set:
                        needs_refresh = shap_scores is None or (
                            not math.isinf(self.T) and step_count % int(self.T) == 0
                        )
                        if needs_refresh:
                            shap_scores = self._refresh_shap_scores(
                                ledger, text, n, target_label_str
                            )

                        record_ref = record_uid or record_name or "<unknown>"
                        self._print_step_risk_scores(
                            step_count, shap_scores, unprocessed_set, text, offsets,
                            record_ref, needs_refresh,
                        )

                        candidates = sorted(
                            unprocessed_set,
                            key=lambda i: float(shap_scores[i]),
                            reverse=True,
                        )
                        idx = candidates[0]

                        eps_for_token = self._compute_token_epsilon(
                            idx, candidates, shap_scores, spent_budgets, eps_val, n
                        )

                        self._apply_token_iterative(
                            idx, ledger, text, n, eps_for_token, masked_spans, runtime_stats
                        )
                        spent_budgets.append(eps_for_token)
                        unprocessed_set.discard(idx)
                        step_count += 1

                    outputs.append(self._build_output(
                        hp=eps_combos[0], ledger=ledger, original_text=text,
                        offsets=offsets, eps_val=eps_val, used_precomputed=used_precomputed,
                        runtime_stats=runtime_stats,
                        threshold_type=None, threshold=None,
                        extra_meta={"selector": "all", "processed_count": n},
                    ))

            finally:
                clear_memory()

        return outputs
