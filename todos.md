# Paper todos

Priority 5 = most urgent. Supervisor notes merged in below; still being updated as they read the paper/reviews/thesis.

## Done
- [x] adapt rat-bench (dataset + TRI ablation pipeline wired up, group-split indices, SLURM tables)
- [x] use a text quality metric, most likely grammatical correctness one, because word diversity is already implied by divergence, and word-order distance is already zero for dpmlms (WG/SG grammar divergence added for rat_bench, tab, db_bio)
- [x] P5: specify the guarantee — added exact document-level and token-level guarantee statements in §Methodology (`sec:method-guarantee`), plus a scoped paragraph stating the SHAP-based allocation step itself is not covered by the DP proof
- [x] P5: relax/remove overclaiming guarantee language — done alongside the above, and in Limitations ("Scope of the formal DP guarantee")
- [x] P4: research question + structured-database anonymization analogy — added to Introduction (line ~85), citing Sweeney 2000
- [x] rat bench citation added (`krco_rat-bench_2026`), paper re-rendered, resolves cleanly
- [x] stopping conditions: kept only k-DP-SHAP (\kstop) as the flagship, headline stopping criterion; ρ-DP-SHAP (\rhostop) fully commented out (via `\iffalse...\fi`, not deleted) across abstract, intro, methodology, results, discussion, conclusion, baselines, and the appendix's Cross-Hyperparameter Analysis section, so it renders nowhere but is easy to reinstate later. \kstop is framed honestly as an empirically dominant (utility-preserving), non-formally-DP stopping heuristic, consistent with the corrected guarantee scoping above.
  - NOTE: if the commented ρ-DP-SHAP appendix section is ever reinstated, its "Practical selection" paragraph incorrectly claims "$k$ provides a formal dataset-level guarantee" — this contradicts the corrected methodology text and must be fixed first (ρ, not k, was the one with the restricted formal DP guarantee).

## Pending

### Priority 5
- Add an example showing the quality of text when perturbing all tokens and motivate a risk-aware selective methodology (the GC/grammar figure — configs are wired up for rat_bench/tab/db_bio, but no figure/text exists in `paper.tex` yet).
  - Maybe one of the figures in page 1 and one on page 2?

### Priority 3
- Utility-trend note across ε for both datasets — arguably already covered by §5.3 "Uniform vs. risk-aware DP rewriting" (lines ~827-832); revisit whether a more explicit standalone note is still wanted.

### Priority 2
- How does the Risk-temperature ($\tau_r$) parameter affect the overall empirical privacy and tradeoffs? Still fixed a-priori at 0.1, no ablation. Needs a new τ-sweep experiment (new compute, same SLURM structure as the ε-sweeps).

### Unprioritized
- get rid of multi-seed for now, this is too complicated and not needed
