# Plot Resizing Handoff (Single-Metric + 5-Panel Consistency)

## Context
The current single-metric plot sizing/layout is not acceptable.  
The goal is visual consistency between:

- single-panel plots (no epsilon faceting), and
- 5-panel epsilon-faceted plots.

Single plots should look like a direct crop/extraction of one panel cell from the 5-panel layout.

## Hard Requirements

1. Use the 5-panel layout as the single source of truth for geometry.
2. Base figure size for 5-panel layout is:
   - `W5 = 18`
   - `H5 = 12`
3. 5-panel grid decomposition is `2 x 23` logical grid units.
4. One panel uses `7` horizontal grid units and one half-row vertically.
5. Therefore single-panel target size must be exactly:
   - `W1 = 18 * 7 / 23`
   - `H1 = 12 / 2`
6. Do not invent separate sizing heuristics for single-metric single-panel plots.
7. Single-panel ratio should visually match one cell of the 5-panel composition.

## Panel Alignment Requirements (Faceted Case)

1. Shared x-axis limits across all epsilon panels.
2. Shared y-axis across all epsilon panels.
3. Global y-order must be fixed once and reused in all panels.
4. Method ordering cannot change per panel.
5. Same method must appear at same y-position in every panel.

## Readability / Clutter Requirements

1. Do not allow labels/annotations to spill into neighboring panels.
2. If needed, clip annotation text to axes bounds.
3. In 5-panel layout, avoid redundant y-label rendering on every panel.
4. Keep legend/title readable without breaking panel geometry.

## What "Good" Looks Like

1. A single-panel figure should look like it was cherry-picked from a 5-panel figure.
2. Faceted panels should allow direct row-by-row comparison across epsilon.
3. No "pipe-like" vertical distortion in single-panel outputs.
4. No per-panel row reordering.

## Suggested Acceptance Checks

1. Size check:
   - Assert single-panel figsize is `(18 * 7 / 23, 12 / 2)`.
2. Order check:
   - Assert y-label list is identical for all epsilon panels.
3. Axis check:
   - Assert x-limits are identical for all epsilon panels.
4. Visual check:
   - Compare one single-panel output against one cell in a 5-panel output for the same method group/metric.

## Deliverable Expectation

Implement a shared layout handler used by both plotting scripts, with one consistent geometry contract.  
No script should have an independent "single panel fallback" ratio that deviates from the 5-panel-derived ratio.
