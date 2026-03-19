# Coding Guidelines

## Self-explanatory code only.
No comments, no decorative emojis, no redundant docstrings. The code should speak for itself. If it needs explaining, it needs rewriting.
## Plug-and-play architecture.
Each component should have a clearly defined, type-strict interface. Avoid ad-hoc control flow (e.g., patchy if-else chains). Everything must integrate cleanly and predictably.
## Type strictness is non-negotiable.
Follow the discipline of languages like Go: explicit types, no silent coercion, no ambiguity. Type hints are mandatory in dynamic languages like Python or TypeScript.
## File-class ownership clarity.
Each file should encapsulate one class or a closely related group of functionalities. The file name should reveal its purpose. You should know exactly where to look when debugging.
## Prefer reuse over reinvention.
Before writing new code, check if the functionality already exists. Extend modular components when possible. Only create new modules when absolutely necessary.
## Do the job — no more, no less.
Implement exactly what’s requested. Don’t speculate on future requirements. Future-proofing is often just premature complexity.
## Minimal-change philosophy.
Working code is sacred. When fixing or extending, make the smallest possible change to achieve the goal. Avoid refactoring unless it’s justified by necessity, not aesthetics.
## Practicality first.
The less effort it takes to run, the better. Default values must be meaningful. Use None or similar placeholders only when it makes the runtime logic cleaner and more predictable.
## Simple initialization.
Classes should be initialized using plain keyword arguments. If dependencies are required, inject them via setter methods — not nested constructors.
## Top-down control.
The algorithm should be understandable and controllable from the top level. Internal details should never hijack the global flow.
## Modularization for clarity.
Shared or repetitive logic should be abstracted into reusable modules. Keep core algorithms clean — free from implementation noise.
## Builder pattern for complex flows.
When multiple configurations exist, implement a builder pattern. Set parameters upfront, then execute through a single entry point.
## Determinism over magic.
No hidden behavior. Avoid implicit side effects. What happens should be explicitly visible from the call site.
## Fail loud, fail fast.
Don’t silently pass exceptions. Error handling should be explicit and meaningful — either recover or raise, but never ignore.
## Dependency minimalism.
Avoid bloating the stack. External libraries should only be used when they provide substantial leverage. Prefer standard libraries and internal modules when possible.
## Tests are contracts.
Unit tests define the interface behavior, not just correctness. If a test breaks, it means a contract was violated — not just a bug was found.

# Writing Guidelines

## Purpose.
Clear, coherent, academically rigorous writing with a human, practical tone.
## Core principles.
Write clearly and directly with simple sentence structures. Keep the tone academic but practical. Avoid rhetorical flourishes, artificial emphasis, and AI-like stylistic markers. Do not over-explain well-known concepts or under-explain paper-specific design choices. Each paragraph answers one question, develops one idea, and links to the next step.
## Vocabulary and consistency.
Preserve the author’s chosen terminology. Use synonyms only to improve clarity or remove ambiguity. Keep deliberate terms consistent across the paper. Avoid buzzwords, marketing language, and exaggerated claims. Prefer technical precision over stylistic variation. Example preferences: static attacker over stylistic alternatives; risk-based over risk-driven; do not rotate method/approach/framework without reason.
## Sentence and paragraph structure.
Prefer short to medium-length sentences. Avoid chained clauses, excessive commas, nested subordinates, and sentence–dash–sentence forms. Avoid artificial emphasis unless technically required. Keep paragraphs compact and purposeful (3–6 sentences).
Avoid manual line breaks in prose; separate paragraphs with blank lines. Keep list items and sentence starts consistently capitalized.
## Flow and ordering.
Definitions appear before use. Motivation precedes mechanism; mechanism precedes evaluation. State assumptions explicitly. Avoid forward references unless required. Expected order: motivation and framing; threat model and preliminaries; method; variants/ablations; methodology; evaluation and discussion.
## Academic tone and claims.
Be precise and restrained. State contributions factually without overselling. Avoid subjective language unless empirically supported. When simplifying or deviating from prior work, explain why clearly and non-defensively. Treat related work respectfully and accurately.
## Practical orientation.
Emphasize design choices, trade-offs, and constraints. Explain the purpose and scope of simplifications. Prefer operational definitions when possible. Keep runtime, scalability, and applicability in view.
## Repetition and references.
Avoid unnecessary repetition. When repetition is needed, keep wording consistent. Refer back to definitions succinctly rather than restating them. Avoid excessive citations in a single sentence.
## Explicit avoidances.
No stylistic markdown emphasis for prose. No rhetorical questions or conversational fillers. No list-heavy prose unless structurally required.
## Overall goal.
The paper should read as careful, literature-aware, precise about assumptions, honest about limitations, and focused on a concrete problem. Clarity over elegance. Correctness over cleverness. Human readability over stylistic flair.
