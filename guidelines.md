# Coding Guidelines (Refined and Complete)

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