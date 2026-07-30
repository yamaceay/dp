Alright, caveman mode:

Setup: You trained 5 attacker-brains on "BART summary" background info, and 5 attacker-brains on "full literal text" background info. Then you asked: do BART-brains and no-BART-brains size up people the same way, or differently?

What numbers say:
- Two brains trained the same way (both BART, or both no-BART) → their risk scores differ by ~0.00045 on average (small).
- Two brains trained different ways (one BART, one no-BART) → their risk scores differ by ~0.00056 on average (bigger).

Gap is small in absolute terms, but consistent.

Permutation test (the trustworthy one): "if group labels were random noise, how often would we see a gap this big by luck?" Answer: only 9 times out of 1000 (p=0.009). Luck very unlikely. Gap is real.

Welch's t-test (the familiar one, side dish): says the same thing, even more confidently (p=0.0002). Cohen's d = 1.3 means the gap is "big" by normal stats-speak standards, not just statistically-real-but-tiny.

Caveman verdict: BART-summary brain and literal-text brain are NOT twins. They judge risk noticeably differently. So swapping literal text for a BART summary is not a free lunch — it measurably changes attacker behavior, small numbers or not. Good ablation result: it justifies why the paper treats this as something worth testing, not something to hand-wave as "basically the same."
