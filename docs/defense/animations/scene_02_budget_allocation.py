"""Scene 2: Risk-weighted noise temperature vs. uniform.

Bar height = sampling temperature ∝ 1/ε_i.
Tall bar = noisier replacement = more private.
High-risk tokens → tall bars (more scrambled).
Low-risk tokens  → short bars (barely touched).
"""
from manim import *
import numpy as np

TUB_RED    = ManimColor("#c50e1f")
TUB_ORANGE = ManimColor("#e87722")
TUB_BLUE   = ManimColor("#1a7abf")
TUB_GRAY   = ManimColor("#aaaaaa")
BG         = ManimColor("#0f0f0f")

RISK_SCORES = np.array([0.92, 0.81, 0.70, 0.55, 0.48, 0.20, 0.12, 0.05])
WORDS       = ["John", "1987", "Munich", "senior", "engineer", "from", "Berlin", "and"]
EPSILON     = 10.0
TAU         = 1.0   # risk temperature τ  (τ=0.1 in paper experiments; τ=1 shows clearer bars)


def softmax_weights(risks, tau):
    """w_i = exp(-r_i/τ) / Σ exp(-r_j/τ)  — paper §2.3"""
    log_w = -risks / tau
    log_w -= log_w.max()          # numerical stability
    w = np.exp(log_w)
    return w / w.sum()


def bar_color(risk):
    if risk >= 0.7: return TUB_RED
    if risk >= 0.4: return TUB_ORANGE
    return TUB_GRAY


class BudgetAllocation(Scene):
    def construct(self):
        self.camera.background_color = BG
        n = len(RISK_SCORES)

        title = Text("Risk-Weighted Noise Temperature",
                     font_size=30, color=WHITE).to_edge(UP, buff=0.4)
        self.play(Write(title))

        bar_w   = 0.6
        spacing = 0.95
        x0      = -(n * spacing) / 2 + spacing / 2
        base_y  = -0.4
        max_h   = 2.6

        # ── compute per-token budgets via correct softmax ─────────
        weights  = softmax_weights(RISK_SCORES, TAU)
        budgets  = EPSILON * n * weights          # ε_i = ε·n·w_i
        phi_bar  = np.mean(np.exp(-RISK_SCORES / TAU))  # φ̄ = average weight
        temps_rw = 1.0 / budgets                  # T_i ∝ 1/ε_i (noise level)
        temps_rw_norm = temps_rw / temps_rw.max()

        # uniform: all same temperature
        temp_unif_norm = np.full(n, 0.35)

        # ── token labels ─────────────────────────────────────────
        tok_labels = VGroup(*[
            Text(w, font_size=16, color=WHITE
                 ).move_to([x0 + i * spacing, base_y - 0.4, 0])
            for i, w in enumerate(WORDS)
        ])
        self.play(FadeIn(tok_labels))
        self.wait(0.3)

        # ── uniform bars ─────────────────────────────────────────
        unif_bars = VGroup()
        for i in range(n):
            h   = max_h * temp_unif_norm[i]
            bar = Rectangle(width=bar_w, height=h,
                            fill_color=WHITE, fill_opacity=0.30,
                            stroke_color=WHITE, stroke_width=0.8)
            bar.move_to([x0 + i * spacing, base_y + h / 2, 0])
            unif_bars.add(bar)

        unif_label = Text("Uniform DP-MLM  — same ε for every token",
                          font_size=17, color=TUB_GRAY).next_to(title, DOWN, buff=0.2)
        unif_temp_labels = VGroup(*[
            Text("ε", font_size=13, color=TUB_GRAY
                 ).next_to(unif_bars[i], UP, buff=0.06)
            for i in range(n)
        ])

        self.play(FadeIn(unif_label),
                  LaggedStart(*[GrowFromEdge(b, DOWN) for b in unif_bars],
                              lag_ratio=0.10),
                  run_time=1.8)
        self.play(FadeIn(unif_temp_labels))
        self.wait(0.8)

        # ── risk score annotations ────────────────────────────────
        risk_dots = VGroup()
        for i, r in enumerate(RISK_SCORES):
            top_y = base_y + max_h * temp_unif_norm[i] + 0.55
            dot   = Dot(color=bar_color(r), radius=0.09).move_to([x0 + i * spacing, top_y, 0])
            rlbl  = Text(f"r={r:.2f}", font_size=11, color=bar_color(r)
                         ).next_to(dot, UP, buff=0.04)
            risk_dots.add(VGroup(dot, rlbl))

        risk_title = Text("SHAP risk →", font_size=14, color=TUB_GRAY
                          ).next_to(risk_dots, LEFT, buff=0.15)
        self.play(FadeIn(risk_dots), FadeIn(risk_title))
        self.wait(0.8)

        # ── transform to risk-weighted temperature bars ───────────
        rw_bars = VGroup()
        for i, r in enumerate(RISK_SCORES):
            h   = max(max_h * temps_rw_norm[i], 0.05)
            bar = Rectangle(width=bar_w, height=h,
                            fill_color=bar_color(r), fill_opacity=0.85,
                            stroke_width=0)
            bar.move_to([x0 + i * spacing, base_y + h / 2, 0])
            rw_bars.add(bar)

        # Show per-token budget ε_i on bars
        budget_labels_rw = VGroup(*[
            Text(f"ε={budgets[i]:.1f}", font_size=10,
                 color=bar_color(RISK_SCORES[i])
                 ).next_to(rw_bars[i], UP, buff=0.06)
            for i in range(n)
        ])

        rw_label = Text("DP-MLM-X  — high risk → small ε_i → maximum noise",
                        font_size=17, color=TUB_RED).next_to(title, DOWN, buff=0.2)

        self.play(FadeOut(unif_label), FadeOut(unif_temp_labels), FadeIn(rw_label))
        self.play(
            *[Transform(unif_bars[i], rw_bars[i]) for i in range(n)],
            run_time=2.2
        )
        self.play(LaggedStart(*[FadeIn(l) for l in budget_labels_rw], lag_ratio=0.08),
                  run_time=0.9)
        self.wait(0.5)

        # ── key formulas ──────────────────────────────────────────
        formula = MathTex(
            r"\varepsilon_i = \varepsilon \cdot n \cdot w_i,",
            r"\quad w_i = \mathrm{softmax}(-r/\tau)_i,",
            r"\quad \textstyle\sum_i \varepsilon_i = n\varepsilon",
            font_size=24, color=YELLOW
        ).to_edge(DOWN, buff=0.7)
        self.play(Write(formula))
        self.wait(0.5)

        # ── per-token guarantee ───────────────────────────────────
        guarantee = MathTex(
            r"\text{Per-token guarantee: }",
            r"\max_j\,\varepsilon_j = \varepsilon / \bar\varphi \ge \varepsilon",
            font_size=20, color=TUB_ORANGE
        ).next_to(formula, DOWN, buff=0.2)
        self.play(Write(guarantee))

        glow = SurroundingRectangle(unif_bars[0], color=TUB_RED, buff=0.06,
                                    stroke_width=2)
        note = Text("Highest risk → smallest ε_i → strongest protection",
                    font_size=15, color=TUB_RED).next_to(glow, RIGHT, buff=0.15)
        self.play(Create(glow), FadeIn(note))

        tau_footnote = Text("Paper uses τ=0.1 for more aggressive redistribution (Scene 4).",
                            font_size=12, color=TUB_GRAY).to_corner(DR, buff=0.2)
        self.play(FadeIn(tau_footnote))
        self.wait(2.0)
