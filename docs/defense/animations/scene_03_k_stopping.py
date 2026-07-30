"""Scene 3: Early stopping via the θ-anonymity criterion (A_m ≥ θ).

Replaces the old k-anonymity formulation with the anonymity level A_m
from risk_weights §4.3: stop as soon as the fraction of total risk
covered reaches the target θ.
"""
from manim import *
import numpy as np

TUB_RED    = ManimColor("#c50e1f")
TUB_ORANGE = ManimColor("#e87722")
TUB_BLUE   = ManimColor("#1a7abf")
TUB_GRAY   = ManimColor("#aaaaaa")
BG         = ManimColor("#0f0f0f")

# Tokens sorted by descending risk
TOKENS_SORTED = [
    ("John",     0.92),
    ("Smith",    0.88),
    ("12 March", 0.81),
    ("Deutsche", 0.70),
    ("Bahn",     0.69),
    ("Munich",   0.61),
    ("senior",   0.55),
    ("engineer", 0.48),
    ("Berlin",   0.12),
    ("and",      0.05),
]
REPLACEMENTS = ["Thomas", "Müller", "3 Oct.", "Bosch", "GmbH",
                "Hamburg", "junior", "analyst", "Berlin", "and"]
THETA = 0.90


def risk_color(r):
    if r >= 0.7: return TUB_RED
    if r >= 0.4: return TUB_ORANGE
    return TUB_GRAY


class AnonymityLevel(Scene):
    def construct(self):
        self.camera.background_color = BG
        n     = len(TOKENS_SORTED)
        risks = np.array([r for _, r in TOKENS_SORTED])
        total = risks.sum()
        cum   = np.cumsum(risks) / total   # A_m for each prefix

        title = Text("Early Stopping: The θ-Anonymity Criterion",
                     font_size=28, color=WHITE).to_edge(UP, buff=0.4)
        self.play(Write(title))

        # ── Token strip ──────────────────────────────────────────
        tok_mobs  = VGroup(*[Text(w, font_size=15, color=risk_color(r))
                              for w, r in TOKENS_SORTED])
        risk_mobs = VGroup(*[Text(f"r={r:.2f}", font_size=10, color=risk_color(r))
                              for _, r in TOKENS_SORTED])
        tok_mobs.arrange(RIGHT, buff=0.20)
        for i, rm in enumerate(risk_mobs):
            rm.next_to(tok_mobs[i], DOWN, buff=0.05)

        strip = VGroup(tok_mobs, risk_mobs).move_to(ORIGIN + UP * 2.0)
        subtitle = Text("Tokens processed riskiest-first",
                        font_size=16, color=TUB_GRAY).next_to(strip, DOWN, buff=0.15)
        self.play(FadeIn(strip), FadeIn(subtitle))
        self.wait(0.4)

        # ── A_m formula ──────────────────────────────────────────
        Am_formula = MathTex(
            r"A_m = \frac{\displaystyle\sum_{i=1}^{m} r_i}{\displaystyle\sum_{j=1}^{n} r_j}",
            r"\;\in [0,1]",
            font_size=26, color=YELLOW,
        ).to_corner(UR, buff=0.55).shift(DOWN * 0.9)
        Am_header = Text("Anonymity level", font_size=14, color=YELLOW
                         ).next_to(Am_formula, UP, buff=0.08)
        self.play(Write(Am_formula), FadeIn(Am_header))
        self.wait(0.3)

        # ── Progress bar ─────────────────────────────────────────
        bar_total_w = 7.2
        bar_h       = 0.50
        bar_center  = ORIGIN + DOWN * 0.35

        bar_outline = Rectangle(
            width=bar_total_w, height=bar_h,
            fill_opacity=0, stroke_color=TUB_GRAY, stroke_width=1.2,
        ).move_to(bar_center)
        prog_label = Text("A_m", font_size=15, color=TUB_GRAY
                          ).next_to(bar_outline, LEFT, buff=0.18)
        self.play(FadeIn(bar_outline), FadeIn(prog_label))

        # θ threshold marker
        left_x   = bar_outline.get_left()[0]
        theta_x  = left_x + bar_total_w * THETA
        center_y = bar_center[1]
        theta_line = DashedLine(
            [theta_x, center_y - bar_h / 2 - 0.15, 0],
            [theta_x, center_y + bar_h / 2 + 0.15, 0],
            color=TUB_RED, stroke_width=2.0, dash_length=0.10,
        )
        theta_lbl = MathTex(rf"\theta = {THETA}", font_size=20, color=TUB_RED
                            ).next_to(theta_line, UP, buff=0.12)
        self.play(Create(theta_line), FadeIn(theta_lbl))
        self.wait(0.3)

        # Helper: build a fill rectangle at fraction val of the bar
        def make_fill(val, color):
            w   = max(bar_total_w * val, 0.001)
            mob = Rectangle(
                width=w, height=bar_h - 0.06,
                fill_color=color, fill_opacity=0.85, stroke_width=0,
            )
            mob.move_to([left_x + w / 2, center_y, 0])
            return mob

        bar_fill    = make_fill(0.0, TUB_BLUE)
        pct_display = Text("0.000", font_size=18, color=WHITE
                           ).next_to(bar_outline, RIGHT, buff=0.18)
        self.play(FadeIn(bar_fill), FadeIn(pct_display))

        # ── Step through tokens riskiest-first ───────────────────
        stop_idx = None
        for i in range(n):
            word, risk = TOKENS_SORTED[i]
            col = risk_color(risk)

            # Flash the token about to be rewritten
            self.play(Indicate(tok_mobs[i], color=WHITE, scale_factor=1.35),
                      run_time=0.35)

            # Replace with anonymized word
            new_tok = Text(REPLACEMENTS[i], font_size=15, color=TUB_BLUE
                           ).move_to(tok_mobs[i].get_center())
            self.play(Transform(tok_mobs[i], new_tok), run_time=0.30)

            # Grow the progress bar
            fill_color = TUB_RED if cum[i] >= THETA else TUB_BLUE
            new_fill   = make_fill(cum[i], fill_color)
            new_pct    = Text(f"{cum[i]:.3f}", font_size=18, color=WHITE
                              ).next_to(bar_outline, RIGHT, buff=0.18)
            self.play(
                Transform(bar_fill, new_fill),
                Transform(pct_display, new_pct),
                run_time=0.38,
            )

            if cum[i] >= THETA:
                stop_idx = i
                break

        self.wait(0.25)

        # ── STOP banner ──────────────────────────────────────────
        n_done    = (stop_idx or 0) + 1
        stop_text = Text(
            f"STOP  —  Aₘ = {cum[stop_idx]:.2f} ≥ θ = {THETA}  ({n_done} tokens rewritten)",
            font_size=20, color=GREEN, weight=BOLD,
        )
        stop_box = SurroundingRectangle(
            stop_text, buff=0.22,
            fill_color=GREEN, fill_opacity=0.10,
            stroke_color=GREEN, stroke_width=1.8,
        )
        stop_group = VGroup(stop_box, stop_text).move_to(ORIGIN + DOWN * 1.1)
        self.play(FadeIn(stop_group, scale=1.15))
        self.wait(0.4)

        # ── Privacy filter note ──────────────────────────────────
        filter_note = VGroup(
            Text("Remaining tokens published verbatim.", font_size=15, color=TUB_GRAY),
            Text("Privacy filter cap B  →  worst-case B-DP  (Theorem 2′).",
                 font_size=15, color=TUB_GRAY),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.10).to_edge(DOWN, buff=0.30)
        self.play(Write(filter_note))
        self.wait(2.5)
