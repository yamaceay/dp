"""Scene 4: The Savings Theorem — two compounding gains.

Shows (a) the cumulative anonymity criterion stopping at θ=0.90,
then (b) the closed-form savings: α*(θ)=1-√(1-θ) and ρ*(θ)<1,
with the dramatic τ=0.1 numbers from risk_weights §5.4 / §6.
"""
from manim import *
import numpy as np

TUB_RED    = ManimColor("#c50e1f")
TUB_ORANGE = ManimColor("#e87722")
TUB_BLUE   = ManimColor("#1a7abf")
TUB_GRAY   = ManimColor("#aaaaaa")
BG         = ManimColor("#0f0f0f")

# Tokens sorted by descending risk (as processed)
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

THETA = 0.90   # anonymity target θ
TAU   = 0.1    # risk temperature τ (paper default)


def N_integral(alpha, tau):
    """Top-α weight mass: ∫_{1-α}^{1} exp(-v/τ) dv = τ(exp(-(1-α)/τ) - exp(-1/τ))"""
    return tau * (np.exp(-(1 - alpha) / tau) - np.exp(-1.0 / tau))


class SavingsTheorem(Scene):
    def construct(self):
        self.camera.background_color = BG
        n     = len(TOKENS_SORTED)
        risks = np.array([r for _, r in TOKENS_SORTED])
        total = risks.sum()
        cum   = np.cumsum(risks) / total   # A_m

        # ── Part 1: risk-bar chart + cumulative A_m curve ─────────
        title = Text("The Savings Theorem: Two Compounding Gains",
                     font_size=28, color=WHITE).to_edge(UP, buff=0.4)
        self.play(Write(title))

        bar_max_h = 2.2
        bar_w     = 0.52
        spacing   = 0.75
        x0        = -(n * spacing) / 2 + spacing / 2
        base_y    = -0.8

        bars      = VGroup()
        tok_lbls  = VGroup()
        for i, (word, risk) in enumerate(TOKENS_SORTED):
            h   = bar_max_h * risk
            bar = Rectangle(width=bar_w, height=h,
                            fill_color=TUB_GRAY, fill_opacity=0.5,
                            stroke_width=0)
            bar.move_to([x0 + i * spacing, base_y + h / 2, 0])
            bars.add(bar)
            lbl = Text(word, font_size=11, color=TUB_GRAY
                       ).move_to([x0 + i * spacing, base_y - 0.22, 0])
            tok_lbls.add(lbl)

        x_axis = Line(
            [x0 - spacing / 2, base_y, 0],
            [x0 + (n - 0.5) * spacing, base_y, 0],
            color=TUB_GRAY, stroke_width=1,
        )
        y_label = Text("Risk  r_i", font_size=13, color=TUB_GRAY
                       ).rotate(PI / 2).next_to(bars, LEFT, buff=0.25)
        subtitle = Text("Tokens sorted by SHAP risk (riskiest first)",
                        font_size=16, color=TUB_GRAY).next_to(title, DOWN, buff=0.2)

        self.play(
            FadeIn(subtitle), FadeIn(x_axis), FadeIn(y_label),
            LaggedStart(*[GrowFromEdge(b, DOWN) for b in bars], lag_ratio=0.07),
            LaggedStart(*[FadeIn(l) for l in tok_lbls], lag_ratio=0.07),
            run_time=1.8,
        )
        self.wait(0.4)

        # ── Cumulative A_m curve ──────────────────────────────────
        curve_pts = [[x0 - spacing / 2, base_y, 0]] + [
            [x0 + i * spacing, base_y + bar_max_h * cum[i], 0]
            for i in range(n)
        ]
        curve = VMobject(color=YELLOW, stroke_width=2.5)
        curve.set_points_as_corners(curve_pts)
        cum_lbl = Text("Cumulative A_m", font_size=14, color=YELLOW
                       ).next_to(curve_pts[-1], RIGHT, buff=0.1).shift(UP * 0.1)
        self.play(Create(curve), run_time=1.6)
        self.play(FadeIn(cum_lbl))
        self.wait(0.35)

        # ── θ threshold line ──────────────────────────────────────
        theta_y = base_y + bar_max_h * THETA
        x_left  = x0 - spacing / 2
        x_right = x0 + (n - 0.5) * spacing
        theta_line = DashedLine(
            [x_left, theta_y, 0], [x_right, theta_y, 0],
            color=TUB_RED, stroke_width=2, dash_length=0.15,
        )
        theta_lbl = MathTex(rf"\theta = {THETA}", font_size=22, color=TUB_RED
                            ).next_to(theta_line, RIGHT, buff=0.12)
        self.play(Create(theta_line), FadeIn(theta_lbl))
        self.wait(0.4)

        # ── Colour bars that get rewritten ────────────────────────
        stop_idx = int(np.searchsorted(cum, THETA))
        stop_idx = min(stop_idx, n - 1)

        replace_anims = []
        for i in range(stop_idx + 1):
            new_bar = bars[i].copy().set_fill(color=TUB_RED, opacity=0.85)
            replace_anims.append(Transform(bars[i], new_bar))
            replace_anims.append(tok_lbls[i].animate.set_color(TUB_RED))
        self.play(LaggedStart(*replace_anims, lag_ratio=0.10), run_time=1.4)

        # Cutoff marker
        cut_x   = x0 + stop_idx * spacing + spacing / 2
        cut_line = DashedLine(
            [cut_x, base_y - 0.1, 0], [cut_x, theta_y + 0.1, 0],
            color=TUB_ORANGE, stroke_width=1.5, dash_length=0.12,
        )
        n_stop  = stop_idx + 1
        alpha_actual = n_stop / n
        cut_lbl  = Text(f"α = {alpha_actual:.0%} of tokens", font_size=12,
                        color=TUB_ORANGE).next_to(cut_line, UP, buff=0.05)
        self.play(Create(cut_line), FadeIn(cut_lbl))
        self.wait(0.6)

        # ── Transition to savings panel ───────────────────────────
        self.play(
            FadeOut(bars), FadeOut(tok_lbls), FadeOut(curve), FadeOut(cum_lbl),
            FadeOut(x_axis), FadeOut(y_label), FadeOut(subtitle),
            FadeOut(theta_line), FadeOut(theta_lbl),
            FadeOut(cut_line), FadeOut(cut_lbl),
            run_time=0.8,
        )
        self.wait(0.2)

        # ── Part 2: The savings formulas ──────────────────────────
        # Gain 1: fewer tokens  (α* < θ)
        gain1_title = Text("Gain 1 — Fewer tokens rewritten", font_size=20,
                           color=TUB_ORANGE).to_edge(UP, buff=1.0)
        gain1_formula = MathTex(
            r"\frac{M_\theta}{n} \;\longrightarrow\; \alpha^*(\theta) "
            r"= 1 - \sqrt{1-\theta} \;<\; \theta",
            font_size=28, color=YELLOW,
        ).next_to(gain1_title, DOWN, buff=0.25)

        # Gain 2: cheaper tokens (budget curve below α line)
        gain2_title = Text("Gain 2 — Riskiest tokens carry the smallest budgets",
                           font_size=20, color=TUB_ORANGE
                           ).next_to(gain1_formula, DOWN, buff=0.40)
        gain2_formula = MathTex(
            r"\frac{\mathbb{E}[S_{M_\theta}]}{\varepsilon n} \;\longrightarrow\; "
            r"\frac{N(\alpha^*)}{N(1)} \;<\; \alpha^* \;<\; \theta",
            font_size=26, color=YELLOW,
        ).next_to(gain2_title, DOWN, buff=0.20)
        N_def = MathTex(
            r"N(\alpha) = \tau\!\left(e^{-(1-\alpha)/\tau} - e^{-1/\tau}\right)",
            font_size=20, color=TUB_GRAY,
        ).next_to(gain2_formula, DOWN, buff=0.15)

        self.play(FadeIn(gain1_title), Write(gain1_formula))
        self.wait(0.5)
        self.play(FadeIn(gain2_title), Write(gain2_formula))
        self.play(Write(N_def))
        self.wait(1.0)

        # ── Part 3: worked numbers for τ=0.1 ─────────────────────
        self.play(
            FadeOut(gain1_title), FadeOut(gain1_formula),
            FadeOut(gain2_title), FadeOut(gain2_formula), FadeOut(N_def),
            run_time=0.6,
        )

        tau_note = Text("τ = 0.1  (paper default  —  most aggressive redistribution)",
                        font_size=18, color=TUB_ORANGE).to_edge(UP, buff=0.8)
        self.play(FadeIn(tau_note))
        self.wait(0.2)

        # θ=0.90 highlight: budget comparison bars
        panel_center = ORIGIN + DOWN * 0.2

        # Compute exact numbers
        alpha_star = 1 - np.sqrt(1 - THETA)           # ≈ 0.684
        N1         = N_integral(1.0, TAU)
        Na         = N_integral(alpha_star, TAU)
        our_frac   = Na / N1                           # ≈ 0.0423
        unif_frac  = THETA                             # 0.90
        rho_star   = our_frac / unif_frac              # ≈ 0.047

        bar_max = 3.5
        gap     = 2.5
        bw      = 1.0

        our_h   = bar_max * our_frac
        unif_h  = bar_max * unif_frac
        our_base_y  = panel_center[1] - bar_max / 2 + our_h / 2 - 0.3
        unif_base_y = panel_center[1] - bar_max / 2 + unif_h / 2 - 0.3

        our_bar = Rectangle(
            width=bw, height=our_h,
            fill_color=TUB_BLUE, fill_opacity=0.85, stroke_width=0,
        ).move_to([-gap / 2, our_base_y, 0])
        unif_bar = Rectangle(
            width=bw, height=unif_h,
            fill_color=TUB_GRAY, fill_opacity=0.50,
            stroke_color=TUB_GRAY, stroke_width=0.8,
        ).move_to([gap / 2, unif_base_y, 0])

        base_line = Line(
            [-gap / 2 - 0.8, panel_center[1] - bar_max / 2 - 0.3, 0],
            [ gap / 2 + 0.8, panel_center[1] - bar_max / 2 - 0.3, 0],
            color=TUB_GRAY, stroke_width=1,
        )

        our_lbl   = Text(f"DP-MLM-X  (τ=0.1)\n{our_frac:.1%} of budget",
                         font_size=15, color=TUB_BLUE
                         ).next_to(our_bar, DOWN, buff=0.15)
        unif_lbl  = Text(f"Uniform baseline\n{unif_frac:.0%} of budget",
                         font_size=15, color=TUB_GRAY
                         ).next_to(unif_bar, DOWN, buff=0.15)

        our_pct   = Text(f"{our_frac:.1%}", font_size=20, color=TUB_BLUE, weight=BOLD
                         ).next_to(our_bar, UP, buff=0.08)
        unif_pct  = Text(f"{unif_frac:.0%}", font_size=20, color=TUB_GRAY
                         ).next_to(unif_bar, UP, buff=0.08)

        self.play(
            FadeIn(base_line),
            GrowFromEdge(our_bar, DOWN),
            GrowFromEdge(unif_bar, DOWN),
            run_time=1.5,
        )
        self.play(
            FadeIn(our_lbl), FadeIn(unif_lbl),
            FadeIn(our_pct), FadeIn(unif_pct),
        )
        self.wait(0.4)

        # Saving ratio callout
        saving_text = Text(
            f"Saving ratio  ρ* = {rho_star:.3f}  —  spend ×{1/rho_star:.0f} less",
            font_size=22, color=GREEN, weight=BOLD,
        ).to_edge(DOWN, buff=0.55)
        both_label = Text(
            f"Both reach θ = {THETA:.0%} anonymity target",
            font_size=17, color=WHITE,
        ).next_to(saving_text, UP, buff=0.20)
        self.play(Write(both_label), Write(saving_text))

        # Arrow between bars
        arrow = DoubleArrow(
            our_bar.get_top() + UP * 0.1,
            unif_bar.get_top() + UP * 0.1,
            color=GREEN, buff=0.1, stroke_width=2.0,
        ).shift(UP * 0.3)
        ratio_lbl = Text(f"×{1/rho_star:.0f}", font_size=24, color=GREEN, weight=BOLD
                         ).next_to(arrow, UP, buff=0.06)
        self.play(Create(arrow), FadeIn(ratio_lbl))
        self.wait(3.0)
