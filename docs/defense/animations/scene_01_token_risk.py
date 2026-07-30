"""Scene 1: Token-level re-identification risk scoring via SHAP."""
from manim import *

TUB_RED    = ManimColor("#c50e1f")
TUB_ORANGE = ManimColor("#e87722")
TUB_BLUE   = ManimColor("#1a7abf")
TUB_GRAY   = ManimColor("#aaaaaa")
BG         = ManimColor("#0f0f0f")

TOKENS = [
    ("John",     0.92, TUB_RED),
    ("Smith",    0.88, TUB_RED),
    ("born",     0.05, TUB_GRAY),
    ("12 March", 0.81, TUB_RED),
    ("1987",     0.76, TUB_RED),
    ("Munich",   0.61, TUB_ORANGE),
    ("senior",   0.55, TUB_ORANGE),
    ("engineer", 0.48, TUB_ORANGE),
    ("Deutsche", 0.70, TUB_RED),
    ("Bahn",     0.69, TUB_RED),
    ("Berlin",   0.12, TUB_GRAY),
]

TILT = -PI / 4   # 45° clockwise — eliminates horizontal overlap


class TokenRisk(Scene):
    def construct(self):
        self.camera.background_color = BG

        title = Text("SHAP: Re-Identification Risk per Token",
                     font_size=30, color=WHITE).to_edge(UP, buff=0.4)
        self.play(Write(title))
        self.wait(0.3)

        n       = len(TOKENS)
        bar_max_h = 1.8
        bar_w     = 0.55
        spacing   = 0.9          # wider spacing to let tilted labels breathe
        total_w   = n * spacing
        x_start   = -total_w / 2 + spacing / 2
        label_y   = 0.85         # row where tilted token labels sit

        token_labels = VGroup()
        bars         = VGroup()
        bar_labels   = VGroup()

        for i, (word, risk, color) in enumerate(TOKENS):
            x = x_start + i * spacing

            # tilted token label
            tok = (Text(word, font_size=15, color=WHITE)
                   .rotate(TILT)
                   .move_to([x, label_y, 0]))
            token_labels.add(tok)

            # risk bar (zero height initially, grows downward from base_y)
            bar = Rectangle(width=bar_w, height=0.001,
                            fill_color=color, fill_opacity=0.85,
                            stroke_width=0)
            bar.move_to([x, 0.0, 0], aligned_edge=UP)
            bars.add(bar)

            # numeric label below bar
            lbl = Text(f"{risk:.2f}", font_size=11, color=color)
            lbl.move_to([x, 0.0 - bar_max_h * risk - 0.22, 0])
            bar_labels.add(lbl)

        y_label = Text("Re-ID risk", font_size=16, color=TUB_GRAY
                       ).rotate(PI / 2).next_to(bars, LEFT, buff=0.3)

        # Step 1: tokens appear with stagger
        self.play(LaggedStart(*[FadeIn(t, shift=UP * 0.2) for t in token_labels],
                              lag_ratio=0.10), run_time=2.0)
        self.wait(0.4)

        # Step 2: SHAP bars grow
        subtitle = Text("SHAP scores — marginal re-ID contribution",
                        font_size=18, color=TUB_GRAY).next_to(title, DOWN, buff=0.15)
        self.play(FadeIn(subtitle), FadeIn(y_label))

        grow_anims = []
        for i, (_, risk, color) in enumerate(TOKENS):
            target_h = max(bar_max_h * risk, 0.05)
            grown = bars[i].copy()
            grown.stretch_to_fit_height(target_h)
            grown.move_to([bars[i].get_center()[0], 0.0 - target_h / 2, 0])
            grow_anims.append(Transform(bars[i], grown))

        self.play(LaggedStart(*grow_anims, lag_ratio=0.09), run_time=2.5)
        self.play(LaggedStart(*[FadeIn(l) for l in bar_labels], lag_ratio=0.07),
                  run_time=0.9)

        # Step 3: color the token labels by risk
        color_anims = [token_labels[i].animate.set_color(color)
                       for i, (_, _, color) in enumerate(TOKENS)]
        self.play(LaggedStart(*color_anims, lag_ratio=0.08), run_time=1.5)
        self.wait(0.4)

        # Step 4: legend + takeaway
        legend_items = [
            (TUB_RED,    "High risk  (r > 0.7)"),
            (TUB_ORANGE, "Medium risk (0.4–0.7)"),
            (TUB_GRAY,   "Low risk   (< 0.4)"),
        ]
        legend = VGroup()
        for color, label in legend_items:
            row = VGroup(Dot(color=color, radius=0.08),
                         Text(label, font_size=15, color=WHITE)).arrange(RIGHT, buff=0.15)
            legend.add(row)
        legend.arrange(DOWN, aligned_edge=LEFT, buff=0.18
                       ).to_corner(UR, buff=0.45).shift(DOWN * 0.5)
        self.play(FadeIn(legend))

        takeaway = Text("Highest-risk tokens fill the anonymization queue first.",
                        font_size=19, color=YELLOW).to_edge(DOWN, buff=0.35)
        self.play(Write(takeaway))
        self.wait(2.0)
