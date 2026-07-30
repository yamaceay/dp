 ▐▛███▜▌   Claude Code v2.1.198
▝▜█████▛▘  Sonnet 4.6 · API Usage Billing
  ▘▘ ▝▝    ~/work/dp/docs/defense/animations

 ▎ Fable 5 is back.
 ▎ Until July 7, you can use up to 50% of your plan's weekly usage limit on Fable 5. If you hit your limit, you can continue on Fable 5 with usage credits. Fable 5 draws down usage faster than Opus 4.8. Learn more

⏺ Fixed. Summary of changes:

  - "What We Measure" slide: removed the wrong "PU = privacy × utility"
  definition; now correctly lists MRR, U (utility), PU (pseudo-utility = U
  without retraining), and relative gain as the single combined number mentioned
  verbally
  - All speaker notes: PU is described as pseudo-utility, relative gain is the
  combined metric cited as a headline number during the talk
  - Takeaway table: "highest PU score" → "best relative gain"

✻ Cogitated for 2m 6s

❯ /compact                                                                      
  ⎿  Compacted (ctrl+o to see full summary)
  ⎿  Read ../../exp.py (72 lines)
  ⎿  Read main.typ (378 lines)
  ⎿  Read ../thesis/presi/main.typ (256 lines)

❯ i think pseudo-utility and test-split oracle are not relevant for the         
  presentation at all. just show the privacy-utility curves of the baselines    
  adn my methoesd                                                               

⏺ Let me check what tradeoff plot files are available.

  Searched for 1 pattern (ctrl+o to expand)

⏺ Good. Now I'll also check the summary image and what's in
  simple_maskers_and_rewriters to understand the baseline grouping, then update
  the slides.

  Listed 1 directory (ctrl+o to expand)

⏺ The summary U_acc.png is the clean utility (not PU) overview. Now I'll update
  the three result slides to use u_nominal.png and drop PU and the summary PU
  image:

⏺ Update(~/work/dp/docs/defense/main.typ)
  ⎿  File must be read first

  Read 1 file (ctrl+o to expand)

⏺ Update(~/work/dp/docs/defense/main.typ)
  ⎿  Removed 1 line
      232    #v(0.4em)
      233    - *MRR* — re-identification risk (lower = better)
      234    - *U* — downstream task utility (higher = better)
      235 -  - *PU* — pseudo-utility: U evaluated without retraining     
      235    - *Relative gain* — single combined improvement over baselin
           e
      236  ][
      237    *Two datasets*

⏺ Update(~/work/dp/docs/defense/main.typ)
  ⎿  Added 4 lines, removed 5 lines
      258  
      259  #speaker-note[
      260    MRR = 1 means the model immediately finds the person. MRR → 
           0 means the author is
      261 -  effectively hidden in the crowd. U is downstream task accura
          -cy. PU is pseudo-utility —                                    
      262 -  the same accuracy measured without retraining, which is hard
          -er to maintain.                                               
      263 -  Relative gain captures the combined improvement over the bas
          -eline as a single number —                                    
      264 -  mentioned during the talk rather than plotted. DB-Bio is our
          - secondary validation                                         
      265 -  dataset; it confirms results from TAB.                      
      261 +  effectively hidden in the crowd. U is downstream task accura
          +cy after retraining on                                        
      262 +  anonymized data. Relative gain captures the combined privacy
          +-utility improvement over                                     
      263 +  the baseline as a single number — mentioned verbally, not pl
          +otted.                                                        
      264 +  DB-Bio is our secondary validation dataset; it confirms resu
          +lts from TAB.                                                 
      265  ]
      266  
      267  == Risk-Guided Allocation Outperforms Uniform DP-MLM

⏺ Update(~/work/dp/docs/defense/main.typ)
  ⎿  Added 15 lines, removed 18 lines
      266  
      267  == Risk-Guided Allocation Outperforms Uniform DP-MLM
      268  
      269 -#image("../thesis/images/summary/tab/P_exact/PU_acc.png", widt
          -h: 100%)                                                      
      269 +#image("../thesis/images/summary/tab/P_exact/U_acc.png", width
          +: 100%)                                                       
      270  
      271  #speaker-note[
      272 -  This shows pseudo-utility (PU) for all methods on TAB. DP-ML
          -M-X with SHAP allocation                                      
      273 -  consistently outperforms uniform DP-MLM at every epsilon lev
          -el. In terms of relative                                      
      274 -  gain — the single combined improvement metric — DP-MLM-X lea
          -ds across the board.                                          
      275 -  DB-Bio confirms the same pattern.                           
      272 +  Privacy (MRR) vs. utility (U) for all methods on TAB. DP-MLM
          +-X with SHAP allocation                                       
      273 +  consistently sits above uniform DP-MLM — stronger privacy at
          + the same or higher utility.                                  
      274 +  In terms of relative gain it leads across the board. DB-Bio 
          +confirms the same pattern.                                    
      275  ]
      276  
      277  == k-DP-MLM-X: Full Privacy Range, Fewest Substitutions
      278  
      280 -#image("../thesis/images/tradeoff/tab/token_level_rewriters_wi
          -th_threshold_k/p_exact/pu_nominal.png", width: 100%)          
      279 +#image("../thesis/images/tradeoff/tab/token_level_rewriters_wi
          +th_threshold_k/p_exact/u_nominal.png", width: 100%)           
      280  
      281  #speaker-note[
      283 -  Each point is a (k, epsilon) combination on the privacy–pseu
          -do-utility trade-off.                                         
      284 -  k-DP-MLM-X covers the full privacy range while rewriting onl
          -y about 6% of tokens at                                       
      285 -  k=2. The exhaustive re-evaluation stops exactly when needed:
          - easy records barely get                                      
      286 -  touched. k can be calibrated in advance against a target ano
          -nymity level.                                                 
      287 -  The relative gain over the baseline is the headline number t
          -o cite here.                                                  
      282 +  Privacy (MRR) vs. utility (U) trade-off for k-DP-MLM-X again
          +st baselines on TAB.                                          
      283 +  Each point is a (k, epsilon) combination. k-DP-MLM-X covers 
          +the full privacy range                                        
      284 +  while rewriting only about 6% of tokens at k=2. The exhausti
          +ve re-evaluation stops                                        
      285 +  exactly when needed — easy records barely get touched.      
      286    DB-Bio shows the same pattern.
      287  ]
      288  
      291 -== ρ-DP-MLM-X: Best Privacy-Utility Combined Score            
      289 +== ρ-DP-MLM-X: Best Privacy-Utility Trade-off                 
      290  
      293 -#image("../thesis/images/tradeoff/tab/token_level_rewriters_wi
          -th_threshold_rho/p_exact/pu_nominal.png", width: 100%)        
      291 +#image("../thesis/images/tradeoff/tab/token_level_rewriters_wi
          +th_threshold_rho/p_exact/u_nominal.png", width: 100%)         
      292  
      293  #speaker-note[
      296 -  Each point is a (rho, epsilon) combination on the privacy–ps
          -eudo-utility trade-off.                                       
      297 -  rho-DP-MLM-X achieves the best relative gain of all evaluate
          -d methods — strong privacy                                    
      298 -  reduction with minimal pseudo-utility loss. One rho value an
          -onymizes the entire corpus                                    
      299 -  consistently. The simplification costs nothing: rho still ou
          -tperforms all baselines.                                      
      300 -  DB-Bio again confirms the same pattern.                     
      294 +  Privacy (MRR) vs. utility (U) trade-off for rho-DP-MLM-X aga
          +inst baselines on TAB.                                        
      295 +  Each point is a (rho, epsilon) combination. rho-DP-MLM-X ach
          +ieves the best combined                                       
      296 +  score — strong privacy reduction with minimal utility loss —
          + and anonymizes the whole                                     
      297 +  corpus in one pass. DB-Bio again confirms the same pattern. 
      298  ]
      299  
      300  // ─────────────────────────────────────────────

⏺ Also fix the stale "MRR and PU score" reference in the takeaway slide:

⏺ Update(~/work/dp/docs/defense/main.typ)
  ⎿  Added 1 line, removed 1 line
      331    #v(0.4em)
      332  
      333    Both $k$ and $rho$ can be calibrated against a target privac
           y level before deployment.
      334 -  A parameter sweep maps directly to MRR and PU score — no gue
          -sswork.                                                       
      334 +  A parameter sweep maps directly to MRR and U — no guesswork.
      335  
      336    #v(0.5em)
      337  

⏺ Now compile to verify:

  Ran 1 shell command (ctrl+o to expand)

⏺ Clean compile. Three changes made:

  - "What We Measure": removed PU bullet, kept MRR, U, and relative gain
  - Summary result slide: switched PU_acc.png → U_acc.png
  - k- and ρ- result slides: switched pu_nominal.png → u_nominal.png (MRR vs U
  curves)
  - Takeaway slide: "MRR and PU score" → "MRR and U"
  - All speaker notes updated to say "utility (U)" throughout

✻ Baked for 1m 27s

❯ ok now, i want you to generate the manim animations that you specified before 
  on eby one. at the end, i want to be able to play the animations separately,  
  maybe then we can add this later to the slides. what do you think?            

⏺ Good plan. Let me first check what's installed, then design the four scenes
  before writing anything.

  Ran 1 shell command (ctrl+o to expand)

⏺ Good, Manim CE 0.18.1 is available. Here's the plan before I write anything:

  4 standalone animations, each ~20–40 seconds:

  1. scene_01_token_risk.py — A sentence appears token-by-token, then SHAP
  scores materialize as colored bars beneath each token, and the tokens
  themselves transition to red/orange/gray by risk level. Ends with the
  "flashlight" metaphor.
  2. scene_02_budget_allocation.py — Uniform ε bars (all equal height) morph
  into risk-weighted bars via softmax. High-risk tokens get tall bars, low-risk
  nearly nothing. Temperature τ_r = 0.1 shown as the knob controlling sharpness.
  3. scene_03_k_stopping.py — Candidate ranking list on screen. True author
  starts at rank 1. Each substitution step: one token flips (red→blue), the
  ranklist re-shuffles, author drops a rank. Animation halts with a green "STOP"
  when rank ≥ k.
  4. scene_04_rho_stopping.py — Tokens sorted by risk score shown as bars.
  Cumulative risk curve animates. The ρ threshold line appears. Tokens above it
  get marked for replacement in a single sweep — no re-evaluation loop.

  All use TUB colors (red #c50e1f, orange #e87722, blue #1a7abf) and render to
  docs/defense/animations/media/. Ready to proceed?

  Ran 1 shell command (ctrl+o to expand)
⏺ Write(scene_01_token_risk.py)
  ⎿  Wrote 117 lines to scene_01_token_risk.py
       1 """Scene 1: Token-level re-identification risk scoring via SHAP.
         """
       2 from manim import *
       3 
       4 TUB_RED    = ManimColor("#c50e1f")
       6 TUB_BLUE   = ManimColor("#1a7abf")
       7 TUB_GRAY   = ManimColor("#aaaaaa")
       8 BG         = ManimColor("#0f0f0f")
       9 
      10 TOKENS = [
     … +118 lines (ctrl+o to expand)

⏺ Write(scene_02_budget_allocation.py)
  ⎿  Wrote 144 lines to scene_02_budget_allocation.py
      1 """Scene 2: Risk-weighted budget allocation vs. uniform."""
      2 from manim import *
      3 import numpy as np
      4 
      5 TUB_RED    = ManimColor("#c50e1f")
      6 TUB_ORANGE = ManimColor("#e87722")
      7 TUB_BLUE   = ManimColor("#1a7abf")
      8 TUB_GRAY   = ManimColor("#aaaaaa")
      9 BG         = ManimColor("#0f0f0f")
     … +149 lines (ctrl+o to expand)

⏺ Write(scene_03_k_stopping.py)
  ⎿  Wrote 154 lines to scene_03_k_stopping.py
       1 """Scene 3: k-DP-MLM-X — exhaustive stopping via rank re-evaluat
         ion."""
       2 from manim import *
       3 
       4 TUB_RED    = ManimColor("#c50e1f")
       6 TUB_BLUE   = ManimColor("#1a7abf")
       7 TUB_GRAY   = ManimColor("#aaaaaa")
       8 BG         = ManimColor("#0f0f0f")
       9 
      10 K_TARGET = 3
     … +166 lines (ctrl+o to expand)

⏺ Write(scene_04_rho_stopping.py)
  ⎿  Wrote 140 lines to scene_04_rho_stopping.py
      1 """Scene 4: ρ-DP-MLM-X — single-pass cumulative risk stopping."""
      2 from manim import *
      3 import numpy as np
      4 
      5 TUB_RED    = ManimColor("#c50e1f")
      6 TUB_ORANGE = ManimColor("#e87722")
      7 TUB_BLUE   = ManimColor("#1a7abf")
      8 TUB_GRAY   = ManimColor("#aaaaaa")
      9 BG         = ManimColor("#0f0f0f")
     … +147 lines (ctrl+o to expand)

⏺ Now a render script, then render all four:

⏺ Write(render.sh)
  ⎿  Wrote 21 lines to render.sh
       1 #!/usr/bin/env bash
       2 # Render all defense animations to 1080p MP4.
       3 set -e
       4 cd "$(dirname "$0")"
       5 
       6 SCENES=(
       7   "scene_01_token_risk.py:TokenRisk"
       8   "scene_02_budget_allocation.py:BudgetAllocation"
       9   "scene_03_k_stopping.py:KStopping"
      10   "scene_04_rho_stopping.py:RhoStopping"
     … +11 lines (ctrl+o to expand)

  Ran 1 shell command (ctrl+o to expand)

⏺ Now render all four scenes (1080p takes a minute each):

  Ran 1 shell command (ctrl+o to expand)

⏺ Scene 1 done. Rendering scenes 2–4 in parallel:

  Ran 3 shell commands (ctrl+o to expand)

⏺ All four rendered cleanly. Let me confirm the output files:

  Searched for 1 pattern (ctrl+o to expand)

⏺ All four are ready at 1080p60. Here's the summary:

  ┌──────────────────────┬───────────────────────────────────────────────────┐
  │         File         │                   What it shows                   │
  ├──────────────────────┼───────────────────────────────────────────────────┤
  │ TokenRisk.mp4        │ Tokens appear → SHAP bars grow → tokens color     │
  │                      │ red/orange/gray by risk                           │
  ├──────────────────────┼───────────────────────────────────────────────────┤
  │ BudgetAllocation.mp4 │ Uniform ε bars morph into risk-weighted bars;     │
  │                      │ formula appears; highest-risk token highlighted   │
  ├──────────────────────┼───────────────────────────────────────────────────┤
  │ KStopping.mp4        │ Token flips one-by-one, ranking re-shuffles after │
  │                      │  each, green STOP at rank ≥ k=3                   │
  ├──────────────────────┼───────────────────────────────────────────────────┤
  │ RhoStopping.mp4      │ Bars sorted by risk → cumulative curve → ρ        │
  │                      │ threshold line → red sweep marks replacements     │
  └──────────────────────┴───────────────────────────────────────────────────┘

  To play them: open 
  docs/defense/animations/media/videos/scene_01_token_risk/1080p60/TokenRisk.mp4
  (or run render.sh to re-render all at once if you tweak the source files).

✻ Cogitated for 4m 30s

❯ i liked the animations. however, i would like to make the risk-weighted       
  budget allocation example a bit more intuitive. instead of epsilon budgets,   
  give temperatures, also scaled by the inverse of epsilon budgets. then we     
  will be able to just work with a more intuitive plot. i quite liked the       
  anonymization examples. in shap re-id risk per token video, i see that texts  
  overlap each other, so there is either not enough distance between words or   
  maybe we tilt the words a bit.                                                

⏺ Two clear fixes:
  1. Scene 01: tilt tokens ~45° to avoid horizontal overlap
  2. Scene 02: replace ε_i bars with noise temperature (1/ε_i) — tall bar = more
  scrambled = high risk, which reads intuitively

  Let me rework both files.

  Read 2 files (ctrl+o to expand)

⏺ Write(scene_01_token_risk.py)
  ⎿  Added 29 lines, removed 28 lines
      21      ("Berlin",   0.12, TUB_GRAY),
      22  ]
      23  
      24 +TILT = -PI / 4   # 45° clockwise — eliminates horizontal overla
         +p                                                              
      25  
      26 +                                                               
      27  class TokenRisk(Scene):
      28      def construct(self):
      29          self.camera.background_color = BG
     ...
      33          self.play(Write(title))
      34          self.wait(0.3)
      35  
      34 -        # Build token labels and bars                          
      35 -        n = len(TOKENS)                                        
      36 +        n       = len(TOKENS)                                  
      37          bar_max_h = 1.8
      38          bar_w     = 0.55
      38 -        spacing   = 0.72                                       
      39 +        spacing   = 0.9          # wider spacing to let tilted 
         +labels breathe                                                 
      40          total_w   = n * spacing
      41          x_start   = -total_w / 2 + spacing / 2
      42 +        label_y   = 0.85         # row where tilted token label
         +s sit                                                          
      43  
      44          token_labels = VGroup()
      45          bars         = VGroup()
     ...
      48          for i, (word, risk, color) in enumerate(TOKENS):
      49              x = x_start + i * spacing
      50  
      49 -            # token word (neutral white initially)             
      50 -            tok = Text(word, font_size=15, color=WHITE).move_to
         -([x, 0.5, 0])                                                  
      51 +            # tilted token label                               
      52 +            tok = (Text(word, font_size=15, color=WHITE)       
      53 +                   .rotate(TILT)                               
      54 +                   .move_to([x, label_y, 0]))                  
      55              token_labels.add(tok)
      56  
      53 -            # risk bar (zero height initially)                 
      57 +            # risk bar (zero height initially, grows downward f
         +rom base_y)                                                    
      58              bar = Rectangle(width=bar_w, height=0.001,
      59                              fill_color=color, fill_opacity=0.85
          ,
      60                              stroke_width=0)
      57 -            bar.move_to([x, -0.5, 0], aligned_edge=DOWN)       
      61 +            bar.move_to([x, 0.0, 0], aligned_edge=UP)          
      62              bars.add(bar)
      63  
      60 -            # risk score label below bar                       
      61 -            lbl = Text(f"{risk:.2f}", font_size=12, color=color
         -)                                                              
      62 -            lbl.move_to([x, -0.5 - bar_max_h * risk - 0.18, 0])
      64 +            # numeric label below bar                          
      65 +            lbl = Text(f"{risk:.2f}", font_size=11, color=color
         +)                                                              
      66 +            lbl.move_to([x, 0.0 - bar_max_h * risk - 0.22, 0]) 
      67              bar_labels.add(lbl)
      68  
      65 -        # Y-axis label                                         
      69          y_label = Text("Re-ID risk", font_size=16, color=TUB_GR
          AY
      70                         ).rotate(PI / 2).next_to(bars, LEFT, buf
          f=0.3)
      71  
      69 -        # Step 1: tokens appear                                
      72 +        # Step 1: tokens appear with stagger                   
      73          self.play(LaggedStart(*[FadeIn(t, shift=UP * 0.2) for t
           in token_labels],
      71 -                              lag_ratio=0.12), run_time=2.0)   
      72 -        self.wait(0.5)                                         
      74 +                              lag_ratio=0.10), run_time=2.0)   
      75 +        self.wait(0.4)                                         
      76  
      74 -        # Step 2: SHAP bars grow one by one                    
      77 +        # Step 2: SHAP bars grow                               
      78          subtitle = Text("SHAP scores — marginal re-ID contribut
          ion",
      79                          font_size=18, color=TUB_GRAY).next_to(t
          itle, DOWN, buff=0.15)
      80          self.play(FadeIn(subtitle), FadeIn(y_label))
      81  
      82          grow_anims = []
      80 -        for i, (word, risk, color) in enumerate(TOKENS):       
      83 +        for i, (_, risk, color) in enumerate(TOKENS):          
      84              target_h = max(bar_max_h * risk, 0.05)
      82 -            bar = bars[i]                                      
      83 -            grown = bar.copy()                                 
      85 +            grown = bars[i].copy()                             
      86              grown.stretch_to_fit_height(target_h)
      85 -            grown.move_to([bar.get_center()[0], -0.5 - target_h
         - / 2, 0])                                                      
      86 -            grow_anims.append(Transform(bar, grown))           
      87 +            grown.move_to([bars[i].get_center()[0], 0.0 - targe
         +t_h / 2, 0])                                                   
      88 +            grow_anims.append(Transform(bars[i], grown))       
      89  
      88 -        self.play(LaggedStart(*grow_anims, lag_ratio=0.10), run
         -_time=2.5)                                                     
      89 -        self.play(LaggedStart(*[FadeIn(l) for l in bar_labels],
         - lag_ratio=0.08),                                              
      90 -                  run_time=1.0)                                
      90 +        self.play(LaggedStart(*grow_anims, lag_ratio=0.09), run
         +_time=2.5)                                                     
      91 +        self.play(LaggedStart(*[FadeIn(l) for l in bar_labels],
         + lag_ratio=0.07),                                              
      92 +                  run_time=0.9)                                
      93  
      92 -        # Step 3: color the tokens by risk                     
      94 +        # Step 3: color the token labels by risk               
      95          color_anims = [token_labels[i].animate.set_color(color)
      96                         for i, (_, _, color) in enumerate(TOKENS
          )]
      97          self.play(LaggedStart(*color_anims, lag_ratio=0.08), ru
          n_time=1.5)
     ...
      105          ]
      106          legend = VGroup()
      107          for color, label in legend_items:
      106 -            dot  = Dot(color=color, radius=0.08)              
      107 -            text = Text(label, font_size=15, color=WHITE)     
      108 -            row  = VGroup(dot, text).arrange(RIGHT, buff=0.15)
      108 +            row = VGroup(Dot(color=color, radius=0.08),       
      109 +                         Text(label, font_size=15, color=WHITE
          +)).arrange(RIGHT, buff=0.15)                                  
      110              legend.add(row)
      111          legend.arrange(DOWN, aligned_edge=LEFT, buff=0.18
      111 -                       ).to_corner(UR, buff=0.5).shift(DOWN * 
          -0.6)                                                          
      112 +                       ).to_corner(UR, buff=0.45).shift(DOWN *
          + 0.5)                                                         
      113          self.play(FadeIn(legend))
      114  
      115          takeaway = Text("Highest-risk tokens fill the anonymiz
           ation queue first.",

⏺ Write(scene_02_budget_allocation.py)
  ⎿  Added 70 lines, removed 62 lines
       1 -"""Scene 2: Risk-weighted budget allocation vs. uniform."""    
       1 +"""Scene 2: Risk-weighted noise temperature vs. uniform.       
       2 +                                                               
       3 +Bar height = sampling temperature ∝ 1/ε_i.                     
       4 +Tall bar = noisier replacement = more private.                 
       5 +High-risk tokens → tall bars (more scrambled).                 
       6 +Low-risk tokens  → short bars (barely touched).                
       7 +"""                                                            
       8  from manim import *
       9  import numpy as np
      10  
     ...
       14  TUB_GRAY   = ManimColor("#aaaaaa")
       15  BG         = ManimColor("#0f0f0f")
       16  
       11 -# Simplified 8-token example                                  
       17  RISK_SCORES = np.array([0.92, 0.81, 0.70, 0.55, 0.48, 0.20, 0.
           12, 0.05])
       18  WORDS       = ["John", "1987", "Munich", "senior", "engineer",
            "from", "Berlin", "and"]
       19  EPSILON     = 10.0
       15 -TAU         = 0.1                                             
       20 +TAU_R       = 0.1                                             
       21  
       22  
       23  def softmax_weights(risks, tau):
       24      inv = risks.max() - risks / (risks.max() - risks.min() + 1
           e-9)
       20 -    a = np.exp(inv / tau)                                     
       25 +    a   = np.exp(inv / tau)                                   
       26      return a / a.sum()
       27  
       28  
       29 +def bar_color(risk):                                          
       30 +    if risk >= 0.7: return TUB_RED                            
       31 +    if risk >= 0.4: return TUB_ORANGE                         
       32 +    return TUB_GRAY                                           
       33 +                                                              
       34 +                                                              
       35  class BudgetAllocation(Scene):
       36      def construct(self):
       37          self.camera.background_color = BG
       38          n = len(RISK_SCORES)
       39  
       29 -        title = Text("Risk-Weighted Budget Allocation",       
       40 +        title = Text("Risk-Weighted Noise Temperature",       
       41                       font_size=30, color=WHITE).to_edge(UP, bu
           ff=0.4)
       42          self.play(Write(title))
       43  
       33 -        bar_w   = 0.55                                        
       34 -        spacing = 0.9                                         
       44 +        bar_w   = 0.6                                         
       45 +        spacing = 0.95                                        
       46          x0      = -(n * spacing) / 2 + spacing / 2
       36 -        base_y  = -0.3                                        
       37 -        max_h   = 2.2                                         
       47 +        base_y  = -0.4                                        
       48 +        max_h   = 2.6                                         
       49  
       39 -        weights     = softmax_weights(RISK_SCORES, TAU)       
       40 -        budgets_rw  = EPSILON * n * weights                   
       41 -        budget_unif = EPSILON  # each token gets same ε       
       50 +        # ── compute temperatures ────────────────────────────
          +─────                                                         
       51 +        weights   = softmax_weights(RISK_SCORES, TAU_R)       
       52 +        budgets   = EPSILON * n * weights          # ε_i      
       53 +        temps_rw  = 1.0 / budgets                 # τ_sample ∝
          + 1/ε_i                                                        
       54 +        temps_rw_norm = temps_rw / temps_rw.max() # normalise 
          +to [0,1]                                                      
       55  
       43 -        def bar_color(risk):                                  
       44 -            if risk >= 0.7:  return TUB_RED                   
       45 -            if risk >= 0.4:  return TUB_ORANGE                
       46 -            return TUB_GRAY                                   
       56 +        # uniform: all same temperature (ε_i = ε for all → τ_i
          + = 1/ε, equal)                                                
       57 +        temp_unif_norm = np.full(n, 0.35)         # display at
          + 35% of max_h                                                 
       58  
       48 -        # ── token labels ────────────────────────────────────
          -──────                                                        
       59 +        # ── token labels ────────────────────────────────────
          +─────                                                         
       60          tok_labels = VGroup(*[
       61              Text(w, font_size=16, color=WHITE
       51 -                 ).move_to([x0 + i * spacing, base_y - 0.35, 0
          -])                                                            
       62 +                 ).move_to([x0 + i * spacing, base_y - 0.4, 0]
          +)                                                             
       63              for i, w in enumerate(WORDS)
       64          ])
       65          self.play(FadeIn(tok_labels))
       66          self.wait(0.3)
       67  
       68          # ── uniform bars ────────────────────────────────────
           ─────
       58 -        unif_h = max_h * (budget_unif / (EPSILON * 2))  # norm
          -alize display                                                 
       59 -                                                              
       69          unif_bars = VGroup()
       70          for i in range(n):
       62 -            bar = Rectangle(width=bar_w, height=unif_h,       
       63 -                            fill_color=WHITE, fill_opacity=0.3
          -5,                                                            
       71 +            h   = max_h * temp_unif_norm[i]                   
       72 +            bar = Rectangle(width=bar_w, height=h,            
       73 +                            fill_color=WHITE, fill_opacity=0.3
          +0,                                                            
       74                              stroke_color=WHITE, stroke_width=0
           .8)
       65 -            bar.move_to([x0 + i * spacing, base_y + unif_h / 2
          -, 0])                                                         
       75 +            bar.move_to([x0 + i * spacing, base_y + h / 2, 0])
       76              unif_bars.add(bar)
       77  
       68 -        unif_label = Text("Uniform DP-MLM  (ε = 10 for every t
          -oken)",                                                       
       69 -                          font_size=18, color=TUB_GRAY).next_t
          -o(title, DOWN, buff=0.2)                                      
       70 -        eps_labels_u = VGroup(*[                              
       71 -            Text(f"ε={budget_unif:.0f}", font_size=12, color=T
          -UB_GRAY                                                       
       72 -                 ).next_to(unif_bars[i], UP, buff=0.05)       
       78 +        unif_label = Text("Uniform DP-MLM  — same noise for ev
          +ery token",                                                   
       79 +                          font_size=17, color=TUB_GRAY).next_t
          +o(title, DOWN, buff=0.2)                                      
       80 +        unif_temp_labels = VGroup(*[                          
       81 +            Text("τ", font_size=13, color=TUB_GRAY            
       82 +                 ).next_to(unif_bars[i], UP, buff=0.06)       
       83              for i in range(n)
       84          ])
       85  
       86          self.play(FadeIn(unif_label),
       87                    LaggedStart(*[GrowFromEdge(b, DOWN) for b in
            unif_bars],
       78 -                              lag_ratio=0.1),                 
       88 +                              lag_ratio=0.10),                
       89                    run_time=1.8)
       80 -        self.play(FadeIn(eps_labels_u))                       
       81 -        self.wait(1.0)                                        
       90 +        self.play(FadeIn(unif_temp_labels))                   
       91 +        self.wait(0.8)                                        
       92  
       83 -        # ── risk score dots above bars ──────────────────────
          -──────                                                        
       93 +        # ── risk score annotations ──────────────────────────
          +──────                                                        
       94          risk_dots = VGroup()
       95          for i, r in enumerate(RISK_SCORES):
       86 -            dot = Dot(color=bar_color(r), radius=0.10         
       87 -                      ).move_to([x0 + i * spacing,            
       88 -                                 base_y + unif_h + 0.3, 0])   
       89 -            risk_label = Text(f"r={r:.2f}", font_size=11,     
       90 -                              color=bar_color(r)).next_to(dot,
          - UP, buff=0.05)                                               
       91 -            risk_dots.add(VGroup(dot, risk_label))            
       96 +            top_y = base_y + max_h * temp_unif_norm[i] + 0.55 
       97 +            dot   = Dot(color=bar_color(r), radius=0.09).move_
          +to([x0 + i * spacing, top_y, 0])                              
       98 +            rlbl  = Text(f"r={r:.2f}", font_size=11, color=bar
          +_color(r)                                                     
       99 +                         ).next_to(dot, UP, buff=0.04)        
      100 +            risk_dots.add(VGroup(dot, rlbl))                  
      101  
       93 -        risk_title = Text("SHAP risk scores →", font_size=15, 
          -color=TUB_GRAY                                                
       94 -                          ).next_to(risk_dots, LEFT, buff=0.2)
      102 +        risk_title = Text("SHAP risk →", font_size=14, color=T
          +UB_GRAY                                                       
      103 +                          ).next_to(risk_dots, LEFT, buff=0.15
          +)                                                             
      104          self.play(FadeIn(risk_dots), FadeIn(risk_title))
      105          self.wait(0.8)
      106  
       98 -        # ── transform to risk-weighted bars ─────────────────
          -─────                                                         
      107 +        # ── transform to risk-weighted temperature bars ─────
          +──────                                                        
      108          rw_bars = VGroup()
      100 -        for i, (r, b) in enumerate(zip(RISK_SCORES, budgets_rw
          -)):                                                           
      101 -            h = max(max_h * b / (EPSILON * 2), 0.04)          
      109 +        for i, r in enumerate(RISK_SCORES):                   
      110 +            h   = max(max_h * temps_rw_norm[i], 0.05)         
      111              bar = Rectangle(width=bar_w, height=h,
      112                              fill_color=bar_color(r), fill_opac
           ity=0.85,
      113                              stroke_width=0)
      114              bar.move_to([x0 + i * spacing, base_y + h / 2, 0])
      115              rw_bars.add(bar)
      116  
      108 -        eps_labels_rw = VGroup(*[                             
      109 -            Text(f"ε={budgets_rw[i]:.1f}", font_size=11,      
      117 +        temp_labels_rw = VGroup(*[                            
      118 +            Text(f"τ={temps_rw[i]:.2f}", font_size=11,        
      119                   color=bar_color(RISK_SCORES[i])
      111 -                 ).next_to(rw_bars[i], UP, buff=0.05)         
      120 +                 ).next_to(rw_bars[i], UP, buff=0.06)         
      121              for i in range(n)
      122          ])
      123  
      115 -        rw_label = Text("DP-MLM-X  (τ = 0.1 · same total budge
          -t)",                                                          
      116 -                        font_size=18, color=TUB_RED).next_to(t
          -itle, DOWN, buff=0.2)                                         
      124 +        rw_label = Text("DP-MLM-X  — noise ∝ 1/ε_i  (high risk
          + → more noise)",                                              
      125 +                        font_size=17, color=TUB_RED).next_to(t
          +itle, DOWN, buff=0.2)                                         
      126  
      127 +        self.play(FadeOut(unif_label), FadeOut(unif_temp_label
          +s), FadeIn(rw_label))                                         
      128          self.play(
      119 -            FadeOut(unif_label), FadeOut(eps_labels_u),       
      120 -            FadeIn(rw_label),                                 
      121 -        )                                                     
      122 -        self.play(                                            
      129              *[Transform(unif_bars[i], rw_bars[i]) for i in ran
           ge(n)],
      124 -            run_time=2.0                                      
      130 +            run_time=2.2                                      
      131          )
      126 -        self.play(FadeIn(eps_labels_rw))                      
      132 +        self.play(LaggedStart(*[FadeIn(l) for l in temp_labels
          +_rw], lag_ratio=0.08),                                        
      133 +                  run_time=0.9)                               
      134          self.wait(0.5)
      135  
      129 -        # ── formula callout ─────────────────────────────────
          -─────                                                         
      136 +        # ── formula ─────────────────────────────────────────
          +──────                                                        
      137          formula = MathTex(
      131 -            r"w_i \propto \exp(-r_i / \tau_r)",               
      132 -            r"\quad \varepsilon_i = \varepsilon \cdot n \cdot 
          -w_i",                                                         
      133 -            font_size=28, color=YELLOW                        
      138 +            r"\tau_{\text{sample},i} \propto \frac{1}{\varepsi
          +lon_i}",                                                      
      139 +            r"\quad \varepsilon_i = \varepsilon \cdot n \cdot 
          +w_i,\;"                                                       
      140 +            r"\quad w_i \propto \exp(-r_i / \tau_r)",         
      141 +            font_size=26, color=YELLOW                        
      142          ).to_edge(DOWN, buff=0.5)
      143          self.play(Write(formula))
      144          self.wait(0.5)
      145  
      138 -        # Highlight first (highest-risk) bar                  
      139 -        glow = SurroundingRectangle(unif_bars[0], color=TUB_RE
          -D, buff=0.05,                                                 
      146 +        # ── call out highest-risk token ─────────────────────
          +──────                                                        
      147 +        glow = SurroundingRectangle(unif_bars[0], color=TUB_RE
          +D, buff=0.06,                                                 
      148                                      stroke_width=2)
      141 -        note = Text("Most budget → highest risk token",       
      142 -                    font_size=16, color=TUB_RED).next_to(glow,
          - RIGHT, buff=0.15)                                            
      149 +        note = Text("Highest risk → highest temperature → stro
          +ngest noise",                                                 
      150 +                    font_size=15, color=TUB_RED).next_to(glow,
          + RIGHT, buff=0.15)                                            
      151          self.play(Create(glow), FadeIn(note))
      152          self.wait(2.0)

⏺ Now render both:

  Ran 1 shell command (ctrl+o to expand)
  ⎿  1 skill available
  ⎿  Not logged in · Please run /login

✻ Sautéed for 2m 58s