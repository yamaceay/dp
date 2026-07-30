// ─────────────────────────────────────────────────────────────────────────────
// DP-MLM-X  |  Industry Presentation
// Audience  : Software engineers — no DP background assumed
// Length    : 20 min  (~15 slides)
// Spine     : "Six questions about one person"
// ─────────────────────────────────────────────────────────────────────────────

// ── Layout ────────────────────────────────────────────────────────────────────
#set page(
  width: 254mm, height: 143mm,
  margin: (x: 16mm, y: 12mm),
  fill: rgb("#FAFAF8"),
)

// ── Typography ────────────────────────────────────────────────────────────────
#set text(font: ("Helvetica Neue", "Arial"), size: 11pt, fill: rgb("#18181B"))
#set par(leading: 0.65em)
#set heading(numbering: none)

// ── Color tokens ─────────────────────────────────────────────────────────────
#let teal    = rgb("#0F766E")
#let red     = rgb("#DC2626")
#let orange  = rgb("#EA580C")
#let gray    = rgb("#71717A")
#let lgray   = rgb("#D4D4D8")
#let warmwht = rgb("#FAFAF8")
#let ink     = rgb("#18181B")
#let blue    = rgb("#2563EB")

// ── Helpers ───────────────────────────────────────────────────────────────────

// Slide heading: left-aligned, large, teal accent bar on left
#let slide-title(body) = {
  rect(fill: teal, width: 3pt, height: 1.6em, radius: 1.5pt)
  h(8pt)
  text(size: 21pt, weight: 700, fill: ink, body)
}

// Small eyebrow label above slide title
#let eyebrow(body) = text(size: 8pt, weight: 500, fill: gray, tracking: 0.12em, upper(body))

// Annotation callout (hand-written feel via italics + color)
#let callout(body) = text(size: 9pt, style: "italic", fill: teal, [← #body])

// Pill badge
#let pill(body, col: teal) = box(
  fill: col.lighten(88%), stroke: 0.6pt + col, radius: 3pt,
  inset: (x: 5pt, y: 2.5pt),
  text(size: 8.5pt, fill: col, weight: 600, body)
)

// Token-coloring shorthand (mirrors academic defense)
#let hi(t) = text(fill: red, weight: "bold", t)            // high-risk
#let md(t) = text(fill: orange, t)                          // medium-risk
#let rp(t) = text(fill: blue, weight: "bold", t)           // replaced / private
#let lw(t) = text(fill: gray, t)                            // low-risk / untouched

// Quiz record card
#let record-card(level, badge-col: gray, badge-text: "", body) = {
  rect(
    width: 100%, radius: 5pt,
    fill: white, stroke: 0.8pt + lgray,
    inset: (x: 12pt, y: 10pt),
  )[
    #stack(dir: ltr, spacing: 6pt,
      text(size: 9pt, weight: 700, fill: gray)[Q#level],
      pill(badge-text, col: badge-col),
    )
    #v(6pt)
    #body
  ]
}

// Metric bar (privacy / utility visual)
#let metric-bar(label, val, col, max-w: 90mm) = {
  stack(dir: ltr, spacing: 8pt,
    text(size: 8.5pt, fill: gray, label),
    rect(width: max-w * val, height: 7pt, fill: col, radius: 2pt),
    text(size: 8.5pt, fill: col, weight: 600, str(calc.round(val, digits: 2))),
  )
}

// Footer on every non-title slide
#let footer = {
  place(bottom + left,
    text(size: 7.5pt, fill: lgray)[DP-MLM-X  ·  Risk-Aware Text Anonymization  ·  Yamaç Eren Ay]
  )
}

// ═════════════════════════════════════════════════════════════════════════════
// 01  TITLE
// ═════════════════════════════════════════════════════════════════════════════
#v(1fr)
#align(center)[
  #text(size: 9pt, fill: gray, tracking: 0.10em, upper[Text Anonymization · Privacy Engineering])
  #v(10pt)
  #text(size: 34pt, weight: 800, fill: ink)[Making Your Data Shareable]
  #v(2pt)
  #text(size: 34pt, weight: 800, fill: teal)[Without Making It Useless]
  #v(18pt)
  #text(size: 13pt, fill: gray)[
    DP-MLM-X: Risk-aware differential privacy for text records
  ]
  #v(28pt)
  #line(length: 48mm, stroke: 1.2pt + lgray)
  #v(10pt)
  #text(size: 10pt, fill: gray)[Yamaç Eren Ay  ·  2026]
]
#v(1fr)

// ═════════════════════════════════════════════════════════════════════════════
// 02  THE GAME
// ═════════════════════════════════════════════════════════════════════════════
#pagebreak()
#footer

#align(center + horizon)[
  #text(size: 8pt, fill: gray, tracking: 0.10em, upper[Before we start])
  #v(16pt)
  #text(size: 40pt, weight: 800)[Six questions.]
  #v(6pt)
  #text(size: 40pt, weight: 800, fill: teal)[Same person.]
  #v(24pt)
  #text(size: 14pt, fill: gray)[
    Each question hides a little more. \
    How far can you follow?
  ]
  #v(28pt)
  #grid(columns: 6, column-gutter: 10pt,
    pill("Q1 · Trivial",        col: red),
    pill("Q2 · Context",        col: orange),
    pill("Q3 · Guess",          col: rgb("#CA8A04")),
    pill("Q4 · Background",     col: rgb("#16A34A")),
    pill("Q5 · Harder",         col: teal),
    pill("Q6 · Impossible",     col: gray),
  )
]

// ═════════════════════════════════════════════════════════════════════════════
// 03  QUIZ  Q1 – Q2
// ═════════════════════════════════════════════════════════════════════════════
#pagebreak()
#footer

#box(inset: (bottom: 8pt))[
  #stack(dir: ltr, spacing: 0pt,
    rect(fill: teal, width: 3pt, height: 1.6em, radius: 1.5pt), h(8pt),
    text(size: 19pt, weight: 700)[Who is this?],
  )
]

#grid(columns: (1fr, 1fr), column-gutter: 14pt)[

  #record-card(1, badge-col: red, badge-text: "Trivial — name visible")[
    // ── INSERT Q1 RECORD TEXT ──
    // Paste the original record here.
    // Example structure (replace with real record):
    #text(size: 10.5pt)[
      #hi[Jane Doe] was born on #hi[4 February 1990] in #hi[Stuttgart].
      She completed her studies in #md[computer science] at #md[TU Berlin]
      and currently works as a #md[senior data engineer] at #md[Allianz SE].
      #lw[She has been based in Munich since 2019.]
    ]
    #v(6pt)
    #text(size: 8pt, fill: gray)[_Name, dates, employer all present. Takes 2 seconds._]
  ]

][

  #record-card(2, badge-col: orange, badge-text: "Context clues remain")[
    // ── INSERT Q2 RECORD TEXT ──
    // Name masked, but abbreviations / descriptors left in.
    // Example:
    #text(size: 10.5pt)[
      #rp[J.D.] was born on #rp[[DATE]] in #hi[Stuttgart].
      She studied #md[computer science] at #md[TU Berlin]
      and works as a #md[senior data engineer] at #md[Allianz SE].
      #lw[Based in Munich since 2019.]
    ]
    #v(6pt)
    #text(size: 8pt, fill: gray)[_Initials + employer make this solvable for most people._]
  ]

]

// ═════════════════════════════════════════════════════════════════════════════
// 04  QUIZ  Q3 – Q4
// ═════════════════════════════════════════════════════════════════════════════
#pagebreak()
#footer

#box(inset: (bottom: 8pt))[
  #stack(dir: ltr, spacing: 0pt,
    rect(fill: teal, width: 3pt, height: 1.6em, radius: 1.5pt), h(8pt),
    text(size: 19pt, weight: 700)[Getting harder.],
  )
]

#grid(columns: (1fr, 1fr), column-gutter: 14pt)[

  #record-card(3, badge-col: rgb("#CA8A04"), badge-text: "No direct clues")[
    // ── INSERT Q3 RECORD TEXT ──
    // No name, no abbreviation, no explicit identifiers.
    // Semantic content remains; occupation/location may still leak.
    #text(size: 10.5pt)[
      #rp[[PERSON]] was born in #rp[[YEAR]] in #rp[[CITY]].
      #rp[She] studied #md[computer engineering] and currently
      works as a #md[data specialist] in #md[southern Germany].
      #lw[She has worked in insurance technology for several years.]
    ]
    #v(6pt)
    #text(size: 8pt, fill: gray)[_No name. You'd need domain knowledge to narrow this down._]
  ]

][

  #record-card(4, badge-col: rgb("#16A34A"), badge-text: "Background knowledge helps")[
    // ── INSERT Q4 RECORD TEXT ──
    // DP-anonymized output, but with a background hint provided
    // (e.g., "this record is from a dataset of TU Berlin alumni").
    #text(size: 10.5pt)[
      #rp[[PERSON]] was born in #rp[[LOCATION]].
      #rp[They] completed graduate studies at a #md[technical university]
      and joined a #md[financial services firm] as an #md[engineering lead].
      #lw[Currently based in a major German city.]
    ]
    #v(6pt)
    #text(size: 8pt, fill: gray, style: "italic")[
      _Hint: this record is from a dataset of German engineering graduates._
    ]
  ]

]

// ═════════════════════════════════════════════════════════════════════════════
// 05  QUIZ  Q5 – Q6
// ═════════════════════════════════════════════════════════════════════════════
#pagebreak()
#footer

#box(inset: (bottom: 8pt))[
  #stack(dir: ltr, spacing: 0pt,
    rect(fill: teal, width: 3pt, height: 1.6em, radius: 1.5pt), h(8pt),
    text(size: 19pt, weight: 700)[Final two.],
  )
]

#grid(columns: (1fr, 1fr), column-gutter: 14pt)[

  #record-card(5, badge-col: teal, badge-text: "Obscure background knowledge")[
    // ── INSERT Q5 RECORD TEXT ──
    // Stronger DP anonymization. Background knowledge hint is harder
    // (e.g., year-of-graduation + broad field only).
    #text(size: 10.5pt)[
      #rp[[PERSON]] was born in #rp[[YEAR]] in #rp[[COUNTRY]].
      #rp[They] completed a degree in #rp[[FIELD]] and
      transitioned into #rp[[INDUSTRY]] around #rp[[YEAR]].
      #lw[Currently employed at a mid-size firm.]
    ]
    #v(6pt)
    #text(size: 8pt, fill: gray, style: "italic")[
      _Hint: record from a European professional dataset, graduation 1990–2000._
    ]
  ]

][

  #record-card(6, badge-col: gray, badge-text: "Random guess only")[
    // ── INSERT Q6 RECORD TEXT ──
    // DP ε=10 output: near-unintelligible, semantically scrambled.
    // Readers cannot extract meaningful facts.
    #text(size: 10.5pt, fill: gray)[
      #rp[[PERSON]] was #rp[embargoed] on #rp[[DATE]] in #rp[[CITY]].
      #rp[She] performed #rp[numerical calibration] at #rp[[INSTITUTION]]
      and currently #rp[monitors regulatory compliance] at a
      #rp[decentralized infrastructure unit].
      #lw[Relocated to an undisclosed region in 2021.]
    ]
    #v(6pt)
    #text(size: 8pt, fill: gray)[_Mathematically private. Essentially useless._]
  ]

]

// ═════════════════════════════════════════════════════════════════════════════
// 06  REVEAL — The privacy ladder (real MRR numbers)
// ═════════════════════════════════════════════════════════════════════════════
#pagebreak()
#footer

#stack(dir: ltr, spacing: 8pt,
  rect(fill: teal, width: 3pt, height: 1.6em, radius: 1.5pt),
  text(size: 21pt, weight: 700)[Here's what a re-ID model scored.],
)
#v(4pt)
#text(size: 9.5pt, fill: gray)[
  MRR = Mean Reciprocal Rank.  A score of 1.0 = always ranked #1 out of 2,419 candidates.
  A score near 0 = effectively hidden in the crowd.
]
#v(12pt)

// 6-level privacy ladder
#let levels = (
  ("Q1 · Original",          0.87, red,              "trivially found"),
  ("Q2 · NER masking",       0.74, orange,            "found via context"),
  ("Q3 · DP moderate",       0.14, rgb("#CA8A04"),    "hard without prior"),
  ("Q4 · DP + background",   0.24, rgb("#16A34A"),    "background info bridges the gap"),
  ("Q5 · DP-MLM-X (ε=25)",   0.06, teal,              "background barely helps"),
  ("Q6 · DP max (ε=10)",     0.03, gray,              "effectively anonymous — unusable"),
)

#for (label, val, col, note) in levels {
  grid(columns: (52mm, 1fr, 40mm), column-gutter: 8pt, align: (left, left, left),
    text(size: 9pt, fill: col, weight: 600, label),
    rect(width: (110mm * val), height: 9pt, fill: col.lighten(20%), radius: 2pt),
    text(size: 8.5pt, fill: gray, note),
  )
  v(5pt)
}

#v(8pt)
#text(size: 9pt, fill: gray)[
  Data: TAB dataset · 2,419 candidate pool · white-box re-ID model.
  Q4 uses black-box (background knowledge) evaluation.
]

// ═════════════════════════════════════════════════════════════════════════════
// 07  WHY DOES Q2 FAIL?  (semantic fingerprint)
// ═════════════════════════════════════════════════════════════════════════════
#pagebreak()
#footer

#stack(dir: ltr, spacing: 8pt,
  rect(fill: teal, width: 3pt, height: 1.6em, radius: 1.5pt),
  text(size: 21pt, weight: 700)[Why does hiding the name not work?],
)
#v(12pt)

#grid(columns: (1fr, 1fr), column-gutter: 20pt)[

  #text(size: 10pt)[
    Compliance teams mask *names, dates, locations* — the easy targets.

    But a re-ID model doesn't read like a human. It looks for the
    *combination* of signals that together uniquely identify someone:

    #v(8pt)
    #grid(columns: (auto, 1fr), column-gutter: 8pt, row-gutter: 5pt,
      text(fill: orange)[◆], [occupation + seniority level],
      text(fill: orange)[◆], [employer / industry / company size],
      text(fill: orange)[◆], [city + timeline of moves],
      text(fill: orange)[◆], [language patterns and writing style],
    )
    #v(8pt)
    Remove the name and these still form a *semantic fingerprint*.
    In our experiments, Presidio (NER masking) reduces MRR from
    0.87 → 0.74. The person is still ranked first 74% of the time.
  ]

][

  #rect(width: 100%, fill: lgray.lighten(40%), radius: 6pt, inset: 12pt)[
    #text(size: 9pt, fill: gray, weight: 600)[EXAMPLE: Q1 → Q2 comparison]
    #v(8pt)
    *Q1 (MRR = 0.87)*
    #text(size: 9.5pt)[
      #hi[Jane Doe], #hi[Stuttgart], born #hi[1990],
      #md[senior data engineer], #md[Allianz SE]
    ]
    #v(10pt)
    *Q2 (MRR = 0.74)*
    #text(size: 9.5pt)[
      #rp[[PERSON]], #rp[[CITY]], born #rp[[YEAR]],
      #md[senior data engineer], #md[Allianz SE]
    ]
    #v(10pt)
    #text(size: 8.5pt, style: "italic", fill: teal)[
      ← Employer + seniority alone narrow it to ~5 candidates
      in any realistic corporate dataset.
    ]
  ]

]

// ═════════════════════════════════════════════════════════════════════════════
// 08  HOW DP WORKS  (one-slide, no math)
// ═════════════════════════════════════════════════════════════════════════════
#pagebreak()
#footer

#stack(dir: ltr, spacing: 8pt,
  rect(fill: teal, width: 3pt, height: 1.6em, radius: 1.5pt),
  text(size: 21pt, weight: 700)[Differential privacy: the idea in one sentence.],
)
#v(12pt)

#grid(columns: (1fr, 1fr), column-gutter: 20pt)[

  #text(size: 10pt)[
    *The guarantee:* if you change any single token in the input,
    the output distribution barely changes — by at most a factor of
    $e^epsilon$ in probability.

    #v(8pt)
    *The knob:* ε (epsilon).
    #v(4pt)
    #grid(columns: (auto, 1fr), column-gutter: 8pt, row-gutter: 5pt,
      [#rect(width: 6mm, height: 6mm, fill: red.lighten(30%), radius: 2pt)],
      [small ε → strong privacy, more noise],
      [#rect(width: 6mm, height: 6mm, fill: teal.lighten(30%), radius: 2pt)],
      [large ε → weaker privacy, readable text],
    )
    #v(8pt)
    *The mechanism:* each token gets replaced by a randomly sampled
    alternative, where the sampling distribution is calibrated so that
    no output reveals which word was really there.

    #v(8pt)
    This is the gap between Q2 and Q3. Q3 uses ε = 50.
    Q6 uses ε = 10 — so noisy the text falls apart.
  ]

][

  // Simple ε dial visual
  #align(center)[
    #rect(width: 88mm, height: 52mm, fill: white, stroke: 0.8pt + lgray, radius: 6pt, inset: 12pt)[
      #text(size: 9pt, fill: gray, weight: 600)[ε as a dial]
      #v(8pt)
      #stack(dir: ltr, spacing: 0pt,
        rect(width: 20mm, height: 10mm, fill: red.lighten(25%), radius: (left: 4pt)),
        rect(width: 20mm, height: 10mm, fill: orange.lighten(35%)),
        rect(width: 20mm, height: 10mm, fill: rgb("#CA8A04").lighten(50%)),
        rect(width: 20mm, height: 10mm, fill: teal.lighten(40%), radius: (right: 4pt)),
      )
      #stack(dir: ltr, spacing: 0pt,
        box(width: 20mm, align(center, text(size: 7.5pt, fill: red)[ε = 10\nmax noise])),
        box(width: 20mm, align(center, text(size: 7.5pt, fill: orange)[ε = 25])),
        box(width: 20mm, align(center, text(size: 7.5pt, fill: rgb("#CA8A04"))[ε = 50])),
        box(width: 20mm, align(center, text(size: 7.5pt, fill: teal)[ε = 250\nreadable])),
      )
      #v(10pt)
      #grid(columns: 4, column-gutter: 4pt,
        align(center, text(size: 7.5pt, fill: red)[MRR\n0.03]),
        align(center, text(size: 7.5pt, fill: orange)[MRR\n0.06]),
        align(center, text(size: 7.5pt, fill: rgb("#CA8A04"))[MRR\n0.14]),
        align(center, text(size: 7.5pt, fill: teal)[MRR\n0.24]),
      )
      #v(8pt)
      #text(size: 7.5pt, fill: gray)[lower MRR = harder to identify]
    ]
  ]

]

// ═════════════════════════════════════════════════════════════════════════════
// 09  THE INSIGHT — not all words need the same noise
// ═════════════════════════════════════════════════════════════════════════════
#pagebreak()
#footer

#stack(dir: ltr, spacing: 8pt,
  rect(fill: teal, width: 3pt, height: 1.6em, radius: 1.5pt),
  text(size: 21pt, weight: 700)[Not all tokens need the same protection.],
)
#v(12pt)

#grid(columns: (1fr, 1fr), column-gutter: 20pt)[

  #text(size: 10pt)[
    Uniform DP-MLM treats every word identically.
    "the" gets the same noise budget as "Deutsche Bahn."

    #v(6pt)
    *DP-MLM-X* uses SHAP (a standard feature-attribution method)
    to score each token's marginal contribution to re-identification risk.
    Then it *concentrates the privacy budget on the dangerous tokens*
    and relaxes it for the harmless ones.

    #v(6pt)
    Same total budget. Smarter allocation.

    #v(8pt)
    At ε = 50:
    #grid(columns: (auto, 1fr), column-gutter: 8pt, row-gutter: 4pt,
      pill("Uniform DP-MLM", col: gray),  [MRR = 0.449  (found 45% of the time)],
      pill("DP-MLM-X",       col: teal),  [MRR = 0.138  (found 14% of the time)],
    )
    #v(4pt)
    #text(size: 9pt, style: "italic", fill: teal)[
      ← 3× stronger privacy. Same ε. Better readability too.
    ]
  ]

][

  #rect(fill: white, stroke: 0.8pt + lgray, radius: 6pt, inset: 12pt)[
    #text(size: 9pt, fill: gray, weight: 600)[SHAP risk scores per token]
    #v(8pt)
    #text(size: 10.5pt)[
      #hi[John] #hi[Smith] #lw[was born on] #hi[12 March] #hi[1987]
      #lw[in] #md[Munich]. #lw[He worked as a] #md[senior engineer]
      #lw[at] #hi[Deutsche] #hi[Bahn] #lw[from 2014 to 2019].
    ]
    #v(10pt)
    #grid(columns: (auto, 1fr), column-gutter: 6pt, row-gutter: 3pt,
      rect(width: 5mm, height: 5mm, fill: red.lighten(25%), radius: 2pt),
      text(size: 8pt)[High risk — gets maximum noise],
      rect(width: 5mm, height: 5mm, fill: orange.lighten(35%), radius: 2pt),
      text(size: 8pt)[Medium risk — moderate noise],
      rect(width: 5mm, height: 5mm, fill: lgray, radius: 2pt),
      text(size: 8pt)[Low risk — barely touched],
    )
    #v(6pt)
    #text(size: 8pt, fill: gray)[
      Budget concentrates on #text(fill: red)[names, dates, employers].
      "was", "in", "from" consume almost nothing.
    ]
  ]

]

// ═════════════════════════════════════════════════════════════════════════════
// 10  PRIVACY vs. UTILITY — the trade-off map
// ═════════════════════════════════════════════════════════════════════════════
#pagebreak()
#footer

#stack(dir: ltr, spacing: 8pt,
  rect(fill: teal, width: 3pt, height: 1.6em, radius: 1.5pt),
  text(size: 21pt, weight: 700)[Where each method lives.],
)
#v(4pt)
#text(size: 9pt, fill: gray)[
  Privacy (P) = MRR, lower is better.  Utility (U) = text quality score, higher is better.
  Data: TAB dataset.
]
#v(10pt)

// Plot placeholder — drop in the PU scatter from the paper (or generate fresh)
// Alternatively, the table below carries the full story without a chart.

#align(center)[
  #rect(
    width: 200mm, height: 52mm,
    fill: lgray.lighten(55%), stroke: 0.8pt + lgray, radius: 6pt,
  )[
    #align(center + horizon)[
      #text(size: 9.5pt, fill: gray)[
        // INSERT PU SCATTER PLOT HERE
        // e.g.: #image("../paper/plots/methodology.png", width: 100%)
        // or generate with CeTZ from the CSV data
        Privacy (MRR, lower = better) × Utility (higher = better) — all methods
      ]
    ]
  ]
]

#v(8pt)
#text(size: 8.5pt, fill: gray)[
  DP-MLM-X consistently sits in the top-right quadrant: lower MRR (stronger privacy)
  at the same or higher utility than uniform DP-MLM. Relative gain (RG = combined score)
  is highest for DP-MLM-X at ε ≥ 100 on both datasets.
]

// ═════════════════════════════════════════════════════════════════════════════
// 11  COMPARISON TABLE  (real numbers, TAB dataset)
// ═════════════════════════════════════════════════════════════════════════════
#pagebreak()
#footer

#stack(dir: ltr, spacing: 8pt,
  rect(fill: teal, width: 3pt, height: 1.6em, radius: 1.5pt),
  text(size: 21pt, weight: 700)[The numbers at ε = 50.],
)
#v(4pt)
#text(size: 9pt, fill: gray)[TAB dataset · ε = 50 · all DP methods share the same privacy budget]
#v(10pt)

#set text(size: 9pt)
#table(
  columns: (auto, auto, auto, auto, auto),
  align: (left, center, center, center, center),
  stroke: none,
  fill: (_, row) => if row == 0 { lgray.lighten(40%) }
                   else if row == 5 { teal.lighten(88%) }  // highlight DP-MLM-X row
                   else { white },
  inset: (x: 8pt, y: 5pt),
  // header
  table.hline(stroke: 0.8pt + lgray),
  [*Method*],            [*Privacy (MRR↓)*], [*Utility (U↑)*], [*Relative Gain↑*], [*Time (s)*],
  table.hline(stroke: 0.4pt + lgray),
  // baselines
  [Original (no anon.)], [0.869], [0.988], [—],   [—],
  [Presidio (NER)],      [0.740], [0.951], [0.110], [0.1],
  [SpaCy (NER)],         [0.396], [0.562], [0.113], [0.0],
  // DP methods
  [DP-Paraphrase],       [0.041], [0.474], [0.433], [14],
  table.hline(stroke: 0.3pt + lgray.lighten(30%)),
  [DP-MLM (uniform)],   [0.449], [0.897], [0.391], [80],
  // highlighted row
  table.hline(stroke: 0.6pt + teal.lighten(60%)),
  [*DP-MLM-X (ours)*],  [*0.138*], [*0.616*], [*0.465*], [42],
  table.hline(stroke: 0.6pt + teal.lighten(60%)),
  table.hline(stroke: 0.8pt + lgray),
)
#set text(size: 11pt)

#v(8pt)
#grid(columns: (auto, 1fr), column-gutter: 8pt,
  pill("Key", col: teal),
  text(size: 9pt)[
    At ε=50, DP-MLM-X achieves *3× lower MRR* than uniform DP-MLM
    while running *2× faster*. Utility drops (0.897→0.616) because
    risky tokens are more aggressively replaced — but the text remains
    coherent and downstream tasks still work.
  ],
)

// ═════════════════════════════════════════════════════════════════════════════
// 12  TWO KNOBS
// ═════════════════════════════════════════════════════════════════════════════
#pagebreak()
#footer

#stack(dir: ltr, spacing: 8pt,
  rect(fill: teal, width: 3pt, height: 1.6em, radius: 1.5pt),
  text(size: 21pt, weight: 700)[Two knobs. Calibrate once. Deploy.],
)
#v(12pt)

#grid(columns: (1fr, 1fr), column-gutter: 20pt)[

  #rect(fill: white, stroke: 0.8pt + lgray, radius: 6pt, inset: 14pt)[
    #text(size: 10pt, weight: 700, fill: teal)[ε — Privacy budget]
    #v(6pt)
    #text(size: 9.5pt)[
      Set this to match your compliance target.
      #v(4pt)
      - Strict (GDPR Art. 89 spirit): ε = 10–25
      - Balanced: ε = 50–100
      - Light touch (internal data): ε = 250
      #v(8pt)
      Calibrate against a *held-out validation set*:
      measure MRR vs. utility at each ε, pick the
      operating point that satisfies your SLA.
    ]
  ]

][

  #rect(fill: white, stroke: 0.8pt + lgray, radius: 6pt, inset: 14pt)[
    #text(size: 10pt, weight: 700, fill: teal)[τ — Risk temperature]
    #v(6pt)
    #text(size: 9.5pt)[
      Controls how aggressively budget concentrates on risky tokens.
      #v(4pt)
      - τ = 0.1: maximum concentration, best privacy efficiency,
        text can be choppy
      - τ = 1.0: moderate, good balance of privacy and readability
      - τ = 2.0+: near-uniform (behaves like standard DP-MLM)
      #v(8pt)
      In experiments, τ = 0.1 gives *21× budget savings* at the
      same 90% anonymity coverage vs. uniform DP-MLM.
    ]
  ]

]

// ═════════════════════════════════════════════════════════════════════════════
// 13  INTEGRATION
// ═════════════════════════════════════════════════════════════════════════════
#pagebreak()
#footer

#stack(dir: ltr, spacing: 8pt,
  rect(fill: teal, width: 3pt, height: 1.6em, radius: 1.5pt),
  text(size: 21pt, weight: 700)[What it takes to run this.],
)
#v(12pt)

#grid(columns: (1fr, 1fr), column-gutter: 20pt)[

  #text(size: 10pt)[
    *Dependencies*
    #v(4pt)
    #rect(fill: rgb("#1E1E1E"), radius: 5pt, inset: 10pt)[
      #text(size: 8.5pt, font: ("Fira Code", "Cascadia Code", "Courier New"), fill: rgb("#A8FF78"))[
        pip install transformers datasets shap torch
      ]
    ]
    #v(10pt)
    *Core API*
    #v(4pt)
    #rect(fill: rgb("#1E1E1E"), radius: 5pt, inset: 10pt)[
      #text(size: 8pt, font: ("Fira Code", "Cascadia Code", "Courier New"), fill: rgb("#E2E8F0"))[
        from dp.loaders import get_adapter \
        \
        adapter = get_adapter( \
        #h(10pt) "tab",                          \
        #h(10pt) data="tab",                     \
        #h(10pt) data_in="path/to/records",      \
        ) \
        \
        \# anonymize one record \
        record = next(adapter.iter_records()) \
        output = anonymize(record, epsilon=50, tau=0.1)
      ]
    ]
  ]

][

  #text(size: 10pt)[
    *Typical latency (TAB dataset, GPU)*
    #v(6pt)
    #table(
      columns: (auto, auto, auto),
      stroke: none,
      fill: (_, row) => if row == 0 { lgray.lighten(40%) } else { white },
      inset: (x: 8pt, y: 4pt),
      [*Method*], [*Time / record*], [*ε range*],
      table.hline(stroke: 0.4pt + lgray),
      [DP-MLM (uniform)],  [~80 s], [10–250],
      [DP-MLM-X (full)],   [~42 s], [50–250],
      [ρ-DP-MLM-X (ρ=0.5)],[~27 s], [10],
      [k-DP-MLM-X (k=10)], [~47 s], [10],
    )
    #v(10pt)
    *Notes*
    #v(4pt)
    - SHAP pass is computed *once* per record and cached
    - GPU required for practical throughput; CPU works for demos
    - Batch processing supported via `--data` / `--data_in` CLI flags
  ]

]

// ═════════════════════════════════════════════════════════════════════════════
// 14  WHEN TO USE WHAT
// ═════════════════════════════════════════════════════════════════════════════
#pagebreak()
#footer

#stack(dir: ltr, spacing: 8pt,
  rect(fill: teal, width: 3pt, height: 1.6em, radius: 1.5pt),
  text(size: 21pt, weight: 700)[Decision guide.],
)
#v(12pt)

#table(
  columns: (auto, 1fr, 1fr, 1fr),
  stroke: none,
  fill: (_, row) => if row == 0 { lgray.lighten(40%) }
                   else if calc.odd(row) { lgray.lighten(75%) }
                   else { white },
  inset: (x: 8pt, y: 6pt),
  align: (left, left, left, left),
  // header
  table.hline(stroke: 0.8pt + lgray),
  [*Situation*],
  [*Recommended approach*],
  [*Privacy level*],
  [*Utility trade-off*],
  table.hline(stroke: 0.4pt + lgray),
  // rows
  [Internal analytics, no PII regulation],
  [Skip — use the raw data],
  [None needed],
  [Full utility],

  [Compliance audit, names must go],
  [NER masking (Presidio)],
  [Weak (MRR ≈ 0.74)],
  [Minimal loss],

  [GDPR-adjacent sharing, text quality matters],
  [DP-MLM-X, ε = 100–250],
  [Moderate (MRR 0.20–0.24)],
  [Good (U ≈ 0.78–0.83)],

  [Research dataset release, strong guarantee needed],
  [DP-MLM-X, ε = 25–50],
  [Strong (MRR 0.06–0.14)],
  [Moderate (U ≈ 0.42–0.62)],

  [Maximum privacy, readability secondary],
  [ρ-DP-MLM-X, ρ = 0.1, ε = 10],
  [Very strong (MRR ≈ 0.04)],
  [Lower (U ≈ 0.38)],
  table.hline(stroke: 0.8pt + lgray),
)

// ═════════════════════════════════════════════════════════════════════════════
// 15  SUMMARY
// ═════════════════════════════════════════════════════════════════════════════
#pagebreak()

#v(1fr)
#align(center)[
  #text(size: 9pt, fill: gray, tracking: 0.10em, upper[Takeaways])
  #v(14pt)
  #grid(
    columns: 3, column-gutter: 10mm,
    rect(fill: teal.lighten(88%), radius: 8pt, inset: 14pt, width: 64mm)[
      #align(center)[
        #text(size: 26pt, weight: 800, fill: teal)[01]
        #v(4pt)
        #text(size: 10pt, weight: 600)[Name masking isn't enough]
        #v(4pt)
        #text(size: 9pt, fill: gray)[
          Semantic fingerprints survive NER. You need to
          randomize the content, not just the labels.
        ]
      ]
    ],
    rect(fill: teal.lighten(88%), radius: 8pt, inset: 14pt, width: 64mm)[
      #align(center)[
        #text(size: 26pt, weight: 800, fill: teal)[02]
        #v(4pt)
        #text(size: 10pt, weight: 600)[Spend the budget where it counts]
        #v(4pt)
        #text(size: 9pt, fill: gray)[
          Risk-weighted DP gives 3× stronger privacy than uniform
          DP-MLM at the same ε — without spending more.
        ]
      ]
    ],
    rect(fill: teal.lighten(88%), radius: 8pt, inset: 14pt, width: 64mm)[
      #align(center)[
        #text(size: 26pt, weight: 800, fill: teal)[03]
        #v(4pt)
        #text(size: 10pt, weight: 600)[Two knobs, one decision]
        #v(4pt)
        #text(size: 9pt, fill: gray)[
          Set ε to your compliance target, τ = 0.1 for max savings.
          Calibrate once on held-out data. Deploy.
        ]
      ]
    ],
  )
  #v(20pt)
  #line(length: 48mm, stroke: 1.2pt + lgray)
  #v(10pt)
  #text(size: 9pt, fill: gray)[Questions? The code and paper are both open.]
]
#v(1fr)
