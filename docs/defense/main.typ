#import "@preview/touying:0.6.1": *
#import "@preview/clear-tub:0.2.0": *

// Inline token coloring for illustrative examples
#let hi(t) = text(fill: rgb("#c50e1f"), weight: "bold", t)   // high-risk  (TUB red)
#let md(t) = text(fill: rgb("#e87722"), t)                    // medium-risk (orange)
#let rp(t) = text(fill: rgb("#1a7abf"), weight: "bold", t)   // replaced   (blue)
#let lw(t) = text(fill: rgb("#aaaaaa"), t)                    // low-risk / untouched

#show: tub-theme.with(
  aspect-ratio: "16-9",
  department: [Faculty of Electrical Engineering and Computer Science],
  logo: image("../thesis/logos/logo.png"),
  progress-bar: true,
  config-info(
    title: [DP-MLM-X],
    subtitle: [Risk-Aware Differential Privacy for Text Anonymization],
    author: [Yamaç Eren Ay],
    date: datetime(year: 2026, month: 7, day: 2),
    institution: [Technische Universität Berlin],
  ),
)

#title-slide()

#outline-slide()

// ─────────────────────────────────────────────
= Motivation
// ─────────────────────────────────────────────

== The Document That Still Identifies You

#slide(composer: (1fr, 1fr))[
  *Original biography*

  #v(0.3em)
  #quote-block[
    #hi[John Smith] was born on #hi[12 March 1987] in #hi[Munich]. \
    He worked as a #md[senior software engineer] at #md[Deutsche Bahn] \
    #lw[from 2014 to 2019 and resided in Berlin].
  ]
  #text(size: 0.65em, fill: tub-gray)[Red = high re-ID risk · Orange = medium · Grey = low]
][
  *After standard name / date masking*

  #v(0.3em)
  #quote-block[
    #rp[[PERSON]] was born on #rp[[DATE]] in #rp[[LOC]]. \
    He worked as a #md[senior software engineer] at #md[Deutsche Bahn] \
    #lw[from 2014 to 2019 and resided in Berlin].
  ]
  #v(0.4em)
  #alert-box[
    *A re-identification model still ranks him \#1 of 2 419 candidates.* \
    Occupation and employer remain untouched.
  ]
]

#speaker-note[
  Standard NER masking removes names, dates, locations — but the re-identification model
  still identifies the person from the remaining context. Occupation and employer together
  are often sufficient. This is the problem we solve.
]

== Uniform Noise Wastes the Privacy Budget

#slide(composer: (1fr, 1fr))[
  *Uniform DP-MLM — ε = 10*

  #v(0.3em)
  #quote-block[
    #rp[Maria Garcia] was born #rp[during] #rp[September 1991] #rp[near] #rp[Hamburg]. \
    He worked as #rp[the] #rp[junior analyst] #rp[over] #rp[Lufthansa] \
    from #rp[2016] to 2019 and #lw[resided in Berlin].
  ]
  #v(0.3em)
  #text(size: 0.72em)[
    - 10 tokens rewritten (45 % of content words)
    - Common words distorted: "a" → "the", "in" → "near"
    - Text becomes unnatural — utility collapses
  ]
][
  *Risk-aware DP-MLM-X — same ε = 10*

  #v(0.3em)
  #quote-block[
    #rp[Thomas Weber] was born on #rp[3 September 1990] in #rp[Stuttgart]. \
    He worked as a #lw[senior software engineer] at #rp[Siemens] \
    #lw[from 2014 to 2019 and resided in Berlin].
  ]
  #v(0.3em)
  #text(size: 0.72em)[
    - 4 tokens rewritten (18 % of content words)
    - Structure and low-risk words preserved
    - Text stays readable — utility maintained
  ]
  #v(0.4em)
  #highlight-box[Same budget. Stronger privacy. Less distortion.]
]

#speaker-note[
  Uniform DP-MLM draws noise for every token equally — it wastes budget on articles,
  prepositions, generic verbs. Risk-aware allocation concentrates the same total epsilon
  on the tokens that actually expose the person. The total budget spent is identical;
  only its distribution changes.
]

// ─────────────────────────────────────────────
= DP-MLM-X: Risk-Aware Anonymization
// ─────────────────────────────────────────────

== SHAP: A Re-Identification Risk Flashlight

#slide(composer: (1fr, 1fr))[
  #image("../thesis/images/explanations/db_bio_shap_highlight.png", width: 100%)
][
  #v(0.5em)
  #tub-block(title: [What SHAP measures])[
    How much does token $x_i$ *increase re-identification confidence*? Its marginal
    contribution to the model's prediction, averaged over all possible subsets.
  ]

  #v(0.5em)

  *Risk ranking*
  - Bright red: this word alone exposes the person
  - Blue / grey: removing it changes very little
  - Every token gets a risk score $r_i in RR$

  #v(0.4em)
  #tub-example[
    Highest-risk tokens fill the anonymization queue first.
  ]
]

#speaker-note[
  We run SHAP attribution against a DistilBERT re-identification model. The output is
  one risk score per token. This ranking drives budget allocation and stopping —
  it is the XAI component that gives DP-MLM-X its name.
]

== Risk-Weighted Budget Allocation

#slide(composer: (1fr, 1fr))[
  #v(0.3em)
  Given global budget $epsilon$ and risk scores $r_i$:

  #v(0.4em)
  #tub-block(title: [Softmax allocation])[
    $w_i prop exp(-r_i \/ tau_r) quad,quad sum_i w_i = 1$

    $epsilon_i = epsilon dot n dot w_i$

    Temperature $tau_r = 0.1$ concentrates budget on highest-risk tokens.
  ]

  #v(0.4em)

  - *Budget preserved*: $EE[epsilon_i] = epsilon$
  - Presidio hard-removes direct identifiers first
  - Risk-ranked tokens then rewritten under per-token $epsilon_i$

  #v(0.3em)
  #highlight-box[Same aggregate privacy cost as uniform — redistributed.]
][
  #image("budget_slide.png", width: 100%)
]

#speaker-note[
  The softmax turns risk scores into budget weights. High-risk tokens receive most of
  the epsilon budget and are replaced with high-noise alternatives. Low-risk tokens
  receive almost nothing and are rarely touched. Presidio handles named entities
  (names, dates, locations) as a hard pre-filter before the risk-weighted step.
]

== Two Stopping Variants

#slide(composer: (1fr, 1fr))[
  #tub-block(title: [k-DP-MLM-X — exhaustive])[
    *Stop once the record is k-anonymous.*

    After *every substitution step*, the re-identification model re-evaluates the record.
    Anonymization halts the moment the true author drops to rank $≥ k$.

    - Adapts to each record individually
    - Easy records stop after very few substitutions
    - Slower: one TRI inference per substitution step
  ]
  #v(0.4em)
  #tub-example[*Use case*: compliance — provable per-record privacy guarantee.]
][
  #tub-block(title: [ρ-DP-MLM-X — simplified])[
    *Stop once cumulative residual risk < ρ.*

    Risk scores are pre-computed *once* from the initial SHAP pass. The top-risk token
    fraction is replaced until the remaining score sum falls below $rho$.

    - Uniform fraction applied across all records
    - No re-evaluation during anonymization — fast
    - Lower $rho$ → more tokens rewritten
  ]
  #v(0.4em)
  #tub-example[*Use case*: dataset release — fast, consistent corpus anonymization.]
]

#speaker-note[
  The key distinction: k is exhaustive — it checks the re-ID model after every
  substitution. rho is a simplification — it uses the initial SHAP ranking as a proxy
  and never looks at the model again during anonymization. rho is much faster and still
  achieves the highest combined score in our experiments.
]

// ─────────────────────────────────────────────
= Results
// ─────────────────────────────────────────────

== What We Measure

#slide(composer: (1fr, 1fr))[
  *The core question*

  #v(0.3em)
  #tub-block[
    After anonymization, can a machine learning model still identify the author of the text?
  ]

  #v(0.4em)
  We train a re-identification model on the original documents. \
  It ranks all candidates. We measure where the true author lands.

  #v(0.4em)
  - *MRR* — re-identification risk (lower = better)
  - *U* — downstream task utility (higher = better)
  - *Relative gain* — single combined improvement over baseline
][
  *Two datasets*

  #v(0.3em)
  #tub-block(title: [TAB — 1 268 ECHR court judgments])[
    Long legal documents. \
    *Who is this applicant?*
  ]

  #v(0.3em)
  #tub-block(title: [DB-Bio — 2 419 Wikipedia biographies])[
    Shorter personal texts. \
    *Who is this person?*
  ]

  #v(0.4em)
  #text(size: 0.72em, fill: tub-gray)[
    Baselines: entity maskers (SpaCy, Presidio), \
    threshold maskers (PETRE), \
    DP rewriters (DP-MLM uniform, DP-BART, DP-Paraphrase)
  ]
]

#speaker-note[
  MRR = 1 means the model immediately finds the person. MRR → 0 means the author is
  effectively hidden in the crowd. U is downstream task accuracy after retraining on
  anonymized data. Relative gain captures the combined privacy-utility improvement over
  the baseline as a single number — mentioned verbally, not plotted.
  DB-Bio is our secondary validation dataset; it confirms results from TAB.
]

== Risk-Guided Allocation Outperforms Uniform DP-MLM

#image("../thesis/images/summary/tab/P_exact/U_acc.png", width: 100%)

#speaker-note[
  Privacy (MRR) vs. utility (U) for all methods on TAB. DP-MLM-X with SHAP allocation
  consistently sits above uniform DP-MLM — stronger privacy at the same or higher utility.
  In terms of relative gain it leads across the board. DB-Bio confirms the same pattern.
]

== k-DP-MLM-X: Full Privacy Range, Fewest Substitutions

#image("../thesis/images/tradeoff/tab/token_level_rewriters_with_threshold_k/p_exact/u_nominal.png", width: 100%)

#speaker-note[
  Privacy (MRR) vs. utility (U) trade-off for k-DP-MLM-X against baselines on TAB.
  Each point is a (k, epsilon) combination. k-DP-MLM-X covers the full privacy range
  while rewriting only about 6% of tokens at k=2. The exhaustive re-evaluation stops
  exactly when needed — easy records barely get touched.
  DB-Bio shows the same pattern.
]

== ρ-DP-MLM-X: Best Privacy-Utility Trade-off

#image("../thesis/images/tradeoff/tab/token_level_rewriters_with_threshold_rho/p_exact/u_nominal.png", width: 100%)

#speaker-note[
  Privacy (MRR) vs. utility (U) trade-off for rho-DP-MLM-X against baselines on TAB.
  Each point is a (rho, epsilon) combination. rho-DP-MLM-X achieves the best combined
  score — strong privacy reduction with minimal utility loss — and anonymizes the whole
  corpus in one pass. DB-Bio again confirms the same pattern.
]

// ─────────────────────────────────────────────
= Takeaway
// ─────────────────────────────────────────────

== One Framework, Two Variants

#slide(composer: (1fr, 1fr))[
  #tub-block(title: [Design principle of DP-MLM-X])[
    Concentrate DP noise where it reduces re-identification risk most. \
    Stop the moment the privacy criterion is satisfied.
  ]

  #v(0.6em)

  Both variants share a *common privacy-utility space* — \
  the same operating points are reachable.

  #v(0.5em)

  #table(
    columns: (auto, 1fr),
    stroke: none,
    inset: (y: 0.5em),
    table.hline(stroke: 0.5pt),
    [*k-DP-MLM-X*], [exhaustive · hard per-record k-anonymity guarantee],
    [*ρ-DP-MLM-X*], [simplified · best relative gain · fast corpus-wide anonymization],
    table.hline(stroke: 0.5pt),
  )
][
  *Practical use*

  #v(0.4em)

  Both $k$ and $rho$ can be calibrated against a target privacy level before deployment.
  A parameter sweep maps directly to MRR and U — no guesswork.

  #v(0.5em)

  #highlight-box[
    One SHAP pass. \
    One parameter to tune. \
    Formal DP guarantee preserved.
  ]

  #v(0.4em)
  #text(size: 0.72em, fill: tub-gray)[
    Code: #link("https://anonymous.4open.science/r/risk-aware-dp")
  ]
]

#speaker-note[
  The practitioner message: pick a variant based on operational needs (compliance vs
  bulk release), run a calibration sweep to find the right k or rho for your target
  privacy level, and deploy. The XAI component — SHAP — is the shared backbone that
  makes both variants more efficient than any uniform baseline.
]

#ending-slide(title: [Thank You — Questions?])[
  #text(size: 0.9em, weight: "bold")[Yamaç Eren Ay]

  #v(0.3cm)

  #text(size: 0.75em, fill: tub-gray)[
    Faculty of Electrical Engineering and Computer Science \
    Technische Universität Berlin
  ]

  #v(0.5cm)

  #text(size: 0.7em, fill: tub-gray)[
    _DP-MLM-X: Risk-Aware Token-Level Differential Privacy for Text Anonymization_
  ]
]
