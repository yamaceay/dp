# Typst Reference for Thesis Work

## Project Structure

```
docs/thesis/
├── main.typ                        # entry point: template config + chapter includes
├── References.bib
├── chapter/
│   ├── introduction.typ
│   ├── dummy_chapter.typ           # example: theorems, equations, figures, tables
│   ├── conclusions_outlook.typ
│   ├── appendix.typ
│   ├── abstract.typ
│   └── declaration.typ
└── customization/
    ├── colors.typ                  # color palette (color1–color5, named variants)
    └── great-theorems-customized.typ
```

**`main.typ` template call:**
```typst
#import "@preview/clean-math-thesis:0.4.0": template
#import "customization/colors.typ": *

#show: template.with(
  uni-logo: image("images/logo.png", width: 60%),
  institute-logo: image("images/logo_qu.jpg", width: 100%),
  body-font: "Libertinus Serif",
  cover-font: "Libertinus Serif",
  abstract: include "chapter/abstract.typ",
  equate-settings: (breakable: true, sub-numbering: true, number-mode: "label"),
  equation-numbering-pattern: "(1.1)",
  cover-color: color1,
  heading-color: color2,
  link-color: color3,
)
```

---

## Typst Language: Three Modes

| Mode | Entry | Exit |
|------|-------|------|
| Markup (default) | — | `#expr` to inject code |
| Code | `#expr` or `#{ ... }` | `[...]` back to markup |
| Math | `$...$` | end delimiter |

**Block math** (display): `$ x^2 $` (spaces around content)
**Inline math**: `$x^2$` (no surrounding spaces)

---

## Markup Syntax

```typst
= Heading 1        == Heading 2       === Heading 3
*bold*             _italic_           `inline code`
- bullet           + numbered         / Term: definition
@label-ref                            <label-name>
\\                 (explicit line break)
```

---

## Code Mode Essentials

```typst
let x = 1                           // binding
let f(x) = 2 * x                    // named function
(x, y) => x + y                     // closure
if cond { .. } else { .. }
while x < 10 { .. }
include "bar.typ"                   // evaluates and returns content
import "bar.typ": a, b              // import specific items
import "bar.typ": a as one, b as two  // rename on import
import "bar.typ": *                 // import all
import "bar.typ": baz.a             // nested item
import "@preview/pkg:version": item // package import
```

**Destructuring:**
```typst
let (a, b) = (1, 2)
let (first, .., last) = (1, 2, 3, 4)  // .. collects remainder
let (_, y, _) = (1, 2, 3)             // _ discards
let (Homer: h) = books                 // dict key → variable h
(a, b) = (b, a)                        // swap via assignment destructuring
```

**Iteration targets:**
```typst
for x in array { .. }
for (k, v) in dict { .. }          // dict pairs (more efficient than .pairs())
for ch in "abc" { .. }             // grapheme clusters
for byte in bytes("abc") { .. }    // byte values 0–255
```

**Array/string methods:**
```typst
arr.len()                           // length
arr.map(x => x + 1)                // transform → new array
arr.filter(x => x > 0)
arr.zip(other)                      // pair elements: ((a1,b1), (a2,b2), ..)
arr.push(val)                       // mutating — must be called as method
let _ = arr.remove(0)              // discard mutating return value
..arr                               // spread as separate arguments
(1fr,) * 3                         // repeat: (1fr, 1fr, 1fr)
"a, b".split(", ").join[ --- ]
calc.min(a, b)                      // calc module for math functions
```

**Operators** (high to low precedence): `-`/`+` (unary) → `*`/`/` → `+`/`-` → `== != < <= > >= in not in` → `not` → `and` → `or` → `= += -= *= /=`

Force-end a `#` expression in markup with `;` if the next char would continue parsing: `#x;.`

---

## Document Metadata and Context

```typst
#set document(title: [My Title], author: "Name")

// Access in header or elsewhere:
context document.title
```

`context` is required because values may vary across compilation passes. Use it anywhere you need to read element properties or counters at a specific location.

---

## Grid and Layout

```typst
// Multi-column grid
#grid(
  columns: (1fr, 1fr),             // two equal columns
  row-gutter: 24pt,
  align(center)[Col A],
  align(center)[Col B],
)

// Dynamic column count
let ncols = calc.min(authors.len(), 3)
grid(
  columns: (1fr,) * ncols,
  ..authors.map(a => [#a.name \ #a.affiliation]),
)
```

**Two-column page layout:**
```typst
#set page(columns: 2)
```

**Column-spanning content** (e.g., title block over two-column body):
```typst
#place(
  top + center,
  float: true,        // reserves space; content doesn't overlap body
  scope: "parent",    // relative to page, not current column
  clearance: 2em,
)[
  // title, authors, abstract...
]
```

Without `float: true`, placed content overlaps other content without affecting flow.

---

## Templates

**Everything show rule** — wraps the whole document:
```typst
#show: template                    // if template takes one content arg
#show: doc => conf([Title], doc)   // closure when extra args needed
#show: conf.with(title: [..], abstract: lorem(80))  // cleanest form
```

**Template function pattern:**
```typst
#let conf(title: [], authors: (), abstract: [], doc) = {
  set page(paper: "us-letter", columns: 2, header: context document.title)
  set text(font: "Libertinus Serif", size: 11pt)
  set par(justify: true)
  show heading.where(level: 1): set align(center)
  // ... other rules
  place(top + center, float: true, scope: "parent")[
    // title block
  ]
  doc
}
```

**Importing a template from another file:**
```typst
#import "conf.typ": conf
#show: conf.with(title: [..], authors: (..), abstract: lorem(80))
```

Set rules inside a content block `[..]` are scoped to that block and do not leak out.

---

## Set and Show Rules

```typst
#set text(14pt)                          // applies to rest of scope
#set heading(numbering: "1.1")
#set text(red) if critical               // conditional set

#show heading: set text(navy)            // show-set
#show heading: it => [~ #emph(it.body)] // transformational
#show "badly": "great"                   // literal replacement
#show heading.where(level: 1): ..        // selector with filter
```

Scope: top-level rules apply to end of file; rules inside `{ }` apply to that block only.

**Accessing element fields in show rules:**
```typst
#show heading: it => [
  #it.body        // the heading text content
  #it.depth       // nesting level (1, 2, ...)
  #it.numbering   // numbering pattern (or none)
  #it.fields()    // dictionary of all fields
]
```
Available fields correspond to the arguments the element was constructed with. Content built without explicit arguments won't have those fields.

---

## Context System

```typst
context text.lang                        // style context
context counter(heading).get()           // location context
context here()                           // current location
counter(heading).at(<label>)             // counter at label
```

Context expressions are opaque outside their evaluation. Typst may compile multiple passes (warns after 5 if unresolved).

---

## Appendix Numbering Pattern

```typst
#set heading(numbering: none)
= Appendix
#counter(heading).update(1)
#set heading(numbering: "A.1", supplement: [Appendix])
```

---

## Math Environments (`great-theorems` + `rich-counters`)

Environments share a single `mathcounter` (inherits section level). All use `my_mathblock` base with rounded borders and inset.

| Name | Color |
|------|-------|
| `theorem` | color1 |
| `proposition` | color2 |
| `corollary` | color3 |
| `lemma` | color4 |
| `definition` | color5 |
| `remark` | color1 |
| `reminder` | color3 |
| `example` | color2 |
| `question` | color3 |
| `proof` | `proofblock()` |

Usage:
```typst
#import "../customization/great-theorems-customized.typ": *

#definition(title: "My Def")[ $e = sum_(k=0)^oo 1/k!$ ]<def:euler>
#theorem[ $e$ is irrational ]
#proof[ Left to the reader. ]
```

---

## Math — Functions and Symbols

**Greek and common symbols** — use by name directly in math mode (no `\`):
```typst
$ alpha beta gamma delta epsilon zeta eta theta $
$ arrow.r.long   arrow.l.squiggly   alef $   // symbol modifiers via dot
$ -> >= <= != $                              // shorthands
```

**Cases and matrices:**
```typst
$ f(x) = cases(
  1 "if" x > 0,
  0 "else",
) $

$ mat(
  1, 2, 3;          // ; separates rows
  4, 5, 6;
) $
```

**Delimiter scaling** — auto-scales by default (like `\left`/`\right`). Override with `lr`:
```typst
$ lr([ sum_(k=0)^n k ], size: 150%) $   // explicit size
$ \[ x \]                                // escaped = no scaling
```

**Inline code in math** — prefix with `#` to use Typst values/functions:
```typst
$ (a + b)^2 = a^2 + text(fill: #maroon, [2ab]) + b^2 $
```

**Fractions** — `/` automatically becomes a fraction; parentheses are resolved:
```typst
$ f(x) = (x + 1) / x $         // renders as proper fraction
$ sum_(k=1)^n k = (n(n+1))/2 $
```

---

## Equations (`equate`)

```typst
// labeled block equation
$
(a + b)^2 = a^2 + 2ab + b^2
$<eq:binom>

// multiline with sub-labels (requires equate-settings in template)
$
15^2 &= (10 + 5)^2 #<eq:sub1>\
     &= 225.
$<eq:multi>

@eq:binom      // reference whole equation
@eq:sub1       // reference sub-equation
```

---

## Figures, Tables, Algorithms

```typst
// figure
#figure(
  image("../images/logo.png", width: 90%),
  caption: "Caption text."
)<fig:label>

// table (programmatic)
#figure(
  caption: "Caption.",
  table(
    columns: 2, stroke: none,
    [], table.vline(stroke: .6pt), [Value],
    table.hline(stroke: .6pt),
    [$sqrt(2)$], [#calc.round(calc.sqrt(2), digits: 2)],
  )
)<table:label>

// algorithm (lovelace)
#import "@preview/lovelace:0.3.0": *
#figure(
  kind: "algorithm",
  supplement: [Algorithm],
  pseudocode-list(booktabs: true, numbered-title: [#smallcaps[My Algo]])[
    - Input: $A$
    + *for* $i in A$
      + do stuff
    + *end*
  ],
)<algo:label>
```

---

## Data Loading

```typst
#let data = json("results.json")       // → dictionary / array
#let rows = csv("data.csv")            // → array of arrays (strings)
#let cfg  = toml("config.toml")        // → dictionary
#let raw  = yaml("params.yaml")        // → dictionary / array
#let text = read("notes.txt")          // → string
```

Use with `table` or `grid` to render programmatically:
```typst
#table(
  columns: 2,
  ..rows.flatten(),                    // spread CSV rows as cells
)
```

---

## Drawing and Shapes

Basic shapes (all auto-marked as PDF artifacts; wrap in `figure` for semantic meaning):
```typst
#rect(width: 3cm, height: 1cm, fill: blue, stroke: black)
#circle(radius: 5mm, fill: red)
#ellipse(width: 4cm, height: 2cm)
#square(size: 1cm)
#line(start: (0pt, 0pt), end: (3cm, 1cm), stroke: 2pt)
```

For complex plots and diagrams, use the **CeTZ** package:
```typst
#import "@preview/cetz:0.4.1"
```

Accessible figures with alt text:
```typst
#figure(
  rect(..),
  caption: [Diagram],
  alt: "Description for screen readers",
)
```

---

## `tablem` — Markdown-like Tables

Use for hand-authored static tables; prefer programmatic `table(...)` for dynamic content.

```typst
#import "@preview/tablem:0.3.0": tablem, three-line-table

#tablem[
  | *Name* | *Score* |
  | ------ | ------- |
  | Alice  | 10      |
]

#three-line-table[
  | *Name* | *Score* |
  | :----: | :-----: |
  | Alice  | 10      |
]
```

**Cell merging:** `<` = horizontal merge, `^` = vertical merge.

**Custom render:**
```typst
#let my-table = tablem.with(
  render: (columns: auto, align: auto, ..args) => table(
    columns: columns, stroke: none,
    align: center + horizon,
    table.hline(y: 0),
    table.hline(y: 1, stroke: .5pt),
    ..args,
    table.hline(),
  )
)
```

Extra args (`fill`, `align`, `stroke`, etc.) pass through to the render function.

---

## Declaration Page

```typst
#set page(numbering: none, header: none)
#heading(level: 2, outlined: false, numbering: none)[Declaration]
#v(1cm)
#align(left)[City, #datetime.today().display()]
#align(right)[#line(length: 40%) \ Name]
```

---

## Color Palette (`customization/colors.typ`)

```typst
color1 = #800080   color2 = #0000ff   color3 = #008002
color4 = #ffa500   color5 = #ff0000

ugent-blue         caribbean-current  proper-purple
federal-blue       earth-yellow       atomic-tangerine
ugent-accent1      ugent-accent2
```

---

## Introspection

```typst
// Counters
counter(heading).get()               // current value (needs context)
counter(heading).at(<label>)         // value at a label
counter(figure.where(kind: image))   // figure counter by kind
counter(page).display()              // page number

// Querying elements
context query(heading)               // all headings → array of content
context query(selector(<my-label>))  // element at label

// State (arbitrary mutable value)
#let s = state("key", 0)
#s.update(v => v + 1)               // mutate
context s.get()                     // read (needs context)
context s.at(<label>)               // read at label

// Location
context here()                      // current location object
locate(<label>).position()          // physical position of element

// Expose value to query without visible output
#metadata("value") <my-label>
```

---

## Additional Layout

```typst
#pagebreak()                         // manual page break
#pagebreak(weak: true)               // only if not already at page start
#colbreak()                          // force next column

#pad(x: 1cm, y: 0.5cm)[content]     // add space around content
#pad(left: -1cm)[content]            // negative pad = extend into margin

#hide[content]                       // invisible but occupies space
#move(dx: 5pt, dy: -3pt)[content]   // shift without affecting layout
#rotate(45deg)[content]
#scale(x: 150%, y: 80%)[content]

// Measure content size (needs context)
context {
  let size = measure(content)
  size.width   // size.height
}

// Access container dimensions (needs context)
layout(size => [
  Container is #size.width wide.
])
```

---

## Bibliography and Citations

```typst
#bibliography("References.bib")                    // load BibTeX file
@key                                               // cite + ref combined
#cite(<key>)                                       // explicit cite
#cite(<key>, form: "prose")                        // "Author (year)" style
```

CSL styles supported. Over 80 built-in; custom `.csl` files also work.

---

## Useful Built-in Functions

```typst
#smallcaps[Text]                    // small capitals
#link("mailto:x@y.com")            // hyperlink
#lorem(80)                          // placeholder text (N words)
#box[content]                       // inline container (no line break across)
#block(below: 1.2em)[content]       // block container with spacing control
#par(justify: false)[content]       // override justification locally
#v(1cm)                             // vertical space
#h(1em)                             // horizontal space
#align(center + bottom)[content]    // 2D alignment (combine with +)
```

---

## LaTeX → Typst Quick Reference

| LaTeX | Typst |
|-------|-------|
| `\textbf{x}` | `*x*` or `#text(weight: "bold")[x]` |
| `\emph{x}` | `_x_` |
| `\textsc{x}` | `#smallcaps[x]` |
| `\texttt{x}` | `` `x` `` (raw) or `#text(font: "monospace")[x]` |
| `\url{x}` | `https://...` or `#link("x")` |
| `\label{x}` / `\ref{x}` | `<x>` / `@x` |
| `\cite{x}` | `@x` |
| `\citet{x}` | `#cite(<x>, form: "prose")` |
| `\begin{itemize}` | `- item` |
| `\begin{enumerate}` | `+ item` |
| `\begin{description}` | `/ Term: desc` |
| `\begin{cases}` | `$ cases(..) $` |
| `\begin{pmatrix}` | `$ mat(..) $` |
| `\left( \right)` | auto-scaled by default |
| `\frac{a}{b}` | `a/b` or `(a)/(b)` in math |
| `\bfseries` | `#set text(weight: "bold")` |
| `\setlength{\parindent}` | `#set par(first-line-indent: 1.8em)` |
| `\usepackage{babel}` | `#set text(lang: "de")` |
| `\textcolor{red}{x}` | `#text(fill: red)[x]` |
| `\pagebreak` | `#pagebreak()` |
| `\newcommand{\foo}` | `#let foo(..args) = ..` |
| `\renewcommand` | show rule |
| `\documentclass` | `#show: template.with(..)` |
| `\input` / `\include` | `#include "file.typ"` |
| `\usepackage{pgfplots}` | `#import "@preview/cetz:0.4.1"` |

**"LaTeX look" approximation:**
```typst
#set page(margin: 1.75in)
#set par(leading: 0.55em, spacing: 0.55em, first-line-indent: 1.8em, justify: true)
#set text(font: "New Computer Modern")
#show raw: set text(font: "New Computer Modern Mono")
#show heading: set block(above: 1.4em, below: 1em)
```

---

## Key Rules

- `#` enters code mode in markup; no additional `#` needed inside code blocks.
- Paths are relative to the invoking `.typ` file. Files outside project root cannot be read.
- Identifiers: start with letter or `_`, may contain `-` (kebab-case preferred for public names).
- Escape special chars with `\`: `\$`, `\#`, `\u{1f600}` (Unicode codepoint).
- Absolute paths start with `/` and resolve from project root; relative paths resolve from the invoking file.
- Use `context` for anything depending on document state (counters, page position, style values).
- Comments: `// line` or `/* block */` — ignored during rendering.
