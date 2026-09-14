# Pre-registration — LLM screeners and the memory of a prior outcome

*Written 7 September 2026, before any paid call. Prompts are frozen in
`prompts.py` (version `2026-09-07.v1`; each prompt's SHA is written to
`out/prompts_used.json` at run time). Anything not stated here is exploratory
and will be labelled as such when reported.*

## 1. The question, and what is ours

The coin-flip simulation established what the *market* does with one arbitrary
hiring decision between two equals: the gap persists (14% ± 1.3 after twenty
years, winner ahead in 70% ± 2.5) and salary anchoring carries most of it. The
decision-maker was a coin; the market did all the remembering.

An LLM screener is not a coin. It reads the whole resume, every time, and a
resume is a record of past outcomes — titles held, salaries earned, jobs won.
So the model itself may *re-transmit* a prior outcome into the next decision,
independently of any group bias. Call the size of that re-transmission the
model's **memory coefficient**, μ. Every audit in the reading list measures
which way a model tilts between groups (ε); none measures μ. Our claim is that
μ, not ε, decides whether arbitrary early luck compounds under AI-mediated
hiring — and that μ lives in the *context window*, which suggests a remedy that
is not an instruction to the model but a decision about what it is shown.

Three questions, in order:

1. **Does μ exist?** Shown two merit-equivalent candidates, one of whom won a
   job a year ago, does the model prefer the prior winner?
2. **Where does it live?** Is the preference carried by the visible salary,
   the visible title, or the bare fact of having moved — and does *removing*
   the signal from the context (redaction) work better than *telling* the model
   to ignore it (instruction)?
3. **Does it compound?** With the model choosing at every contested posting for
   twenty years, is the twenty-year gap between equals larger than under the
   coin — and does redaction bring it back?

## 2. Materials

**Pairs.** Matched twins from `twins.make_pair` (seed 0): identical level,
years of experience, degree, number of positions, breaks, promotions, latent
ability, current title, and pay within 2%; different school, field, employers,
locations and wording. Pairs that fail the audit, or that sit at a level with
no step above it, are dropped (about 7%). Twin A is the "winner" in even pairs
and twin B in odd pairs.

**Treatment (year 1).** The winner took posting J0 at year 0; the loser stayed.
Four versions of what the winner's resume now carries, against the same loser
(+2.5% in place, two new accomplishment bullets — the winner also gets exactly
two new bullets, so content is matched and only the treatment differs):

| treatment | new employer | title | pay vs own prior | visible pay gap vs loser |
|---|---|---|---|---|
| `move` | yes | same | +2.5% | 0% |
| `salary` | yes | same | +12% | +9.3% |
| `title` | yes | one level up | +2.5% | 0% |
| `both` | yes | one level up | +12% | +9.3% |

`both` is the market's actual offer and the primary treatment; the other three
decompose it. The +12% step is the same branch point used in `divergence.py`
and in the trajectory experiment, so the three studies share one shock.

**Views (redaction — what is in the context).** From `render.VIEWS`:
`full`; `no_salary` (salary-history-ban world); `no_titles`; `blind` (no pay,
no titles; dates and accomplishments stay); `no_history` (education and summary
only — the floor). On template resumes the views are native; on LLM prose they
are applied by `prose.redact`, and the residual-leak rate per view is reported.

**Prompts (instruction — what the model is told).** One-factor-at-a-time from
`neutral`; full text in `prompts.py`, reproduced in `PROMPTS.md`:

| id | change from `neutral` | hypothesis it serves |
|---|---|---|
| `neutral` | — | raw μ |
| `ignore_comp` | one sentence: decide on skills and experience only; do not consider compensation history or titles | instruction vs redaction |
| `ban_framing` | one sentence: under applicable law you may not consider salary history | does legal framing strengthen compliance |
| `tie_allowed` | answer 0 if there is no job-relevant difference | does forced choice manufacture the preference |
| `rubric` | rate both on three criteria (experience, impact, readiness for level), then choose; JSON | does structure reduce μ, or does "readiness for level" let titles in |
| `reason` | think step by step, then choose; JSON | does deliberation reduce or rationalise μ |

**Posting shown.** Title (ladder title with the pair's specialism), a real
employer name from the ASEE postings, an NY-metro location, the level with its
minimum years, and a one-line focus. No salary in the posting, so any pay
signal comes from the candidate side only.

**Decision-maker.** OpenAI chat completions. Default `gpt-4o-mini` for the full
grid, `gpt-4o` for the primary cells. Temperature 0 (the modal answer; pairs are
the sampling unit). Single-character answers with `logprobs` so the model's
preference is also observed continuously; JSON answers for `rubric` and
`reason`. Every call logged verbatim (`out/prompt_log.jsonl`), cached, metered.

**Resume text.** Static experiment: LLM prose, written once per state by
`gpt-4o-2024-08-06` at temperature 0.75 (the FAccT '25 settings) from the
fixed career, with a layout rule that puts each salary on its own line so
redaction is mechanical. Trajectory: the deterministic template, because
careers change every year. The full grid is also run on template resumes so the
two media can be compared on the primary cells.

## 3. Design controls

- **Both presentation orders for every comparison.** μ is the mean over the two
  orders, so position bias cancels exactly in the estimate; `pick_first_rate`
  reports it on its own, and `consistent_rate` reports how often the two orders
  chose the same *candidate*.
- **Winner label balanced** (A in even pairs, B in odd).
- **Baseline cell** (year 0, no treatment, `neutral`, `full`): the model
  chooses between the twins as they are. Expected μ_A ≈ 0. A significant
  non-zero value means the model does not treat the twins as equals, and every
  treatment estimate must then be read against it.
- **Floor cell** (`neutral`, `no_history`, `both`): expected μ ≈ 0.
- **Content matched**: equal number of new bullets for winner and loser; same
  specialism in both twins' titles; same years of experience rendered.
- **Unrequested ties** (a 0 under a forced-choice prompt) are protocol
  violations, excluded and counted, not scored.

## 4. Outcome and estimator

For pair *i* in cell *c*, over the two orders *o*:
μ̂_ic = ½ Σ_o 1[model picks the prior winner in order *o*] − ½ ,
with a tie (only where offered) scored as ½ and also reported separately as
μ among non-tie pairs. Cell estimate: mean over pairs; SE from the pair-level
sd; 95% CI with the *t* critical value; *p* from a paired-by-design one-sample
*t* test. Contrasts between cells are paired differences over the same pairs.
Secondary, continuous: the order-averaged log-probability the model assigns to
the prior winner (direct-answer prompts only).

## 5. Hypotheses and predictions

**H1 (μ exists).** μ(`neutral`, `full`, `both`) > 0.
*Prediction: yes, in the range 0.10–0.30.* If μ ≈ 0 the rest of the programme
is moot and that is itself the result.

**H2 (where it lives).** Under `neutral`/`full`: μ(`both`) ≥ μ(`salary`),
μ(`title`) > μ(`move`) ≥ 0. *Prediction: the title carries more than the salary
(a level is a stronger "readiness" cue than 9% pay), and `move` alone is small
but positive.* This is the model-side analogue of the market decomposition,
where anchoring carried ~85%.

**H3 (redaction beats instruction) — the primary contrast.**
μ(`neutral`, `no_salary`) < μ(`ignore_comp`, `full`), and
μ(`neutral`, `blind`) < μ(`ignore_comp`, `full`).
*Prediction: redaction removes most of the salary component; the instruction
removes some of it but leaves a larger residual, because the signal is still in
the window.* If instruction works as well as redaction, the "context is the
mechanism" framing is wrong and we will say so. `ban_framing` is a secondary
check on whether framing strength matters.

**H4 (structure and deliberation) — exploratory.** μ(`rubric`) vs μ(`neutral`)
and μ(`reason`) vs μ(`neutral`). *Prediction: rubric ≥ neutral, because
criterion (c) invites the title; reason is unpredictable and its text will be
coded for whether the model notices equivalence.*

**H5 (forced choice) — exploratory.** Under `tie_allowed`, the tie rate on
equivalent candidates, and μ among non-ties. *Prediction: ties are declared in
a minority of pairs and μ among non-ties is at least as large as under
`neutral`.*

**H6 (compounding).** In the twenty-year loop, under the default market:
re-selection rate (share of later contests won by the year-0 winner) and the
twenty-year gap are larger with the model choosing under `full` than under the
coin; `no_salary` reduces both toward the coin. *Prediction: re-selection
0.55–0.70 under `full`; gap above the coin's; `no_salary` roughly halves the
excess.* Under the `no_anchoring` market the same comparison isolates the
model's contribution from the market's.

**Multiplicity.** H1, H3 (two contrasts) and H6 (one contrast) are the
confirmatory tests, Holm-corrected as a family of four. Everything else is
exploratory and will be presented as estimates with intervals, not as
significance.

## 6. Sample size and cost

Pair-level sd of μ̂ is ~0.35 (mock and pilot). With 100 pairs, SE ≈ 0.035 per
cell and ≈ 0.045 for a paired contrast; with 150 pairs, ≈ 0.029 and 0.037.
Projected costs from `--dry-run` on 7 Sept (token counts approximate here,
exact on the run machine):

| step | model | calls | projected |
|---|---|---|---|
| pilot: 7 cells × 20 pairs, template | gpt-4o-mini | 280 | $0.05 |
| full grid: 29 cells × 100 pairs, template | gpt-4o-mini | 5,800 | $0.94 |
| prose resumes: 600 states (120 pairs) | gpt-4o-2024-08-06 | 600 | $4.55 |
| full grid, prose | gpt-4o-mini | 5,800 | ~$1.1 |
| primary: 7 cells × 120 pairs, prose | gpt-4o | 1,680 | ~$4.8 |
| trajectory: 4 LLM cells × 50 pairs (+ coin controls) | gpt-4o-mini | ~2,200 | $0.37 |
| trajectory: 2 default-market cells × 50 pairs | gpt-4o | ~1,100 | $3.09 |
| **total** | | | **≈ $15** |

All plans draw from one fixed pool of pairs, so the pilot's 20 are the first
20 of the grid's 100, which are the first 100 of the primary run's 120, and
one set of prose resumes serves every plan.

Each entry point projects its own cost and refuses to send if the projection
exceeds `--budget`.

## 7. Order of operations

1. `--model mock` runs of every entry point (done; the analysis recovers the
   mock's built-in preference in the salary-visible cells and ~0 elsewhere).
2. Pilot on `gpt-4o-mini`, 20 pairs, template: parse failures, position bias,
   refusal rate, tie rate. Fix mechanics only; no hypothesis is touched.
3. Prose generation; leak report; a hand check of ten resumes.
4. Full grid on `gpt-4o-mini`, template then prose.
5. Primary cells on `gpt-4o`, prose.
6. Trajectory on `gpt-4o-mini`, then the two default-market cells on `gpt-4o`.
7. Report: per-cell table with intervals, the contrasts, the trajectory table
   against the coin, the prompt log, and a list of everything that deviated
   from this document.

## 8. What would falsify the framing

- μ ≈ 0 under `full`: the model does not carry outcomes; the market alone does.
- μ(`ignore_comp`) ≈ μ(`no_salary`) ≈ 0: instruction is as good as redaction
  and the context-window argument is not needed.
- Baseline μ_A far from 0: the twins are not equivalent to the model and the
  instrument needs rebuilding before anything else is claimed.
- Trajectory gap under the model ≤ coin: the model's memory does not compound
  beyond the market's.

## 9. Known limitations, stated in advance

One occupation, one metro, synthetic careers, template prose for the
trajectory, a single model family, temperature 0. The prose views are applied
by regex redaction on free text; leaks are counted, and the template arm exists
so the primary cells can be checked on text with no leaks at all. Names are not
varied here (ε is not this experiment); the identity insertion from the FAccT
appendix is available for the follow-up that multiplies ε by μ.

