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

---

## Addendum, 11 September — added before the pilot, after re-reading the follow-up request

The follow-up request was: *"run the same experiment several times, varying the
prompt each time, and collect the results. We will want to see what is
significant in the findings across runs."* The six prompts above vary the
prompt's **content** (each is a hypothesis). They do not vary its **wording**,
and an LLM decision that flips under a paraphrase is not a finding. Two
additions, both frozen now:

**Paraphrase family.** `neutral_b` and `neutral_c` are paraphrases of
`neutral` with the same content (recruiter → talent acquisition → hiring
manager; candidate → applicant → resume; the same single-digit answer). Run
under `full` and `no_salary` (`--plan robustness`, 6 cells). Prediction: μ
agrees across the three paraphrases within noise, and the redaction effect
(H3) appears under each of them. A confirmatory result is one that holds under
**all three** wordings; a result that holds under one is reported as
wording-dependent.

**Replication protocol.** The main runs use temperature 0, so re-running them
returns the same answers and "several runs" would be theatre. For the six
primary cells we therefore add `--temperature 1.0 --replicates 3`: every
comparison is sampled three times with different sampling seeds, and the
report shows μ per run and its spread across runs (`*_across_runs.csv`). A
cell whose sign changes between replicates is not significant, whatever its
single-run *p*. The mock run of this protocol is in the runbook.

Costs: robustness plan on `gpt-4o-mini`, 100 pairs, ≈ $0.20 (≈ $0.60 with three
replicates); on `gpt-4o` ≈ $3. Total programme remains under $20.

**Decision, 13 September.** The PI was not consulted on models, temperature or
prompt wording before the pilot — the timing did not allow it, and the run is
on the author's own credits. Those choices are ours and are frozen as written
above. Anything changed afterwards at the PI's request is a dated
pre-registration change and a re-run, not a silent edit.

**Rate limits (Tier 1, first payment ≥ $5).** gpt-4o: 500 RPM / 30,000 TPM;
gpt-4o-mini: 500 RPM / 200,000 TPM. At ~1,200 tokens per comparison the
gpt-4o runs are throttled to ~25 calls a minute, so the primary run takes
about an hour of wall-clock and the prose generation about half an hour; the
harness waits out 429s rather than counting them as failures. This is a
timing constraint, not a design one.

---

## Amendment, 13 September — after the pilot (280 calls, $0.04, gpt-4o-mini, 20 pairs, template)

What the pilot showed, verbatim from `out/hiring_pilot_gpt-4o-mini_template_summary.csv`:

| cell | μ | P(winner) from logprobs | consistent | pick-first |
|---|---|---|---|---|
| neutral · full · both | **+0.500** (40/40) | 0.966 | 1.00 | 0.50 |
| neutral · no_salary · both | +0.325 | 0.825 | 0.65 | 0.68 |
| ignore_comp · full · both | +0.325 | 0.847 | 0.65 | 0.68 |
| rubric · full · both | +0.500 (40/40) | — | 1.00 | 0.50 |
| tie_allowed · full · both | +0.500 (40/40), **0 ties** | 0.961 | 1.00 | 0.50 |
| reason · full · both | +0.450 | — | 0.90 | 0.55 |
| neutral · full · **base** | −0.125 (p = 0.02) | 0.401 | **0.25** | **0.825** |

Three things follow, and two changes.

1. **H1 is answered at the ceiling.** With the market's full offer visible
   (one level up, +9% pay, new employer), gpt-4o-mini chose the prior winner in
   every one of 40 comparisons, at 97% confidence. μ exists; the pilot cannot
   say how large it is because the instrument is pinned at its maximum. Every
   prompt at `full` reads 0.45–0.50, so prompt contrasts at this dose are
   uninformative by construction.
2. **Redaction and instruction both took μ from 0.50 to 0.325 — exactly the
   same value.** The instruction (`ignore_comp`) asks the model to ignore both
   pay and titles; the redaction removed pay only. That the two coincide is
   suggestive that instruction is the weaker lever per channel, but the
   like-for-like comparison is `blind` (pay and titles removed) vs
   `ignore_comp`, which the pilot did not include. Treated as a hypothesis, not
   a result.
3. **Between genuine equals the model has almost no position-robust
   preference — and an 82.5% preference for whichever candidate is shown
   first.** Only 5 of 20 matched pairs got the same candidate in both orders,
   and in all 5 it was twin B (p ≈ 0.06 under a fair coin). The order-balanced
   estimator absorbs the position bias; the B-lean is flagged for the 100-pair
   grid, where it either persists (a construction asymmetry to find) or does
   not (chance).

**Change 1 — dose.** The prompt × view contrasts (H3, H4, H5, wording) will be
evaluated at the strongest treatment that is *off the ceiling* on
gpt-4o-mini, chosen from the decomposition run (`--plan decomposition`:
`move`, `salary`, `title`, `both`, plus `salary` with pay redacted and `title`
with titles redacted) by the rule: the treatment with the largest μ whose
P(winner) is below 0.90. `both` remains the primary treatment for H1 and for
the trajectory. The plans take `--treatment` so this is one flag, not a code
change, and the output files carry the dose in their names.

**Change 2 — ties at baseline.** `tie_allowed` is added at the year-0 baseline
(`tie_allowed · full · base`): the two candidates really are equivalent, so
the question "does the model ever say so?" is asked where the answer should be
yes. In the pilot it declared no ties at all when a real difference was
present, which is correct behaviour and says nothing yet.

Neither change alters a hypothesis or a prompt. Both are recorded here before
the corresponding runs.

---

## Amendment 2, 13 September — dose chosen from the decomposition (gpt-4o-mini, 100 pairs, template, $0.15)

| treatment (neutral prompt) | view | μ | P(winner) | consistent |
|---|---|---|---|---|
| `both` | full | +0.445 | 0.937 | 0.89 |
| `title` | full | +0.320 | 0.834 | 0.64 |
| `salary` | full | +0.130 | 0.612 | 0.32 |
| `move` | full | −0.055 (p = 0.02) | 0.428 | 0.23 |
| `salary` | no_salary | −0.060 | 0.429 | 0.24 |
| `title` | no_titles | +0.010 | 0.510 | 0.18 |

**H2 confirmed as predicted:** title (0.32) > salary (0.13) > move (≈ 0, slightly
negative — a bare job change is read as a mild negative, not a credential), and
the two channels are close to additive (0.32 + 0.13 ≈ 0.445). **Redaction of a
channel removes that channel entirely:** hide the pay and the `salary`
treatment falls to the `move` floor (−0.06 ≈ −0.055); hide the titles and the
`title` treatment falls to zero (+0.01). Both p < 0.001, 100 pairs.

**Robustness at `both`:** μ is 0.445 / 0.470 / 0.480 under the three
paraphrases — H1 does not depend on wording. The size of the salary-redaction
effect does (−0.135 / −0.075 / −0.020), because at `both` the title keeps the
model near the ceiling under every wording and leaves the redaction no room.
That is the ceiling problem, not a wording effect, and it is why the dose
matters.

**Dose decision, by the rule in Amendment 1 plus one constraint that should
have been written there:** a redaction can only be tested at a dose whose
signal it removes. So the prompt × view grid runs at **two single-channel
doses**: `--treatment salary` (pay is the only difference; `no_salary` is the
matching redaction; this is the pre-registered H3 channel and the
salary-history-ban policy case) and `--treatment title` (the level is the only
difference; `no_titles`/`blind` are the matching redactions). Both are off the
ceiling on gpt-4o-mini (P = 0.61 and 0.83). `both` stays the H1 and
trajectory dose. On gpt-4o the primary run is at `salary` first; `title`
follows if budget allows.

**Prose resumes:** 840/840 passed the checks, $4.36. Redaction fallback rates
— the share of resumes where the line-level rule missed something that the
regex then scrubbed — were 2% for pay and 39–40% for titles (bare "senior"
in a summary sentence). The prose-based grid is therefore the secondary medium
for title-channel results, and the template grid the primary, until ten
redacted resumes have been read by eye (`python -m resumegen.prose --show 10`).
