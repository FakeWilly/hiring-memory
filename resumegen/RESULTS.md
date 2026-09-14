# Results log — LLM memory-coefficient experiments

*Numbers are copied from the `out/*_summary.csv` and `*_contrasts.csv` files
named in each entry; interpretation is kept in separate paragraphs so the two
cannot be confused. Model: gpt-4o-mini unless stated. Temperature 0. Template
resumes unless stated. μ = P(model picks the prior winner) − ½, both
presentation orders averaged, pairs as the unit. Pre-registration and its
dated amendments: `PREREG.md`.*

## 13 Sept — pilot (20 pairs, `both` dose) → `hiring_pilot_gpt-4o-mini_template_*`

| cell | μ | P(win) | consistent | pick-first |
|---|---|---|---|---|
| neutral · full · both | +0.500 (40/40) | 0.966 | 1.00 | 0.50 |
| neutral · no_salary · both | +0.325 | 0.825 | 0.65 | 0.68 |
| ignore_comp · full · both | +0.325 | 0.847 | 0.65 | 0.68 |
| rubric · full · both | +0.500 | — | 1.00 | 0.50 |
| tie_allowed · full · both | +0.500, 0 ties | 0.961 | 1.00 | 0.50 |
| reason · full · both | +0.450 | — | 0.90 | 0.55 |
| neutral · full · base | −0.125 | 0.401 | 0.25 | 0.825 |

Reading: μ exists and the `both` dose is at the ceiling (Amendment 1).

## 13 Sept — decomposition (100 pairs) → `hiring_decomposition_gpt-4o-mini_template_*`

| treatment | view | μ | se | P(win) | consistent |
|---|---|---|---|---|---|
| both | full | +0.445 | 0.016 | 0.937 | 0.89 |
| title | full | +0.320 | 0.024 | 0.834 | 0.64 |
| salary | full | +0.130 | 0.025 | 0.612 | 0.32 |
| move | full | −0.055 | 0.023 | 0.428 | 0.23 |
| salary | no_salary | −0.060 | 0.024 | 0.429 | 0.24 |
| title | no_titles | +0.010 | 0.021 | 0.510 | 0.18 |

Reading (H2, as predicted): title > salary > move; the two channels are
nearly additive (0.32 + 0.13 ≈ 0.445); redacting a channel removes it
entirely, back to the `move` floor. A bare job change with nothing to show for
it is read as a mild negative, not a credential.

## 13 Sept — wording robustness (100 pairs, `both` dose) → `hiring_robustness_gpt-4o-mini_template_*`

| prompt | full | no_salary | redaction effect |
|---|---|---|---|
| neutral | +0.445 | +0.310 | −0.135 (p < .001) |
| neutral_b | +0.470 | +0.395 | −0.075 (p < .001) |
| neutral_c | +0.480 | +0.460 | −0.020 (p = .10) |

Reading: H1 does not depend on wording. The size of the salary-redaction
effect at `both` does, because the title keeps every wording near the ceiling
(Amendment 2). To be re-read at the `salary` dose when the paraphrase cells
finish.

## 13 Sept — salary-dose grid, PARTIAL: 15 of 34 cells (100 pairs) → `hiring_full_gpt-4o-mini_template_salary_partial_*`

Daily request cap reached (Tier 1: 10,000 gpt-4o-mini requests/day); the
remaining 19 cells run tomorrow from cache. The cells below are complete
(200 trials each; `ban_framing · no_salary` is 38 pairs and excluded from
reading).

| cell | μ | se | p | P(win) | consistent | pick-first |
|---|---|---|---|---|---|---|
| neutral · full · **base** | −0.005 | 0.031 | 0.87 | 0.498 | 0.39 | 0.765 |
| tie_allowed · full · **base** | +0.018 | 0.028 | 0.53 | 0.505 | 0.32 | 0.801 (ties 4.5%) |
| neutral · full · salary | **+0.130** | 0.025 | <.001 | 0.612 | 0.32 | 0.82 |
| neutral · no_salary · salary | **−0.060** | 0.024 | .014 | 0.429 | 0.24 | 0.87 |
| neutral · no_titles · salary | +0.070 | 0.020 | .001 | 0.573 | 0.18 | 0.89 |
| neutral · blind · salary | −0.080 | 0.022 | <.001 | 0.427 | 0.22 | 0.89 |
| ignore_comp · full · salary | **+0.020** | 0.023 | .40 | 0.530 | 0.22 | 0.85 |
| ignore_comp · no_salary · salary | −0.050 | 0.023 | .03 | 0.444 | 0.22 | 0.88 |
| ignore_comp · no_titles · salary | −0.015 | 0.024 | .53 | 0.493 | 0.23 | 0.875 |
| ignore_comp · blind · salary | −0.075 | 0.022 | .001 | 0.429 | 0.21 | 0.885 |
| ban_framing · full · salary | **+0.125** | 0.025 | <.001 | 0.644 | 0.31 | 0.845 |
| neutral · full · move | −0.055 | 0.023 | .02 | 0.428 | 0.23 | 0.885 |

Pre-registered contrasts (paired over the same 100 pairs):

| contrast | μ_x | μ_y | diff | se | p |
|---|---|---|---|---|---|
| H3 redact salary vs full | −0.060 | +0.130 | **−0.190** | 0.028 | <.0001 |
| H3 instruct (ignore_comp) vs full | +0.020 | +0.130 | **−0.110** | 0.024 | <.0001 |
| H3 ban framing vs full | +0.125 | +0.130 | −0.005 | 0.017 | .76 |
| **H3 REDACTION vs INSTRUCTION** | −0.060 | +0.020 | **−0.080** | 0.020 | **.0001** |
| H3 BLIND vs INSTRUCTION | −0.080 | +0.020 | −0.100 | 0.024 | <.0001 |
| H3 redact titles vs full | +0.070 | +0.130 | −0.060 | 0.022 | .007 |

Reading, in order of confidence:

1. **The twins are equals to the model.** Baseline μ_A = −0.005 (p = .87) on
   100 pairs; the pilot's B-lean was chance. Every treatment effect is read
   against a clean zero.
2. **A 9% pay difference alone moves the model from a coin to 63/37**
   (μ = +0.130) for the higher-paid of two otherwise-matched people.
3. **Redaction removes the effect completely.** Hide the pay and μ falls to
   −0.060 — the `move` floor, i.e. what is left is the mild penalty for having
   changed jobs at all. Blind (pay and titles hidden) is the same, −0.080.
4. **Instruction removes most but not all of it, and less than redaction.**
   "Decide on skills and experience only; do not take compensation history or
   titles into account" takes μ from +0.130 to +0.020 — not distinguishable
   from zero on its own (p = .40), but 0.080 above what redaction achieves
   (p = .0001), and 0.075 above the `move` floor. Measured against the floor,
   instruction leaves about 40% of the pay effect in place; redaction leaves
   none. This is the pre-registered primary contrast, and it comes out in the
   predicted direction.
5. **Legal framing does nothing.** "Under applicable law you may not consider
   an applicant's salary history" leaves μ at +0.125, indistinguishable from
   no instruction (p = .76). A prohibition without a positive instruction on
   what to weigh instead is not followed; a redirect ("skills and experience
   only") is, partly. Worth its own paragraph in any write-up.
6. **Given an explicit "no job-relevant difference" option, the model says so
   4.5% of the time for candidates who are, by construction, equivalent** —
   and picks whichever is shown first 80% of the time instead. Forced choice
   plus position bias manufactures a decision between equals almost every
   time.
7. Exploratory: hiding titles at the salary dose (where titles do not differ)
   still lowers μ (+0.070, p = .007) and raises pick-first to 0.89 — less
   information on the page pushes the model toward its position default.
   Redaction of anything trades signal for position bias.

Still to run: tie_allowed, rubric, reason and the paraphrases at the salary
dose; the `title` grid; gpt-4o on prose (in progress); the trajectory.

## 13 Sept — gpt-4o, GPT-written resumes, salary dose, 120 pairs → `hiring_primary_gpt-4o_prose_salary_*`

1,920 calls, $4.45, temperature 0. Resumes are the `gpt-4o-2024-08-06` prose
(views applied by `prose.redact`; 210 renderings needed the fallback scrub for
a pay or title token the line rule missed — those tokens were removed, so the
figure is a measure of how untidy the prose was, not of what the judge saw).

| cell | μ | se | p | P(win) | consistent | pick-first |
|---|---|---|---|---|---|---|
| neutral · full · **base** | −0.033 | 0.022 | .13 | 0.478 | 0.23 | 0.850 |
| tie_allowed · full · **base** | −0.008 | 0.022 | .70 | 0.485 | 0.24 | 0.872 (ties 9.2%) |
| neutral · full · move | +0.033 | 0.022 | .13 | 0.537 | 0.23 | 0.850 |
| neutral · full · salary | **+0.146** | 0.022 | <.001 | 0.644 | 0.33 | 0.812 |
| neutral · no_salary · salary | **+0.058** | 0.021 | .006 | 0.564 | 0.22 | 0.875 |
| neutral · blind · salary | +0.042 | 0.024 | .09 | 0.554 | 0.28 | 0.858 |
| ignore_comp · full · salary | **+0.138** | 0.027 | <.001 | 0.638 | 0.43 | 0.754 |
| rubric · full · salary | **+0.275** | 0.024 | <.001 | — | 0.58 | 0.642 |

| contrast (paired, 120 pairs) | diff | se | p |
|---|---|---|---|
| H3 redact salary vs full | −0.088 | 0.017 | <.0001 |
| H3 blind vs full | −0.104 | 0.023 | <.0001 |
| H3 instruct vs full | **−0.008** | 0.020 | **.69** |
| **H3 REDACTION vs INSTRUCTION** | **−0.079** | 0.023 | **.001** |
| H3 BLIND vs INSTRUCTION | −0.096 | 0.023 | <.0001 |
| H4 rubric vs neutral | **+0.129** | 0.023 | <.0001 |

### The two models side by side (salary dose)

| | gpt-4o-mini, template, 100 pairs | gpt-4o, prose, 120 pairs |
|---|---|---|
| baseline μ_A (should be 0) | −0.005 | −0.033 |
| ties declared for identical candidates | 4.5% | 9.2% |
| `move` floor | −0.055 | +0.033 |
| pay shown (`full`) | **+0.130** | **+0.146** |
| pay redacted (`no_salary`) | −0.060 (= floor) | +0.058 (≈ floor, +0.025 ± 0.03) |
| pay and titles redacted (`blind`) | −0.080 (= floor) | +0.042 (≈ floor) |
| instruction to ignore pay & titles (`ignore_comp`) | +0.020 (removes ~60% above floor) | **+0.138 (removes nothing)** |
| legal framing (`ban_framing`) | +0.125 (removes nothing) | not run |
| rubric with "readiness for level" | not yet run | **+0.275 (doubles it)** |

Reading:

1. **The pay effect replicates across model and medium.** A 9% pay
   difference between matched twins moves the pick from 50/50 to 63/37 on
   gpt-4o-mini reading template resumes and to 64/36 on gpt-4o reading
   GPT-written prose. Both baselines are clean zeros.
2. **Redaction removes it to the floor on both.** On gpt-4o the floor is
   slightly positive (a job move is read as mildly good rather than mildly
   suspect), and the redacted cells sit on it within noise.
3. **Instruction is model-dependent and never as good as redaction.** On
   gpt-4o-mini, "decide on skills and experience only" removed most of the
   effect; on gpt-4o it removed none of it — μ 0.146 → 0.138, p = .69 — while
   redaction removed all of it (redaction − instruction = −0.079, p = .001).
   The pre-registered primary contrast holds on both models, and on the
   stronger model it is not close. The prediction in `PREREG.md` §5 was that
   instruction would leave "a larger residual because the signal is still in
   the window"; on gpt-4o the residual is the whole effect.
4. **A scoring rubric makes it worse.** Asking gpt-4o to rate "relevant
   experience, evidence of impact, readiness for the level" before choosing
   raises μ from 0.146 to 0.275 — the pay difference is read as readiness. The
   rubric also halves position bias (pick-first 0.64) and raises consistency
   to 0.58: it makes the model more decisive, and it decides on pay. This is
   the standard fairness recommendation for LLM screening, and on this
   channel it doubles the memory coefficient (H4, predicted direction).
5. **Neither model will say "equal".** Offered an explicit no-difference
   option for candidates that are equivalent by construction, gpt-4o-mini uses
   it 4.5% of the time and gpt-4o 9.2%; both default to whichever resume is
   listed first about 85% of the time.

Caveats specific to this run: prose redaction is regex-based, so the
`no_salary`/`blind` cells on prose are, if anything, upper bounds on residual
μ (a leaked token can only help the winner); the template run is the clean
medium for the redaction cells and agrees. gpt-4o was run at one dose and
without the paraphrase or replicate protocol (budget); those are mini-only
until Tier 2.

Spend to date: ≈ $9.35 of $20.
