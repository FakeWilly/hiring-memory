# Realism audit — the generated resume database

Every number below is reproducible with `python -m resumegen.qa` and was
verified on the shipped dataset (200 careers → 1,508 resumes across two
snapshot years and four identities). Where an external anchor exists, it is
named. Where the data falls short of realism, that is stated rather than
smoothed over — the failures are as load-bearing as the passes.

## 1. Salaries match the market they were drawn from

Cross-section of current salaries vs the 1,855 real ASEE software postings the
bands were estimated from:

| percentile | generated | real postings |
|---|---|---|
| p10 | $82,300 | $85,366 |
| median | $123,400 | $129,000 |
| p90 | $186,100 | $196,657 |

Within 5% at every quantile. Dispersion is comparable too: sd of log salary
0.287 vs 0.33 in the postings. This is partly by construction — level bands are
estimated from the postings, then back-dated at 2.2%/yr so a 2008 role is not
paid 2024 money — but the construction is the point: the cross-section cannot
drift away from the observed market.

## 2. The seniority pyramid narrows the way real ones do

| level | share |
|---|---|
| Junior | 11.9% |
| Engineer | 40.6% |
| Senior | 36.1% |
| Staff | 10.9% |
| Principal | 0.5% |

Staff-and-above at ~11% is in line with industry norms (roughly 10–15% at most
established engineering organisations). An earlier version put half the cohort
at Staff+; the promotion model now carries a per-level penalty — the pyramid
narrows because promotion gets harder the higher you go.

## 3. Careers move like careers

Mean 4.3 positions per career (max 8), tenures of 2–5 years, median 10 years of
experience, 18% of careers containing a break. Job-to-job moves average +18.8
log points with sd 0.122 — meaningful raises with realistic scatter, not the
near-random redraws that produced the original pipeline's trajectories.

**Pay cuts are the honest weak spot: 4.4% of moves, all following a career
break.** The original pipeline produced 40.1% (a bug, verified against its own
saved output); real wage panels sit nearer 10–15%. So this database is now
*too smooth* — over-corrected — and the rate should be calibrated against PSID
income paths rather than set by judgement, which is a one-parameter change
(`lateral_cut_prob`).

## 4. Merit exists but does not dominate

corr(latent ability, salary) = +0.43, (ability, level) = +0.25, (ability,
promotions) = +0.12. Ability enters in exactly two named places — promotion
odds and a persistent pay multiplier — and everything else is tenure and luck.
These are **set parameters, not findings**; what the audit establishes is that
they are visible, modest, and adjustable, so "how much of the gap is merit?"
remains a question the data can answer rather than one it forecloses.

## 5. Surface details survive inspection

Real employer names from the postings, with recruiting cruft filtered (an
earlier pass produced "Dennis Group for New Grads, Co-Ops & Internships" as an
employer and a 16-year veteran whose current role was an internship — both
classes of failure are now caught by the QA). Titles come from a canonical
ladder with a specialism, because raw posting titles are adverts, not history.
Locations are NY-metro, matching the prompt's premise. Resume length runs
536–2,454 characters and grows with seniority.

## 6. The pairs are equivalent, and the audit says so per pair

Of 400 matched pairs, **400/400 match on every merit dimension** (level,
experience, degree, positions, breaks, promotions, ability, pay within 2%).
Three pairs failed the *surface-difference* requirement — the twins came out
too similar (same school drawn by chance) — and are flagged, not hidden. Each
pair in `twin_pairs.md` carries its own audit table, so "these two are equal"
is checkable line by line rather than asserted.

## 7. One realistic property that is also a threat

**corr(resume length, salary) = +0.77.** Real resumes do grow with careers, so
this is realism — and it is also a warning, because LLM judges favour longer
inputs and here length is *caused by* success. It is a mediator, not a
confounder: controlling for it would delete part of the effect under study,
ignoring it risks measuring length rather than substance. The database makes
the problem measurable; it does not make it go away.

## Known limitations, plainly

- Prose is templated; the GPT-4o backend (`--backend openai`, the FAccT
  appendix prompt extended with career and salary) is wired but has never been
  run — no API key. The three prose examples in `prose_examples.md` were
  written by Claude to the same prompt and are labelled as such.
- One occupation ladder (software engineering), one metro (New York), following
  the FAccT appendix.
- No firm effects, layoff waves, or business cycles — the market is stationary.
- Names follow the audit-literature conventions (Bertrand & Mullainathan and
  successors); the FAccT paper generates its own list and does not publish it.
