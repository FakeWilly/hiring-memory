# resumegen — synthetic resumes with salaries and careers

Extends the resume generator from **Zollo, Rajaneesh, Zemel, Gillis & Black,
*Towards Effective Discrimination Testing for Generative AI*, FAccT '25**
(Appendix B.1) with the two things a cumulative-advantage study needs and that
paper does not produce: **a salary attached to every role**, and **a career that
moves through time**.

Runs with **no API key**. `--backend openai` swaps in the paper's GPT-4o
generation prompt when one is available.

```bash
python -m resumegen.generate --n 200 --source psid --identities --snapshots 2013 2021
python -m resumegen.qa
```

## What the paper gives us, and what it doesn't

| | FAccT '25 Appendix B.1 | here |
|---|---|---|
| Personas | Table 3, 19 traits sampled independently | same table, **plus** a PSID-grounded option |
| Resume text | GPT-4o, Table 2 prompt | same prompt, extended; plus an offline renderer |
| `[NAME]` / `[EMAIL]` placeholders | yes | yes — kept, it's the best idea in the appendix |
| **Salary** | none (income is a persona trait) | **every role, banded from real postings** |
| **Time** | one resume, one moment | **a career; render it at any year** |
| Occupation | Social Worker | software engineering ladder |
| Ablations | — | `full`, `no_salary`, `no_titles`, `terse`, `no_history` |

## Why two persona sources

`--source table3` reproduces the paper: 19 traits, sampled independently.

`--source psid` draws the same trait vocabulary from real PSID people, so the
**joint** distribution survives — education, occupation and income co-vary as
they do in life.

That distinction matters more here than it did in the paper. For a bias audit,
independent sampling is close to harmless, because every counterfactual pair
shares one body and only the name changes. For a cumulative-advantage study the
correlation structure **is** the object of interest, so sampling it away removes
what we came to measure.

Both are one flag apart, so the choice is testable rather than assumed.

The PSID path also caps ages at 25–44, following the paper. That happens to fix
a live problem in the main pipeline: the full PSID sample averages 45 in 2021,
so a run to 2049 loses most of the cohort to retirement and the final-year
estimates land on ~15 people.

## The salary model

1. Each seniority level gets its **empirical band** — the median of real
   postings inferred to that level.
2. Each person carries a **persistent multiplier** (~0.80–1.30). Some people sit
   above band their whole career. This is the thing a Matthew effect would act on.
3. Salaries are anchored to the band for the person's **current** level, then
   **back-dated** at 2.2%/yr. The postings are present-day adverts, so a 2008
   role has to be discounted or the cross-section comes out far richer than the
   market it was drawn from.
4. Staying put earns ~2.8%/yr. Moving earns a premium over current pay, capped
   at +35% for one move.
5. **Only a career break can cut pay**, plus a 10% chance of a lateral move
   taken for something other than money.

Rule 5 is deliberate. The current pipeline produces **40.1%** pay cuts on
job-to-job moves; this produces **4.4%**. Real wage panels sit nearer 10–15%,
so if anything this is now slightly too smooth — worth tuning against PSID
rather than by eye.

## Verify it

`python -m resumegen.qa` checks the things a reviewer will:

```
generated  p10 $ 82,300  median $123,400  p90 $186,100
postings   p10 $ 85,366  median $129,000  p90 $196,657

L0 Junior      11.9%    pay cuts 4.4%, all after a break
L1 Engineer    40.6%    corr(ability, salary) +0.43
L2 Senior      36.1%    corr(ability, level)  +0.25
L3 Staff       10.9%    corr(resume length, salary) +0.77
L4 Principal    0.5%    resume length median 1,113 chars
```

The ability correlations are deliberately modest. Ability enters in exactly two
places — promotion odds and the pay multiplier — and everything else is tenure
and luck, so it stays possible to ask later how much of the observed advantage
is merit.

That last number is the one to keep in view: **resume length correlates +0.77
with salary** across the shipped set, because resumes get longer as careers
progress. (Restricted to the 2021 snapshot alone it is higher still, ~+0.82 —
the pooled figure is diluted by the shorter 2013 careers.) LLM judges favour
longer inputs. So length is a *mediator*, not a confounder — controlling for it
would delete part of the effect under study.

## The twin study

`TWIN_STUDY.md` is the piece that answers the research question directly:
matched pairs, one arbitrary hiring decision, twenty years of divergence, and a
channel decomposition. Headline, over 20 seeds × 400 pairs: an advantage worth
12.1% on the day is worth **14.1% ± 1.3** twenty years later, the winner is
ahead in **70% ± 2.5** of pairs, and **salary anchoring carries ~85% of the
mechanism** — remove it and the gap falls to 2% and the winner is ahead in
53% ± 2.6.

`BIAS_VS_MEMORY.md` separates the selection rule from the market's memory, and
records the (correct) objection that a rule which always picks twin A is a
relabelling of the coin flip until "A" maps onto something a real
decision-maker can see. `slope.py` and `replicate.py` answer his other two
questions: whether higher-paid trajectories are steeper (not in this model, and
not in the PSID file either), and how much the headline moves across seeds and
parameters. `PROMPTS.md` records every prompt and template that touched the
outputs.

```bash
python -m resumegen.show_pairs --pairs 6
python -m resumegen.divergence --pairs 400 --years 20 --channels
python -m resumegen.plot_divergence
python -m resumegen.replicate --seeds 20 --pairs 400 --years 20
python -m resumegen.slope --pairs 400 --years 20
```

## The LLM experiments (designed, mock-tested, not yet run)

`PREREG.md` fixes the design before any paid call: matched twins, one of whom
won a job a year ago; an LLM chooses between them for the next job; the
**memory coefficient** μ = P(picks the prior winner) − ½. Two dimensions are
kept apart on purpose — what the model **sees** (views: `full`, `no_salary`,
`no_titles`, `blind`, `no_history`) and what it is **told** (six prompts, each
one change from `neutral`). The primary contrast is *redaction vs
instruction*: does removing salary from the context beat telling the model to
ignore it? `trajectory.py` then puts the model in the loop for twenty years
against a coin control run through identical code.

| file | what |
|---|---|
| `PREREG.md` | hypotheses, prompts, controls, estimator, sample size, cost, falsifiers |
| `RUNBOOK.md` | key setup (environment variable, never in chat or repo), mock → dry-run → pilot → full |
| `prompts.py` | every prompt, versioned, with the hypothesis each serves |
| `llm.py` | the only file that calls the API: logging, caching, cost meter, budget guard, mock |
| `hiring.py` | the static experiment: treatments, views, both orders, per-cell μ and contrasts |
| `prose.py` | LLM-written resumes at the paper's settings; redaction with leak counts |
| `trajectory.py` | twenty years with the model choosing at each contested posting |

```bash
python -m resumegen.hiring --model mock --plan pilot              # offline proof, $0
python -m resumegen.hiring --model gpt-4o-mini --plan pilot --dry-run
python -m resumegen.hiring --model gpt-4o-mini --plan pilot --budget 1
```

## Outputs

| file | contents |
|---|---|
| `out/personas.csv` | one row per person, all traits, latent ability |
| `out/careers.jsonl` | full career objects — every position, every salary |
| `out/resumes.csv` | one row per (person, year, identity, view) + resume text |
| `out/resumes.md` | readable sample |
| `out/twin_pairs.md` | matched pairs with their equivalence audits |
| `out/divergence_summary.csv` | the channel decomposition |
| `out/divergence_paths.csv` | every pair's 20-year path (seed 0) |
| `out/divergence.png` | the figure (seed 0) |
| `out/replicate_seeds.csv`, `replicate_params.csv`, `replicate_report.txt` | the 20-seed and parameter-perturbation distributions |
| `out/slope_elasticity.csv` | what happens to the gap when pay itself buys growth |
| `out/bias_vs_memory.csv` | the 2×2, seed 0 |
| `out/prose_examples.md` | what the LLM backend produces — **hand-written by Claude to the same prompt, not GPT-4o output** |

## Open questions

1. **Where should salaries come from?** Currently banded from the ASEE
   postings. Alternatives: PSID income paths, or LLM-assigned. This is the
   central unresolved choice.
2. **PSID personas or Table 3 sampling?** The joint-distribution argument above.
3. **How much should ability explain?** +0.43 with salary is a parameter, not a
   finding. It should be calibrated against something real.
4. **Pay-cut rate.** 4.4% here, 40% in the current pipeline, ~10–15% in wage
   panels. Worth setting against PSID.

## Not done

- Never run through the real LLM backend — no API key available. The offline
  renderer is deterministic and fully reproducible, but nothing here has met
  GPT-4o yet.
- Single occupation ladder (software engineering), single metro (NY), as in the
  paper.
- Names are drawn from conventions in the audit literature; the paper generates
  its own list with GPT-4o and does not publish it.
