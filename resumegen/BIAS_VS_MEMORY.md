# Bias sets the direction. Memory sets the magnitude.

*A design result about cumulative advantage in hiring. No API calls — the core
claim is a symmetry argument the simulation only confirms.*

---

## The argument

Build two candidates who are **exchangeable**: matched on seniority, years of
experience, degree, number of positions, career breaks, promotions, pay
(within 2%), and identical latent ability, differing only in surface —
employers, school, city, wording, name. (Construction and per-pair audits:
`twins.py`, samples in `out/twin_pairs.md`.)

Now let any decision-maker pick one for a job. Because the two are
statistically identical, the career process that unfolds afterwards is the same
process regardless of *who* was picked. The selection rule — a fair coin, a
biased coin, an LLM with any preference whatsoever — chooses **who receives the
winner's path. It cannot change what the winner's path is.**

So the long-run consequences split exactly in two:

- the **magnitude** of divergence between equals is governed entirely by
  *market memory* — the channels through which today's outcome reaches
  tomorrow's opportunity (salary anchoring, the title ladder, wage compounding);
- the **direction** of divergence — which group ends up favoured — is governed
  entirely by the *selection rule*.

That is a logical consequence of exchangeability, not a simulation finding.
The simulation's job is to confirm the model respects it and to put numbers on
the memory term.

## The verification — and what it does and does not show

400 matched pairs, 20 years, crossing the selection rule with market memory
(`bias_vs_memory.py`), reported over 20 seeds (`replicate.py`). "Always picks
A" is a *maximally* biased decision-maker — it discriminates on every decision.

| selection rule | market memory | winner-vs-loser gap | A-vs-B group gap |
|---|---|---|---|
| fair coin | on | **+14.1% ± 1.3** | −0.5% ± 1.1 |
| always picks A | on | **+13.0% ± 1.2** | **+13.0% ± 1.2** |
| fair coin | off | +0.0% ± 0.9 | −0.2% ± 1.2 |
| always picks A | off | −0.1% ± 1.2 | **−0.1% ± 1.2** |

**The objection, which is correct.** There is no consistent difference
between twin A and twin B — that is the whole point of the matching, and the
audit confirms it (same level, experience, multiplier; pay within 0.05% on
average). So *always picks A* is a relabelling of the fair coin: it produces the
same individual gap (column 3 agrees within Monte Carlo noise, ~1 point), and
the group gap in column 4 equals the individual gap **by definition**, because
"group A" was defined as "whoever the rule picks". The table is therefore not
an empirical finding about bias. It is a check that the code respects the
symmetry the twins impose, plus a tautology.

What the table is *for* is the next step, where it stops being a tautology.
Replace "always picks A" with a real decision-maker — an LLM reading the two
resumes, one of which carries a name from group X — and the rule's preference
ε becomes a *measured* quantity between 0 and 1, not a design choice. The
identity then reads: **twenty-year group gap = ε × (the memory term)**. The
memory term is what the fair-coin row measures. ε is what every audit in the
reading list measures. Neither alone predicts long-run group inequality; the
product does — and the corner cell says that at zero memory, no ε does any
lasting damage.

Within the memory term, the channel decomposition (`divergence.py`,
`TWIN_STUDY.md`) says **salary anchoring carries about 85% of the mechanism**:
switch off only the rule that your current pay floors your next offer and the
gap falls from 14% to 2%, and the winner is ahead in 53% ± 2.6 of pairs — not
the "exactly 50.0%" in an earlier version, which was a single unreproducible draw
(see the correction note in `TWIN_STUDY.md`).

**The two arms, defined.** *"Salary history not carried"* switches off
one channel only — anchoring — and leaves the title ladder and wage compounding
on, so the winner's advantage from the day-one title step-up survives for a
while and then decays as the loser is promoted. *"No memory"* switches off all
three channels *and* the day-one title step-up, so a job move redraws a person
from the population band regardless of where they were; the initial pay gap
lasts only until the first move. The first is the salary-history-ban world; the
second is the no-history null.

## Why this reframes the reading list

Every paper in the collection we reviewed — the audits, the correspondence
experiments, the name-swap studies — measures the **direction term**: which way
does the model tilt between candidates, and for whom. Call it ε.

None of them measures the **memory term**: how much of a candidate's *past
outcomes* the model re-transmits into their *next* outcome. Call it μ.

The decomposition says μ, not ε, controls whether inequality between equals
exists and persists at all. ε only addresses *who* it lands on. Debiasing the
decision-maker (ε → 0) equalises groups but leaves the total amount of
arbitrary, luck-driven inequality between equals untouched. Cutting memory
(μ → 0) shrinks that inequality for everyone — and salary-history bans, blind
screening, and title-free evaluation are all μ-interventions that exist in
policy today.

## The proposed experiment

A measurement program, in three stages. Stage 1 is done; stages 2–3 are
designed and costed but not run.

**Stage 1 (done, no API).** The twin design, the symmetry result, and the
simulated memory term above. Establishes the instrument.

**Stage 2 — measure μ for an LLM screener.** Give a model matched candidate
histories that differ *only* in a past outcome: same person, ±15% prior
salary, or with/without a title bump, everything else fixed by the twin
machinery. The movement in the model's score or offer estimates how much of
yesterday's outcome the model re-transmits. Repeat across the visibility
ablations (`full`, `no_salary`, `no_titles`) to decompose μ by channel. A few
hundred calls; the pairs already exist.

**Hypothesis worth registering in advance:** LLM screeners have *higher* μ
than human markets — a model reads everything on the page and anchors on it
with perfect consistency, where humans anchor noisily. If that holds, moving
hiring into LLMs increases cumulative advantage **even at zero bias**, and the
fairness literature is auditing the wrong term.

**Stage 3 — project and intervene.** Put the measured μ into the calibrated
career simulation and re-run the twenty years. Then price the interventions:
how much does a salary-history ban (forcing the anchoring component of μ to
zero) reduce twenty-year inequality between equals, under an LLM screener vs
the human baseline?

## The question

Every audit in the folder asks: *which way does the model tilt?* This design
says the damage is controlled by a different quantity: *how much of a person's
past the model carries into their future.* Is measuring that — the memory
coefficient of AI-mediated hiring — the paper?

---

*Caveats. Exchangeability holds by construction in synthetic twins — that is
precisely why the design is the instrument; real candidates never satisfy it.
The 14% magnitude is a parameter of the simulated market, not an estimate of
the world's; stage 2/3 are what would ground it. And whether an LLM actually
treats our twins as equals is itself an empirical claim the twin design can
test (run both orders, check for systematic preference) — if it fails, that is
measurement bias, and it is separable from cumulative advantage only because
the pairs are matched.*
