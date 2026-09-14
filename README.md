# hiring-memory

**How much of a candidate's past does an LLM screener carry into its next
decision — and does that make arbitrary early luck compound?**

Two people with identical qualifications apply for the same job. One gets it.
A year later they apply for the next job, and the decision-maker is a language
model reading both resumes. This repository measures how much the model's
choice is driven by the earlier outcome itself — the higher title, the higher
pay, the fact of having been chosen — rather than by anything either
candidate can do. We call that quantity the model's **memory coefficient**,
and we test whether removing the signal from the resume (redaction) does more
than telling the model to ignore it (instruction).

Everything is in [`resumegen/`](resumegen/):

| start here | what it is |
|---|---|
| [`resumegen/README.md`](resumegen/README.md) | the resume/career generator, the twin design, the coin-flip simulation |
| [`resumegen/PREREG.md`](resumegen/PREREG.md) | the pre-registration for the LLM experiments, with dated amendments |
| [`resumegen/RESULTS.md`](resumegen/RESULTS.md) | the results log — numbers and reading kept separate |
| [`resumegen/PROMPTS.md`](resumegen/PROMPTS.md) | every prompt that touches a result |
| [`resumegen/RUNBOOK.md`](resumegen/RUNBOOK.md) | how to run it, from API key to tables |

Headline so far (gpt-4o-mini and gpt-4o, 100–120 matched pairs each): a 9%
pay difference between otherwise-identical candidates moves an LLM screener
from a coin to about 64/36 for the higher-paid one. Redacting the pay removes
the effect on both models. Instructing the model to ignore pay removes most of
it on gpt-4o-mini and none of it on gpt-4o. A scoring rubric doubles it.

## Setup

```bash
pip install -r requirements.txt
```

Two data files are required and are **not** included — see
[`data/README.md`](data/README.md). Everything else runs offline with
`--model mock`; the LLM experiments need an `OPENAI_API_KEY` in the
environment (never in a file).

```bash
python -m resumegen.hiring --model mock --plan pilot          # no key, no cost
python -m resumegen.hiring --model gpt-4o-mini --plan pilot --dry-run
```

## Provenance

The resume generator extends the persona and prompt design of Zollo,
Rajaneesh, Zemel, Gillis & Black, *Towards Effective Discrimination Testing
for Generative AI* (FAccT '25), Appendix B.1. The cumulative-advantage framing
and the matched-pair design were developed as part of a project on the Matthew
effect in AI-mediated hiring; the coin-flip market simulation, the twin
audits, the LLM harness and the pre-registered experiments here are the
contribution of this repository. Every model call is logged verbatim
(`resumegen/out/prompt_log.jsonl`, shared on request; git-ignored for size).
