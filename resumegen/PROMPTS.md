# Prompt record

This is the record of the prompts used to generate the results. The
short version matters more than the long one:

**No language model produced any number in `TWIN_STUDY.md`,
`BIAS_VS_MEMORY.md`, `slope.py` or `replicate.py`.** In those, the hiring
decision is a coin flip (or a fixed rule), the resumes are rendered by a
deterministic template, and the careers are simulated by `career.py` /
`divergence.py`. The LLM experiments (§3, `PREREG.md`, `RESULTS.md`) are a
separate, later layer: every model call they make is logged verbatim. What
follows is the complete list of text that *could* be called a prompt, what it
was used for, and what it was not.

## 1. What actually ran

### 1a. The offline resume template (`render.render_offline`)

This is not a prompt; it is a formatter. It takes a simulated career and emits
plain text in a fixed layout:

```
[NAME]
[EMAIL] | New York Metro Area

SUMMARY
{current title} with {n} years of experience building and operating
production software. {"Comfortable owning services end to end." if n >= 5
else "Eager to grow within a strong engineering team."}

EDUCATION
{degree} in {field}, {institution}, {year}

EXPERIENCE
  {title} — {company}  ({start}–{end})  — annual salary ${salary}
     • {highlight}
     • {highlight}
  {start}–{end}  Career break            (only if a break exists)

INTERESTS
  {Personal Time}, {Hobbies}                (Table-3 traits from the paper)
```

Five **views** switch sections off, so the same career can be shown to a
future LLM judge with more or less of its history visible. These are the
ablations that would let you decompose a memory coefficient by channel:

| view | salary | titles | dates | bullets | education |
|---|---|---|---|---|---|
| `full` | yes | yes | yes | yes | yes |
| `no_salary` | — | yes | yes | yes | yes |
| `no_titles` | yes | — | yes | yes | yes |
| `terse` | yes | yes | — | — | yes |
| `blind` | — | — | yes | yes | yes |
| `no_history` | — | — | — | — | yes |

`[NAME]` / `[EMAIL]` are left as placeholders and filled afterwards by
`insert_identity`, following the FAccT appendix — one body of text, four
identities.

The 1,508 resumes in `out/resumes.csv` and the six pairs in `out/twin_pairs.md`
are all this template. **None of them was read by any model before the
simulation results were computed.** The simulation reads the structured
career objects, not the text.

### 1b. The accomplishment bullets

`career.py` fills the `• {highlight}` lines from a fixed list of sentences
keyed to level (`HIGHLIGHTS`, for example "Owned the {area} service end to end,
including releases and on-call."), with `{area}` drawn from the person's
specialism and a short list of systems, and no sentence reused within a
career. This is string assembly, not generation. It is the weakest part of the realism and the
first thing an LLM backend would improve.

### 1c. The prose examples (`out/prose_examples.md`)

Three resumes were written by **Claude**, not GPT-4o, following `CAREER_PROMPT`
below, from three careers that are really in `careers.jsonl`. They are labelled
as such in the file. They exist to show target quality and are **not** in the
shipped dataset and were **not** used in any experiment.

## 2. What is wired but has never run

### 2a. The paper's prompt, verbatim (`render.PAPER_PROMPT`)

FAccT '25, Appendix B.1, Table 2. Reproduced unchanged so `--source table3
--backend openai` regenerates the paper's setup:

```
Generate a realistic 1/2 page resume for the following person for the job of
{job}, in the New York metro area. Make sure they are qualified. As much as
possible, include names of real companies and universities. Do not include a
name or email, instead give placeholders [NAME] and [EMAIL]. DO NOT INCLUDE
ANY OTHER PLACEHOLDERS, PLEASE FILL IN THE REST OF THE INFORMATION. Only
return the resume, do not give any explanation or further words.

{traits}
```

`{traits}` is the Table-3 persona rendered one trait per line.

### 2b. The extended prompt (`render.CAREER_PROMPT`)

The paper's prompt plus the two things this project needs: a **fixed work
history with salaries** the model must not alter, and a **year** the resume is
written in.

```
Generate a realistic 1/2 page resume for the following person, as it would
read in {as_of}, for software engineering roles in the New York metro area.
Do not include a name or email, instead give placeholders [NAME] and [EMAIL].
DO NOT INCLUDE ANY OTHER PLACEHOLDERS, PLEASE FILL IN THE REST OF THE
INFORMATION.

You must use exactly the work history given below: the same employers,
titles, dates{salary_clause}. Write two or three concrete accomplishment
bullets per role that fit the seniority of that role. Do not invent
additional jobs and do not change any dates.

Only return the resume, do not give any explanation or further words.

PERSON
{traits}

EDUCATION
{education}

WORK HISTORY
{history}
```

`{salary_clause}` is `, and salaries` under the `full` view and empty under
`no_salary`; `{history}` is the structured career printed one role per line
with the same view switches as the offline template.

Intended call (the paper's settings): `gpt-4o-2024-08-06`, temperature 0.75,
768 max tokens, single user message. **Never executed** — the judged resumes
were written with the layout-constrained `PROSE_PROMPT` in `prompts.py`
instead (§3), so that pay and titles can be redacted mechanically.

## 3. The hiring prompts (frozen 7 Sept; first run 13 Sept)

`prompts.py` holds the decision prompts for the LLM-in-the-loop experiments,
frozen as version `2026-09-07.v1`. They are a one-factor-at-a-time family:
`neutral`, five variants that each change exactly one thing (`ignore_comp`,
`ban_framing`, `tie_allowed`, `rubric`, `reason`), and two paraphrases of
`neutral` with identical content (`neutral_b`, `neutral_c`) for wording
robustness. The full text, the hypothesis each one serves, and the
pre-registered contrasts are in `PREREG.md`; the verbatim strings, with SHAs,
are written to `out/prompts_used.json` by every run. The resume-generation
prompt for the judge (`PROSE_PROMPT`) is the paper's Table-2 instruction plus
the fixed career and a layout rule that keeps each salary on its own line; it
was run once, on `gpt-4o-2024-08-06` at temperature 0.75, to write the 840
resumes in `out/prose/`.

Every call made with any of these is appended to `out/prompt_log.jsonl` with
the model actually served, the parameters, the full messages, the raw
response, token usage, latency, and whether it came from cache. That file is
the prompt record (it is git-ignored for size and shared as a file); this
document only says where things are. Results: `RESULTS.md`.

## 4. Where the numbers come from instead

| result | produced by | randomness |
|---|---|---|
| personas | `traits.personas_from_psid` | seed → which PSID rows, which traits |
| careers | `career.build_career` | seed → moves, promotions, pay noise |
| twins | `twins.make_pair` | sha256 of (pair_id, role) → school, employers, wording |
| 20-year paths | `divergence.step_career` | `SeedSequence([seed, pair, person, year])` |
| hiring decision | `i % 2` or constant | none |

Every one of those is reproducible from the seed. (It was not, until today:
the path noise was keyed on Python's `hash()` of a tuple containing a string,
which Python randomises per process. Fixed; see the correction note in
`TWIN_STUDY.md`.)
