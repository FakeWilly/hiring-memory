# Runbook — from key to results

Everything below runs from the repo root in the `stage4` environment. One
line at a time. Nothing sends a paid call until step 4.

## 0. One-time setup

```bash
cd ~/synthetic-resumes && conda activate stage4
```
```bash
pip install openai tiktoken
```

## 1. The key — never in the chat, never in a file in the repo

Get an API key from the OpenAI dashboard (the project's key if there is one,
otherwise your own). Then, in the terminal you will run from:

```bash
export OPENAI_API_KEY=sk-...paste-here...
```

That sets it for this terminal window only. Close the window and it is gone;
open a new one and you set it again. The code reads it from the environment
and never prints, logs or writes it. Check it is set without revealing it:

```bash
echo ${OPENAI_API_KEY:0:7}...   # prints "sk-proj..." or similar
```

If you would rather not retype it, put the `export` line in `~/.zshrc` on
your own machine only — not on a shared one, and never commit that file.

## 2. Prove the pipeline with no key and no cost

```bash
python -m resumegen.hiring --model mock --plan pilot
```
```bash
python -m resumegen.trajectory --model mock --pairs 20
```

You should see the mock's built-in preference (μ ≈ +0.2 in `full`, ≈ 0 in
`no_salary`, baseline ≈ 0, position bias ≈ 0.55). If those numbers are wrong,
stop — the analysis is wrong, not the model.

## 3. See what it would cost — still nothing sent

```bash
python -m resumegen.hiring --model gpt-4o-mini --plan pilot --dry-run
```
```bash
python -m resumegen.prose --dry-run
```

Every entry point prints its projection and refuses to run if it exceeds
`--budget` (default $25 per invocation; the prose step defaults to $8).

## 4. The pilot — about five cents

```bash
python -m resumegen.hiring --model gpt-4o-mini --plan pilot --budget 1
```

Look at three things in the table: `parse_fail_rate` (should be 0.00),
`pick_first_rate` (position bias; anything is fine, it is measured), and the
baseline cell `neutral|full|base` (μ should be near 0). Open
`out/prompt_log.jsonl` and read two or three entries end to end — that is the
record the project asked for.

## 5. The prose resumes — about $4, once

```bash
python -m resumegen.prose --budget 6
```

Writes 600 resumes to `out/prose/` (year-0 for both twins, year-1 loser,
year-1 winner under `both` and `move`), checks each one, and prints the
redaction-leak rate per view. Read ten of them. If the paper's snapshot
`gpt-4o-2024-08-06` is refused by the API, add `--model gpt-4o`.

## 6. The grid — under $2 on mini, then the primary cells on gpt-4o

```bash
python -m resumegen.hiring --model gpt-4o-mini --plan full --resumes template --budget 3
```
```bash
python -m resumegen.hiring --model gpt-4o-mini --plan full --resumes prose --budget 3
```
```bash
python -m resumegen.hiring --model gpt-4o --plan primary --resumes prose --budget 8
```

Each writes `out/hiring_<plan>_<model>_<resumes>_{trials,summary,contrasts}.csv`.

## 6b. "Several runs, several prompts" — under $1

```bash
python -m resumegen.hiring --model gpt-4o-mini --plan robustness --budget 1
```
```bash
python -m resumegen.hiring --model gpt-4o-mini --plan robustness --temperature 1.0 --replicates 3 --budget 2
```

The first varies the wording (three paraphrases of the same instruction); the
second samples every comparison three times and prints μ per run with its
spread. A finding counts only if it survives both.

## 7. Twenty years with the model choosing — under $1 on mini, ~$3 on gpt-4o

```bash
python -m resumegen.trajectory --model gpt-4o-mini --pairs 50 --budget 2
```
```bash
python -m resumegen.trajectory --model gpt-4o --pairs 50 --cells default:full,default:no_salary --budget 6
```

## The one-hour version

Two terminal windows, both with the key exported (step 1) and `stage4`
active. Window A runs gpt-4o-mini, window B runs gpt-4o; they draw on separate
rate-limit buckets, so they do not slow each other down.

Window A, in order, each after the previous finishes:

```bash
python -m resumegen.hiring --model mock --plan pilot
```
```bash
python -m resumegen.hiring --model gpt-4o-mini --plan pilot --budget 1
```
```bash
python -m resumegen.hiring --model gpt-4o-mini --plan robustness --budget 1
```
```bash
python -m resumegen.hiring --model gpt-4o-mini --plan full --resumes template --budget 3
```
```bash
python -m resumegen.trajectory --model gpt-4o-mini --pairs 50 --budget 2
```

Window B, started as soon as the pilot in A looks clean:

```bash
python -m resumegen.prose --budget 6 --workers 4
```
```bash
python -m resumegen.hiring --model gpt-4o --plan primary --resumes prose --budget 8 --workers 4
```

Window B is throttled by the 30,000-tokens-a-minute limit on gpt-4o, so
expect the prose step to take about half an hour and the primary run about an
hour; the progress line every 200 calls shows spend and pace. Everything in
window A finishes inside the hour. Total for both windows ≈ $12.

## Things that will happen

- **gpt-4o-mini crawls at ~7 calls a minute with "requests per day (RPD):
  Limit 10000" in the heartbeat.** Tier-1 accounts get 10,000 gpt-4o-mini
  requests per day, refilled continuously (10,000 / 1,440 min ≈ 7 a minute).
  One full grid is 6,800 calls, so it is one grid per day on mini. Ctrl+C,
  then print what is already done with no calls at all:
  `python -m resumegen.hiring --model gpt-4o-mini --plan full --resumes template --treatment salary --cache-only`
  — it reports which cells are complete and writes `*_partial_*.csv`. Rerun
  the same command without `--cache-only` the next day to finish; the cache
  resumes it. gpt-4o has its own, separate limit (30,000 tokens a minute, no
  daily cap at Tier 1), so Window B is unaffected.

- **The laptop sleeps and the run stalls.** On a Mac, prefix any run longer
  than a few minutes with `caffeinate -i` (built in; keeps the machine awake
  while the command runs): `caffeinate -i python -m resumegen.hiring ...`.
  If a run has stalled, press Ctrl+C and run the same command again — every
  finished call is cached, so the restart is free up to where it stopped, and
  the progress line will show the cache count climbing before new spend starts.

- **A run stops with "spent $X > budget".** Raise `--budget` and rerun; every
  finished call is cached, so you pay only for what is left.
- **A run is interrupted.** Rerun the same command; cached calls are free and
  are re-logged with `"cached": true`.
- **You want fresh answers, not cached ones.** Delete `out/llm_cache/` (the
  mock uses its own cache directories).
- **The model refuses a parameter** (`temperature`, `logprobs`): the wrapper
  drops it, retries, and records the drop in the `dropped_params` column.
- **Rate limits.** The wrapper waits the time the server asks for and does not
  count it as a failure; other transient errors back off up to five attempts,
  then the run stops with the error. Rerun to continue.

## What to share

`PREREG.md` (before running), then `out/prompts_used.json`, the three summary
CSVs, the trajectory summary, `out/prompt_log.jsonl`, and a short note listing
anything that deviated from the pre-registration. Not the key.
