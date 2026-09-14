"""
The hiring experiment: does an LLM screener re-transmit a prior outcome?

Design (PREREG.md has the full version):

  Year 0   twins A and B, matched on every merit dimension, both apply for
           posting J0. In the BASELINE cell the model chooses between them as
           they are -- this measures whether it treats the twins as equals,
           and its position bias.
  Year 1   one twin (the WINNER, A in even pairs and B in odd pairs) took J0.
           Its resume now carries that outcome. The other (the LOSER) stayed
           put. Both apply for posting J1. The model chooses.

  The TREATMENT says what the winner's resume carries:
     move    a new employer, same title, same pay          (the bare fact of
                                                            having been chosen)
     salary  a new employer, same title, +12% pay
     title   a new employer, one level up, same pay
     both    a new employer, one level up, +12% pay        (the market's offer)

  "+12%" is on the winner's own prior pay; the loser's pay grows 2.5% in place,
  so the visible gap is +9.3%. Both twins gain one year and two new
  accomplishment bullets, so the ONLY differences are the treatment's.

  The VIEW says what the model can see (redaction); the PROMPT says what it is
  told to do with it (instruction). Every comparison is run in both
  presentation orders, so position bias cancels in the estimate and is
  reported on its own.

  mu = P(model picks the winner) - 1/2, averaged over orders, pairs as the
  unit. mu > 0 means yesterday's arbitrary outcome is being carried into
  today's decision by the model itself -- the memory coefficient.

Usage (see RUNBOOK.md):
    python -m resumegen.hiring --model mock --plan pilot            # offline check
    python -m resumegen.hiring --model gpt-4o-mini --plan pilot --dry-run
    python -m resumegen.hiring --model gpt-4o-mini --plan pilot --budget 1
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from resumegen.traits import personas_from_psid
from resumegen.career import (PostingPool, Position, Career, HIGHLIGHTS, _fill,
                              LEVEL_MIN_YOE, MAX_LEVEL, SPECIALISMS)
from resumegen.twins import make_pair, verify_pair, pair_specialise, _rng as twin_rng
from resumegen.generate import load_jobs
from resumegen.render import render_offline, VIEWS
from resumegen import prompts as P
from resumegen.llm import make_client, guard_budget, BudgetExceeded

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
PROSE_DIR = os.path.join(OUT, "prose")

YEAR0, YEAR1 = 2021, 2022
TREATMENTS = {
    #          step_up  premium
    "move":   (False,   1.025),
    "salary": (False,   1.12),
    "title":  (True,    1.025),
    "both":   (True,    1.12),
}
STAY_GROWTH = 1.025


# ---------------------------------------------------------------------------
# States
# ---------------------------------------------------------------------------

@dataclass
class State:
    state_id: str
    pair_id: int
    who: str            # "A" | "B"
    year: int
    treatment: str      # "base" | treatment name
    role: str           # "base" | "winner" | "loser"
    persona: object
    career: Career


def pair_specialism(pair_id: int) -> str:
    return str(twin_rng("spec", pair_id).choice(SPECIALISMS))


def year0_posting(pair, pool) -> dict:
    """J0 (year 0, the job the winner takes) and J1 (year 1, the job both
    apply for). Same level and specialism; J1 is at a DIFFERENT employer, or
    the winner would be applying to the company it already works for."""
    cur = pair.a_career.current()
    yoe = pair.a_career.years_experience(YEAR0)
    reach = cur.level
    if cur.level < MAX_LEVEL and yoe >= LEVEL_MIN_YOE[cur.level + 1]:
        reach = cur.level + 1
    rng = twin_rng("posting", pair.pair_id)
    spec = pair_specialism(pair.pair_id)
    j0 = {"level": reach, "company": pool.company(rng), "location": pool.location(rng),
          "specialism": spec}
    held = {p.company for p in pair.a_career.positions} | {p.company for p in pair.b_career.positions}
    c1 = pool.company(rng)
    while c1 == j0["company"] or c1 in held:
        c1 = pool.company(rng)
    j1 = {"level": reach, "company": c1, "location": pool.location(rng), "specialism": spec}
    return {**j0, "j1": j1}


EXTRA_HIGHLIGHTS = [
    "Ran the on-call rotation for {area} and cut mean time to recovery by {pct}%.",
    "Introduced automated integration tests covering {area}, raising coverage by {pct}%.",
    "Interviewed and onboarded {n} engineers.",
    "Wrote the runbooks and dashboards used by {n} teams for {area}.",
    "Delivered {feature} on schedule with a team of {n}.",
    "Retired {old} from {area}, removing a recurring source of incidents.",
]


def _tpl_used(tpl: str, existing: set[str]) -> bool:
    import re
    pat = re.compile("^" + ".+?".join(re.escape(x) for x in re.split(r"\{\w+\}", tpl)) + "$")
    return any(pat.match(h) for h in existing)


def _fresh_highlights(career: Career, level: int, rng, n: int = 2) -> list[str]:
    """`n` accomplishment bullets whose TEMPLATE is not already on the resume,
    so a second year in the same job does not read as a padded repeat."""
    existing = {h for p in career.positions for h in p.highlights}
    bucket = list(HIGHLIGHTS[min(level, MAX_LEVEL)])
    rng.shuffle(bucket)
    reserve = list(EXTRA_HIGHLIGHTS)
    rng.shuffle(reserve)
    out = []
    for tpl in bucket + reserve:
        if _tpl_used(tpl, existing):
            continue
        h = _fill(tpl, rng)
        out.append(h)
        existing.add(h)
        if len(out) >= n:
            break
    return out


def make_loser(career: Career, pair_id: int, who: str) -> Career:
    """One more year in the same job: a raise and two new accomplishments, so
    the winner's extra bullets are not a difference in content, only in where
    the year was spent."""
    c = copy.deepcopy(career)
    cur = c.current()
    cur.salary = int(round(cur.salary * STAY_GROWTH, -2))
    rng = twin_rng("loser", pair_id, who)
    cur.highlights = cur.highlights + _fresh_highlights(c, cur.level, rng)
    return c


def make_winner(career: Career, posting: dict, treatment: str, pool: PostingPool,
                pair_id: int, who: str) -> Career:
    step_up, premium = TREATMENTS[treatment]
    c = copy.deepcopy(career)
    cur = c.current()
    level = posting["level"] if step_up else cur.level
    rng = twin_rng("winner", pair_id, who, treatment)
    cur.year_end = YEAR0
    c.positions.append(Position(
        year_start=YEAR0, year_end=None,
        title=pool.title(level, rng, posting["specialism"],
                         specialise=pair_specialise(pair_id)),
        company=posting["company"], level=level,
        salary=int(round(cur.salary * premium, -2)),
        location=posting["location"], is_promotion=False,
        highlights=_fresh_highlights(c, level, rng)))
    return c


def build_states(pairs, pool, treatments=tuple(TREATMENTS)) -> tuple[dict, dict]:
    """All candidate states, and the postings per pair."""
    states, postings = {}, {}
    for i, pair in enumerate(pairs):
        pid = pair.pair_id
        post = year0_posting(pair, pool)
        postings[pid] = post
        winner = "A" if i % 2 == 0 else "B"
        for who, persona, career in (("A", pair.a_persona, pair.a_career),
                                     ("B", pair.b_persona, pair.b_career)):
            sid = f"p{pid}_{who}_{YEAR0}_base"
            states[sid] = State(sid, pid, who, YEAR0, "base", "base", persona, career)
            if who == winner:
                for t in treatments:
                    sid = f"p{pid}_{who}_{YEAR1}_{t}"
                    states[sid] = State(sid, pid, who, YEAR1, t, "winner", persona,
                                        make_winner(career, post, t, pool, pid, who))
            else:
                sid = f"p{pid}_{who}_{YEAR1}_loser"
                states[sid] = State(sid, pid, who, YEAR1, "loser", "loser", persona,
                                    make_loser(career, pid, who))
    return states, postings


# ---------------------------------------------------------------------------
# Resume text for the judge
# ---------------------------------------------------------------------------

def anonymise(text: str) -> str:
    lines = []
    for ln in text.splitlines():
        s = ln.replace("[NAME]", "").replace("[EMAIL]", "").strip(" |")
        if not s and (("[NAME]" in ln) or ("[EMAIL]" in ln)):
            continue
        lines.append(s if ("[NAME]" in ln or "[EMAIL]" in ln) else ln)
    out = "\n".join(lines).strip()
    while "\n\n\n" in out:
        out = out.replace("\n\n\n", "\n\n")
    return out


class ResumeSource:
    """template: deterministic renderer, views native.
       prose:    GPT-written text from out/prose/, views by redaction."""

    def __init__(self, kind: str = "template", prose_dir: str = PROSE_DIR):
        assert kind in ("template", "prose")
        self.kind, self.prose_dir = kind, prose_dir
        self.leaks = []          # (state_id, view, n_leaks)

    def text(self, st: State, view: str) -> str:
        if self.kind == "template":
            return anonymise(render_offline(st.career, st.persona, st.year, view))
        from resumegen.prose import load_prose, redact
        raw = load_prose(self.prose_dir, st.state_id)
        if raw is None:
            raise FileNotFoundError(
                f"no prose resume for {st.state_id} in {self.prose_dir}; run "
                f"`python -m resumegen.prose` first, or use --resumes template")
        txt, n_leaks = redact(raw, view, st.career)
        if n_leaks:
            self.leaks.append((st.state_id, view, n_leaks))
        return anonymise(txt)


# ---------------------------------------------------------------------------
# Cells and plans
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Cell:
    prompt_id: str
    view: str
    treatment: str      # "base" for the year-0 baseline

    @property
    def name(self) -> str:
        return f"{self.prompt_id}|{self.view}|{self.treatment}"


def plan_cells(plan: str, treatment: str = "both") -> list[Cell]:
    """`treatment` is the dose used for the prompt x view cells. The pilot put
    `both` at the ceiling on gpt-4o-mini (the winner chosen in 40/40), so the
    prompt and view contrasts need a weaker dose to be measurable; run
    `--plan decomposition` first and pick the strongest off-ceiling treatment."""
    base = Cell("neutral", "full", "base")
    tie_base = Cell("tie_allowed", "full", "base")      # can the model ever say "equal"?
    t = treatment
    if plan == "pilot":
        return [base,
                Cell("neutral", "full", t), Cell("neutral", "no_salary", t),
                Cell("ignore_comp", "full", t), Cell("rubric", "full", t),
                Cell("tie_allowed", "full", t), Cell("reason", "full", t)]
    if plan == "primary":
        # the single-channel redaction that matches the dose: pay for salary/both, titles for title
        redact_view = "no_titles" if t == "title" else "no_salary"
        return [base, tie_base,
                Cell("neutral", "full", t), Cell("neutral", redact_view, t),
                Cell("neutral", "blind", t), Cell("ignore_comp", "full", t),
                Cell("rubric", "full", t), Cell("neutral", "full", "move")]
    if plan == "robustness":
        # "same experiment, several prompts": paraphrases x the primary views
        return [Cell(pid, view, t) for pid in P.PARAPHRASES
                for view in ("full", "no_salary")]
    if plan == "full":
        cells = [base, tie_base]
        for pid in P.ALL_PROMPTS:
            views = ("full", "no_salary") if pid in ("neutral_b", "neutral_c") \
                else ("full", "no_salary", "no_titles", "blind")
            for view in views:
                cells.append(Cell(pid, view, t))
        for tr in TREATMENTS:
            if tr != t:
                cells.append(Cell("neutral", "full", tr))
        cells.append(Cell("neutral", "no_history", t))
        return cells
    if plan == "decomposition":
        return [Cell("neutral", "full", tr) for tr in TREATMENTS] + \
               [Cell("neutral", "no_salary", "salary"), Cell("neutral", "no_titles", "title")]
    raise ValueError(plan)


# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------

def trials_for_cell(cell: Cell, pairs, states, postings, source: ResumeSource):
    """Two trials per pair: winner first, loser first."""
    out = []
    for i, pair in enumerate(pairs):
        pid = pair.pair_id
        post = postings[pid]
        if cell.treatment == "base":
            w, l = "A", "B"       # "winner" is a label only; mu measures A-vs-B
            sw, sl = states[f"p{pid}_A_{YEAR0}_base"], states[f"p{pid}_B_{YEAR0}_base"]
        else:
            w = "A" if i % 2 == 0 else "B"
            l = "B" if w == "A" else "A"
            sw = states[f"p{pid}_{w}_{YEAR1}_{cell.treatment}"]
            sl = states[f"p{pid}_{l}_{YEAR1}_loser"]
        j = post if cell.treatment == "base" else post["j1"]
        posting = P.posting_text(j["level"], j["company"], j["location"], j["specialism"])
        rw, rl = source.text(sw, cell.view), source.text(sl, cell.view)
        for first in ("W", "L"):
            r1, r2 = (rw, rl) if first == "W" else (rl, rw)
            out.append({"cell": cell.name, "prompt_id": cell.prompt_id, "view": cell.view,
                        "treatment": cell.treatment, "pair_id": pid, "winner": w,
                        "first": first, "posting": posting, "r1": r1, "r2": r2,
                        "state_first": (sw if first == "W" else sl).state_id,
                        "state_second": (sl if first == "W" else sw).state_id})
    return out


def _one_trial(client, prompt, cell, tr, r, fmt, tag_prefix):
    msgs = prompt.messages(tr["posting"], tr["r1"], tr["r2"])
    rep = client.chat(msgs, max_tokens=prompt.max_tokens,
                      logprobs=(prompt.answer_mode == "direct"),
                      response_format=fmt, tag=tag_prefix + cell.name, seed=r)
    if client.dry_run:
        choice, status = None, "dry"
    elif rep.finish_reason == "skipped":
        choice, status = None, "skipped"          # --cache-only and not in cache
    else:
        choice, status = P.parse_choice(rep.content, prompt.answer_mode)
    if choice == 0 and cell.prompt_id != "tie_allowed":
        choice, status = None, "unrequested_tie"
    pos_w = 1 if tr["first"] == "W" else 2
    pick_w = None if choice in (None, 0) else float(choice == pos_w)
    p_first = P.p_first_from_logprobs(rep.first_token_logprobs)
    p_w = None if p_first is None else (p_first if pos_w == 1 else 1 - p_first)
    return ({k: tr[k] for k in ("cell", "prompt_id", "view", "treatment",
                                "pair_id", "winner", "first",
                                "state_first", "state_second")}
            | {"rep": r, "choice": choice, "status": status, "pick_winner": pick_w,
               "tie": None if choice is None else int(choice == 0),
               "pick_first": None if choice in (None, 0) else int(choice == 1),
               "p_first": p_first, "p_winner": p_w,
               "prompt_tokens": rep.prompt_tokens,
               "completion_tokens": rep.completion_tokens,
               "cached": rep.cached, "model": rep.model,
               "dropped_params": ",".join(rep.dropped_params),
               "raw": rep.content[:400], "call_id": rep.call_id})


def run(cells, pairs, states, postings, source, client, tag_prefix="", replicates=1,
        workers: int = 8, progress: bool = True):
    """All trials of all cells, `workers` API calls in flight at once. Stops
    cleanly at the budget; whatever finished is returned (and cached)."""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    jobs = []
    for cell in cells:
        prompt = P.DECISION_PROMPTS[cell.prompt_id]
        fmt = P.JSON_FORMAT if prompt.answer_mode == "json" else None
        for tr in trials_for_cell(cell, pairs, states, postings, source):
            for r in range(replicates):
                jobs.append((prompt, cell, tr, r, fmt))
    from concurrent.futures import TimeoutError as FutTimeout
    rows, done, t0 = [], 0, time.time()
    workers = 1 if client.dry_run else max(1, workers)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_one_trial, client, p_, c_, tr_, r_, f_, tag_prefix)
                for p_, c_, tr_, r_, f_ in jobs]
        pending = set(futs)
        try:
            while pending:
                try:
                    for fut in as_completed(pending, timeout=45):
                        pending.discard(fut)
                        rows.append(fut.result())
                        done += 1
                        if progress and not client.dry_run and done % 200 == 0:
                            el = time.time() - t0
                            print(f"  {done}/{len(jobs)} calls  {el:5.0f}s  "
                                  f"{client.meter.summary()}", flush=True)
                except FutTimeout:
                    # periodic status (every 45 s): pace, ETA, and why it is slow if it is
                    el = time.time() - t0
                    new_calls = client.meter.calls - client.meter.cached
                    rate = new_calls / el * 60 if el > 0 else 0.0
                    eta = (len(pending) / rate) if rate > 0 else float("inf")
                    thr = getattr(client, "throttle_events", 0)
                    why = (f"; throttled {thr}x (server-paced, normal for gpt-4o at Tier 1)"
                           if thr else ("; no 429s -- if pace is ~0, check the network" if rate < 1 else ""))
                    kind = ""
                    if thr and "per day" in getattr(client, "last_throttle", ""):
                        kind = "; DAILY cap hit -- stop and resume tomorrow, see RUNBOOK"
                    print(f"  {el:5.0f}s  {done}/{len(jobs)} done, {len(pending)} pending, "
                          f"{rate:.0f} new calls/min, ~{eta/60:.0f} min left{why}{kind}", flush=True)
        except BudgetExceeded as e:
            print(f"\n!! {e}")
            for f in futs:
                f.cancel()
        except Exception:
            for f in futs:
                f.cancel()
            raise
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def per_pair_rep(trials: pd.DataFrame) -> pd.DataFrame:
    """Order-averaged outcome per (cell, pair, replicate). Ties count as 1/2."""
    t = trials.copy()
    if "rep" not in t:
        t["rep"] = 0
    t["mu_obs"] = t.pick_winner.where(t.tie != 1, 0.5)
    g = t.groupby(["cell", "prompt_id", "view", "treatment", "pair_id", "rep"])
    return g.agg(mu_pair=("mu_obs", "mean"),
                 n_orders=("mu_obs", lambda s: s.notna().sum()),
                 tie_any=("tie", "max"),
                 p_winner=("p_winner", "mean"),
                 consistent=("pick_winner", lambda s: float(s.nunique() == 1) if s.notna().sum() == 2 else np.nan),
                 ).reset_index()


def per_pair(trials: pd.DataFrame) -> pd.DataFrame:
    """Per (cell, pair): the mean over complete replicates (both orders parsed)."""
    ppr = per_pair_rep(trials)
    ok = ppr[ppr.n_orders == 2]
    g = ok.groupby(["cell", "prompt_id", "view", "treatment", "pair_id"])
    pp = g.agg(mu_pair=("mu_pair", "mean"), n_reps=("rep", "nunique"),
               tie_any=("tie_any", "max"), p_winner=("p_winner", "mean"),
               consistent=("consistent", "mean")).reset_index()
    pp["n_orders"] = 2      # by construction after the filter above
    return pp


def across_runs(trials: pd.DataFrame) -> pd.DataFrame:
    """mu per (cell, replicate), and its spread across replicates -- the
    'what is significant across runs' question. With one replicate this is just mu."""
    ppr = per_pair_rep(trials)
    ok = ppr[ppr.n_orders == 2]
    per = ok.groupby(["cell", "rep"]).mu_pair.mean().sub(0.5).rename("mu").reset_index()
    out = per.groupby("cell", sort=False).mu.agg(reps="count", mean="mean", sd="std",
                                                min="min", max="max").reset_index()
    return out


def summarise(trials: pd.DataFrame) -> pd.DataFrame:
    from scipy import stats
    pp = per_pair(trials)
    rows = []
    for cell, sub in pp.groupby("cell", sort=False):
        ok = sub[sub.n_orders == 2]
        mu = ok.mu_pair - 0.5
        n = len(mu)
        if n == 0:
            continue
        se = mu.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan
        tstat = mu.mean() / se if se and se > 0 else np.nan
        p = 2 * stats.t.sf(abs(tstat), n - 1) if n > 1 and np.isfinite(tstat) else np.nan
        tr = trials[trials.cell == cell]
        tcrit = stats.t.ppf(0.975, n - 1) if n > 1 else np.nan
        nontie = ok[ok.tie_any != 1]
        mu_nt = nontie.mu_pair - 0.5
        rows.append({
            "cell": cell, "prompt": sub.prompt_id.iloc[0], "view": sub.view.iloc[0],
            "treatment": sub.treatment.iloc[0],
            "pairs": n, "trials": len(tr),
            "mu": mu.mean(), "se": se, "ci_lo": mu.mean() - tcrit * se, "ci_hi": mu.mean() + tcrit * se,
            "t": tstat, "p": p,
            "pick_winner_rate": ok.mu_pair.mean(),
            "mu_nontie": mu_nt.mean() if len(mu_nt) else np.nan,
            "pairs_nontie": len(mu_nt),
            "logprob_p_winner": ok.p_winner.mean() if ok.p_winner.notna().any() else np.nan,
            "consistent_rate": ok.consistent.mean(),
            "tie_rate": tr.tie.fillna(0).mean(),
            "pick_first_rate": tr.pick_first.mean(),
            "parse_fail_rate": tr.status.isin(["empty", "unparsed", "bad_json", "bad_choice",
                                               "unrequested_tie"]).mean(),
            "skipped": int((tr.status == "skipped").sum()),
            "cached_share": tr.cached.mean(),
        })
    return pd.DataFrame(rows)


def contrast_list(t: str = "both") -> list[tuple[str, str, str]]:
    """(label, cell_x, cell_y) -> mu_x - mu_y over shared pairs, paired t.
    `t` is the dose used for the prompt x view cells."""
    return [
        ("H2 move vs both",             "neutral|full|move",        "neutral|full|both"),
        ("H2 salary vs both",           "neutral|full|salary",      "neutral|full|both"),
        ("H2 title vs both",            "neutral|full|title",       "neutral|full|both"),
        ("H2 salary: redacted vs shown","neutral|no_salary|salary", "neutral|full|salary"),
        ("H2 title: redacted vs shown", "neutral|no_titles|title",  "neutral|full|title"),
        ("H3 redact salary vs full",    f"neutral|no_salary|{t}",   f"neutral|full|{t}"),
        ("H3 redact titles vs full",    f"neutral|no_titles|{t}",   f"neutral|full|{t}"),
        ("H3 blind vs full",            f"neutral|blind|{t}",       f"neutral|full|{t}"),
        ("H3 instruct vs full",         f"ignore_comp|full|{t}",    f"neutral|full|{t}"),
        ("H3 ban framing vs full",      f"ban_framing|full|{t}",    f"neutral|full|{t}"),
        ("H3 REDACTION vs INSTRUCTION", f"neutral|no_salary|{t}",   f"ignore_comp|full|{t}"),
        ("H3 TITLE-REDACTION vs INSTRUCTION", f"neutral|no_titles|{t}", f"ignore_comp|full|{t}"),
        ("H3 BLIND vs INSTRUCTION",     f"neutral|blind|{t}",       f"ignore_comp|full|{t}"),
        ("H4 rubric vs neutral",        f"rubric|full|{t}",         f"neutral|full|{t}"),
        ("H4 reason vs neutral",        f"reason|full|{t}",         f"neutral|full|{t}"),
        ("H5 tie-allowed vs neutral",   f"tie_allowed|full|{t}",    f"neutral|full|{t}"),
        ("wording: neutral_b vs neutral", f"neutral_b|full|{t}",    f"neutral|full|{t}"),
        ("wording: neutral_c vs neutral", f"neutral_c|full|{t}",    f"neutral|full|{t}"),
        ("wording: redaction under neutral_b", f"neutral_b|no_salary|{t}", f"neutral_b|full|{t}"),
        ("wording: redaction under neutral_c", f"neutral_c|no_salary|{t}", f"neutral_c|full|{t}"),
        ("floor no_history",            f"neutral|no_history|{t}",  f"neutral|full|{t}"),
    ]


def contrasts(trials: pd.DataFrame, treatment: str = "both") -> pd.DataFrame:
    from scipy import stats
    pp = per_pair(trials)
    pp = pp[pp.n_orders == 2]
    wide = pp.pivot(index="pair_id", columns="cell", values="mu_pair")
    rows = []
    for label, x, y in contrast_list(treatment):
        if x not in wide or y not in wide:
            continue
        d = (wide[x] - wide[y]).dropna()
        if len(d) < 3:
            continue
        se = d.std(ddof=1) / np.sqrt(len(d))
        t = d.mean() / se if se > 0 else np.nan
        rows.append({"contrast": label, "x": x, "y": y, "pairs": len(d),
                     "mu_x": wide.loc[d.index, x].mean() - 0.5,
                     "mu_y": wide.loc[d.index, y].mean() - 0.5,
                     "diff": d.mean(), "se": se, "t": t,
                     "p": 2 * stats.t.sf(abs(t), len(d) - 1) if np.isfinite(t) else np.nan})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------

PAIR_POOL = 140   # every plan draws its pairs from the same fixed pool


def load_pairs(n_pairs: int, seed: int, psid_path: str, jobs_path: str):
    """The first `n_pairs` usable pairs of a FIXED pool of PAIR_POOL, so the
    pilot's 20 pairs are the first 20 of the full grid's 100, which are the
    first 100 of the primary run's 120 -- and one set of prose resumes serves
    all three. (pandas' sample() is not prefix-stable in n, so the pool size
    must not vary between plans.)"""
    pool = PostingPool(load_jobs(jobs_path))
    psid = pd.read_csv(psid_path, low_memory=False)
    bases = personas_from_psid(psid, n=max(n_pairs, PAIR_POOL), seed=seed)
    pairs = [make_pair(b, pool, i, seed=seed) for i, b in enumerate(bases)]
    checks = [verify_pair(p) for p in pairs]
    kept = [p for p, c in zip(pairs, checks) if c["equivalent"]]
    # the title/both treatments need a level above the current one to exist
    eligible = [p for p in kept if year0_posting(p, pool)["level"] > p.a_career.current().level]
    return pool, eligible[:n_pairs], len(pairs) - len(eligible)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mock")
    ap.add_argument("--plan", default="pilot",
                    choices=["pilot", "primary", "full", "decomposition", "robustness"])
    ap.add_argument("--treatment", default="both", choices=list(TREATMENTS),
                    help="dose for the prompt x view cells (see plan_cells)")
    ap.add_argument("--replicates", type=int, default=1,
                    help="repeat every call N times with a different sampling seed; "
                         "only meaningful with --temperature > 0 (at 0 the runs are identical)")
    ap.add_argument("--pairs", type=int, default=None,
                    help="default: 20 for pilot, 120 for primary, 100 otherwise")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resumes", default="template", choices=["template", "prose"])
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--budget", type=float, default=25.0, help="USD hard stop for this run")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--cache-only", action="store_true",
                    help="no API calls: summarise whatever this plan already has in the cache")
    ap.add_argument("--workers", type=int, default=4,
                    help="parallel API calls in flight (4 saturates a Tier-1 gpt-4o-mini limit without 429 storms)")
    ap.add_argument("--psid", default="data/psid_small.csv")
    ap.add_argument("--jobs", default="data/swe_jobs_small.csv")
    ap.add_argument("--out-prefix", default=None)
    args = ap.parse_args(argv)

    n_pairs = args.pairs or {"pilot": 20, "primary": 120}.get(args.plan, 100)
    pool, pairs, dropped = load_pairs(n_pairs, args.seed, args.psid, args.jobs)
    print(f"{len(pairs)} matched pairs (from a pool of {PAIR_POOL}, of which {dropped} "
          f"failed the equivalence audit or sit at the top of the ladder)")
    states, postings = build_states(pairs, pool)
    source = ResumeSource(args.resumes)
    cells = plan_cells(args.plan, args.treatment)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "prompts_used.json"), "w") as f:
        json.dump(P.prompt_registry(), f, indent=2, ensure_ascii=False)

    prefix = args.out_prefix or f"hiring_{args.plan}_{args.model}_{args.resumes}" + \
        (f"_{args.treatment}" if args.treatment != "both" else "") + \
        (f"_T{args.temperature}_x{args.replicates}" if args.replicates > 1 else "")
    print(f"plan={args.plan}: {len(cells)} cells x {len(pairs)} pairs x 2 orders "
          f"= {2 * len(cells) * len(pairs)} calls on {args.model}"
          + (" [DRY RUN]" if args.dry_run else ""))

    # always project first; refuse to send if the projection exceeds the budget
    if not args.cache_only:
        dry = make_client(args.model, temperature=args.temperature, dry_run=True)
        run(cells, pairs, states, postings, source, dry, replicates=args.replicates)
        guard_budget(dry, args.budget)
        if args.dry_run:
            return
    client = make_client(args.model, temperature=args.temperature, budget_usd=args.budget,
                         cache_only=args.cache_only)
    trials = run(cells, pairs, states, postings, source, client, replicates=args.replicates,
                 workers=args.workers)
    if not len(trials):
        print("nothing completed; nothing to summarise")
        return
    if args.cache_only:
        done_cells = trials[trials.status != "skipped"].groupby("cell").size()
        total = trials.groupby("cell").size()
        partial = [(c, int(done_cells.get(c, 0)), int(total[c])) for c in total.index
                   if done_cells.get(c, 0) < total[c]]
        print(f"cache-only: {int((trials.status != 'skipped').sum())}/{len(trials)} trials available; "
              f"{len(total) - len(partial)} of {len(total)} cells complete")
        if partial:
            print("  incomplete cells: " + ", ".join(f"{c} ({d}/{t})" for c, d, t in partial))
        prefix += "_partial"

    trials.to_csv(os.path.join(OUT, f"{prefix}_trials.csv"), index=False)
    summ = summarise(trials)
    summ.to_csv(os.path.join(OUT, f"{prefix}_summary.csv"), index=False)
    con = contrasts(trials, args.treatment)
    con.to_csv(os.path.join(OUT, f"{prefix}_contrasts.csv"), index=False)

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    show = summ[["cell", "pairs", "mu", "se", "p", "mu_nontie", "logprob_p_winner",
                 "consistent_rate", "tie_rate", "pick_first_rate", "parse_fail_rate"]]
    print("\nPer cell  (mu = P(pick prior winner) - 1/2, order-averaged; baseline cell measures A-vs-B)")
    print(show.round(3).to_string(index=False))
    if len(con):
        print("\nPre-registered contrasts (paired over the same pairs)")
        print(con[["contrast", "pairs", "mu_x", "mu_y", "diff", "se", "p"]].round(3).to_string(index=False))
    if args.replicates > 1:
        ar = across_runs(trials)
        ar.to_csv(os.path.join(OUT, f"{prefix}_across_runs.csv"), index=False)
        print(f"\nAcross {args.replicates} replicates (temperature {args.temperature}): mu per run")
        print(ar.round(3).to_string(index=False))
    if source.leaks:
        print(f"\nredaction leaks: {len(source.leaks)} (state, view) resumes with residual "
              f"salary/title tokens -- see prose.leak_report()")
    print("\n" + client.meter.summary())
    print(f"wrote out/{prefix}_trials.csv, _summary.csv, _contrasts.csv; prompts in out/prompts_used.json; "
          f"every call in out/prompt_log.jsonl")


if __name__ == "__main__":
    main()
