"""
LLM-written resumes for the judge, and the redaction that makes the VIEWS
work on free text.

Generation follows the FAccT '25 appendix settings (gpt-4o-2024-08-06,
temperature 0.75, 768 max tokens) with the prompt in prompts.PROSE_PROMPT: the
paper's instruction plus a fixed work history with salaries and a layout rule
("Annual salary: $X" on its own line) so that salary redaction is a line
deletion rather than a guess.

Every generated resume is checked before it is accepted:
  * [NAME] and [EMAIL] present, no other bracketed placeholders
  * every employer in the career appears in the text
  * every salary line present under the full view
  * no employer that is NOT in the career (invented jobs)
Failures are regenerated once with a fresh seed, then flagged.

Redaction for a view (used by hiring.ResumeSource):
  no_salary   drop "Annual salary" lines; blank any residual $ amounts
  no_titles   replace each known title string, and bare ladder words, with a
              neutral phrase
  blind       both
The number of residual leaks is counted and reported, never hidden.

    python -m resumegen.prose --dry-run
    python -m resumegen.prose --model gpt-4o-2024-08-06 --budget 6
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from resumegen.career import LEVEL_TITLE
from resumegen import prompts as P
from resumegen.llm import make_client, guard_budget, BudgetExceeded

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
PROSE_DIR = os.path.join(OUT, "prose")

# ladder words only when they modify a role noun ("senior engineer", "Staff
# Backend Engineer"), so "Lead a team" and "Senior Living Partners" survive
LADDER_WORDS = re.compile(
    r"\b(Junior|Senior|Staff|Principal|Lead)\b(?=(?:\s+\w+){0,2}?\s+"
    r"(?:Engineer|Developer|Role|Position|SWE))", re.I)
MONEY = re.compile(r"(?:USD\s?|\$\s?)\d[\d,]*(?:\.\d+)?(?:\s?[kKmM])?\b|\b\d{2,3},\d{3}\s+(?:USD|dollars)\b")
# whole line, including its newline, so no blank line marks the deletion
SALARY_LINE = re.compile(
    r"^[ \t]*[-*•]?[ \t]*(?:annual\s+|base\s+)?(?:salary|compensation|pay)\b[^\n]*\n?",
    re.I | re.M)
NEUTRAL_TITLE = "[title withheld]"


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------

def prose_path(prose_dir: str, state_id: str) -> str:
    return os.path.join(prose_dir, f"{state_id}.txt")


def load_prose(prose_dir: str, state_id: str):
    p = prose_path(prose_dir, state_id)
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return f.read()


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

def check_resume(text: str, career, as_of: int) -> list[str]:
    problems = []
    if "[NAME]" not in text:
        problems.append("missing [NAME]")
    if "[EMAIL]" not in text:
        problems.append("missing [EMAIL]")
    extra = [m for m in re.findall(r"\[([A-Z _]{2,})\]", text) if m not in ("NAME", "EMAIL")]
    if extra:
        problems.append(f"other placeholders: {sorted(set(extra))}")
    c = career.as_of(as_of)
    real = [p for p in c.positions if p.employment_type != "gap"]
    for p in real:
        key = p.company.split(",")[0][:18].lower()
        if key not in text.lower():
            problems.append(f"employer missing: {p.company}")
    n_sal = len(SALARY_LINE.findall(text))
    if n_sal < len(real):
        problems.append(f"salary lines {n_sal} < roles {len(real)}")
    return problems


# ---------------------------------------------------------------------------
# Redaction
# ---------------------------------------------------------------------------

def redact(text: str, view: str, career) -> tuple[str, int]:
    """Apply a view to prose. Returns (text, n_residual_leaks)."""
    from resumegen.render import VIEWS
    cfg = VIEWS[view]
    out = text
    leaks = 0
    if not cfg["salary"]:
        out = SALARY_LINE.sub("", out)
        residual = MONEY.findall(out)
        leaks += len(residual)
        out = MONEY.sub("[amount withheld]", out)
    if not cfg["titles"]:
        # one pass, longest first, so the replacement text is never re-matched
        titles = sorted({p.title for p in career.positions if p.title}
                        | set(LEVEL_TITLE.values()), key=len, reverse=True)
        pat = re.compile("|".join(re.escape(t) for t in titles), re.I)
        out = pat.sub(NEUTRAL_TITLE, out)
        # ladder words left in free text ("senior engineer with ...")
        residual = LADDER_WORDS.findall(out)
        leaks += len(residual)
        out = LADDER_WORDS.sub("", out)
        out = re.sub(r"[ \t]{2,}", " ", out)
    if not cfg["tenure"]:
        out = re.sub(r"\(?\b(19|20)\d{2}\s?(?:[–—-]|to)\s?((19|20)\d{2}|Present)\)?", "", out, flags=re.I)
    if not cfg["highlights"]:
        out = "\n".join(ln for ln in out.splitlines()
                        if not re.match(r"^\s*[-*•]", ln))
    while "\n\n\n" in out:
        out = out.replace("\n\n\n", "\n\n")
    return out.strip(), leaks


def leak_report(prose_dir: str, states: dict) -> pd.DataFrame:
    rows = []
    for sid, st in states.items():
        raw = load_prose(prose_dir, sid)
        if raw is None:
            continue
        for view in ("no_salary", "no_titles", "blind"):
            _, n = redact(raw, view, st.career)
            rows.append({"state_id": sid, "view": view, "leaks": n})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def _gen_one(st, client, prose_dir):
    path = prose_path(prose_dir, st.state_id)
    if os.path.exists(path) and not client.dry_run:
        return {"state_id": st.state_id, "status": "exists"}
    msgs = P.prose_messages(st.career, st.persona, st.year)
    problems, text = ["not run"], ""
    for attempt in range(2):
        rep = client.chat(msgs, max_tokens=P.PROSE_MAX_TOKENS,
                          temperature=P.PROSE_TEMPERATURE, seed=attempt,
                          tag=f"prose|{st.role}|{st.treatment}")
        if client.dry_run:
            problems = []
            break
        text = rep.content
        problems = check_resume(text, st.career, st.year)
        if not problems:
            break
    if not client.dry_run:
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            f.write(text)
        os.replace(tmp, path)
    return {"state_id": st.state_id, "pair_id": st.pair_id, "who": st.who, "year": st.year,
            "treatment": st.treatment, "role": st.role,
            "status": "ok" if not problems else "flagged",
            "problems": "; ".join(problems), "chars": len(text)}


def generate(states: dict, client, prose_dir: str = PROSE_DIR, only=None,
             workers: int = 8) -> pd.DataFrame:
    from concurrent.futures import ThreadPoolExecutor, as_completed
    os.makedirs(prose_dir, exist_ok=True)
    todo = [st for st in states.values()
            if not (only and st.treatment not in only and st.role != "base")]
    rows = []
    workers = 1 if client.dry_run else max(1, workers)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_gen_one, st, client, prose_dir) for st in todo]
        try:
            for i, fut in enumerate(as_completed(futs), 1):
                rows.append(fut.result())
                if not client.dry_run and i % 50 == 0:
                    print(f"  {i}/{len(todo)} resumes  {client.meter.summary()}", flush=True)
        except BudgetExceeded as e:
            print(f"\n!! {e}")
            for f in futs:
                f.cancel()
    df = pd.DataFrame(rows)
    if not client.dry_run and len(df):
        idx = os.path.join(prose_dir, "index.csv")
        old = pd.read_csv(idx) if os.path.exists(idx) else pd.DataFrame()
        new = pd.concat([old, df[df.status != "exists"]], ignore_index=True)
        new.drop_duplicates("state_id", keep="last").to_csv(idx, index=False)
    return df


def main(argv=None):
    from resumegen.hiring import load_pairs, build_states
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=P.PROSE_MODEL)
    ap.add_argument("--pairs", type=int, default=120,
                    help="cover the largest plan you will run (primary = 120)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--treatments", default="both,move,salary,title",
                    help="which winner treatments get prose (loser and year-0 always do)")
    ap.add_argument("--budget", type=float, default=8.0)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--workers", type=int, default=6, help="parallel API calls in flight")
    ap.add_argument("--show", type=int, default=0,
                    help="print N generated resumes with their blind redaction and exit (no calls)")
    ap.add_argument("--psid", default="data/psid_small.csv")
    ap.add_argument("--jobs", default="data/swe_jobs_small.csv")
    args = ap.parse_args(argv)

    pool, pairs, dropped = load_pairs(args.pairs, args.seed, args.psid, args.jobs)
    states, _ = build_states(pairs, pool)
    if args.show:
        import random
        have = [s for s in states.values() if load_prose(PROSE_DIR, s.state_id) is not None]
        random.Random(0).shuffle(have)
        for st in have[:args.show]:
            raw = load_prose(PROSE_DIR, st.state_id)
            red, n = redact(raw, "blind", st.career)
            print("=" * 78 + f"\n{st.state_id}  (role: {st.role}, treatment: {st.treatment})\n" + "=" * 78)
            print(raw)
            print("-" * 30 + f" blind view, {n} fallback scrubs " + "-" * 30)
            print(red)
            print()
        return
    only = set(args.treatments.split(",")) | {"loser"}
    n = sum(1 for s in states.values() if s.role == "base" or s.treatment in only)
    print(f"{len(pairs)} pairs -> {n} resumes to write with {args.model} "
          f"(temperature {P.PROSE_TEMPERATURE}, max {P.PROSE_MAX_TOKENS} tokens)"
          + (" [DRY RUN]" if args.dry_run else ""))
    dry = make_client(args.model, temperature=P.PROSE_TEMPERATURE, dry_run=True)
    generate(states, dry, only=only)
    guard_budget(dry, args.budget, per_call_out=650)
    if args.dry_run:
        return
    client = make_client(args.model, temperature=P.PROSE_TEMPERATURE, budget_usd=args.budget)
    df = generate(states, client, only=only, workers=args.workers)
    if not len(df):
        print("nothing generated")
        return
    print(df.status.value_counts().to_string())
    flagged = df[df.status == "flagged"]
    if len(flagged):
        print("\nflagged (kept, but look at them):")
        print(flagged[["state_id", "problems"]].head(20).to_string(index=False))
    lr = leak_report(PROSE_DIR, states)
    if len(lr):
        print("\nredaction leak rate by view (share of resumes with any residual token):")
        print(lr.groupby("view").leaks.apply(lambda s: (s > 0).mean()).round(3).to_string())
    print("\n" + client.meter.summary())


if __name__ == "__main__":
    main()
