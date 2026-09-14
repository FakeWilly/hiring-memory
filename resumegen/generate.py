"""
Build the resume database.

    python -m resumegen.generate --n 200 --source psid
    python -m resumegen.generate --n 250 --source table3 --backend openai

Outputs into resumegen/out/:
    personas.csv    one row per person, all traits + latent ability
    careers.jsonl   full career objects (every position, every salary)
    resumes.csv     one row per (person, snapshot year, identity, view)
    resumes.md      a readable sample for humans

Runs with no API key by default. `--backend openai` swaps in the paper's
gpt-4o-2024-08-06 generation prompt; everything else is unchanged, so the two
are directly comparable.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from resumegen.traits import (sample_table3, personas_from_psid, assign_identity,
                              count_markers, GROUPS)
from resumegen.career import PostingPool, build_career
from resumegen import render as R

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")

ENG = re.compile(r"engineer|developer|programmer|architect|\bsre\b|devops|"
                 r"software|data scien|machine learning|\bsde\b", re.I)


def load_jobs(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        print(f"  ! no postings file at {path} — using fallback salary bands")
        return pd.DataFrame()
    j = pd.read_csv(path, low_memory=False)
    n0 = len(j)
    j = j[(j["avg_amount"] >= 40_000) & (j["avg_amount"] <= 400_000)]
    j = j.dropna(subset=["title", "company", "avg_amount"])
    j = j[j["title"].fillna("").str.contains(ENG)]
    print(f"  postings: {n0} -> {len(j)} after salary, company and title filters")
    return j.reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--source", choices=["psid", "table3"], default="psid")
    ap.add_argument("--backend", choices=["offline", "openai"], default="offline")
    ap.add_argument("--as-of", type=int, default=2021)
    ap.add_argument("--snapshots", type=int, nargs="*", default=None,
                    help="extra years to render each career at (default: as-of only)")
    ap.add_argument("--identities", action="store_true",
                    help="render each resume under all four name groups")
    ap.add_argument("--views", nargs="*", default=["full"], choices=list(R.VIEWS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--psid", default="data/psid_small.csv")
    ap.add_argument("--jobs", default="data/swe_jobs_small.csv")
    ap.add_argument("--model", default="gpt-4o-2024-08-06")
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    print(f"backend={args.backend}  source={args.source}  n={args.n}")

    jobs = load_jobs(args.jobs)
    pool = PostingPool(jobs)
    if not pool.empty:
        print("  postings per level:",
              {k: len(v) for k, v in sorted(pool.by_level.items())})

    # ---- personas --------------------------------------------------------
    if args.source == "psid":
        psid = pd.read_csv(args.psid, low_memory=False)
        personas = personas_from_psid(psid, n=args.n, seed=args.seed,
                                      as_of=args.as_of)
        print(f"  personas: {len(personas)} from PSID "
              f"(joint distribution preserved, ages 25-44)")
    else:
        personas = sample_table3(args.n, seed=args.seed)
        print(f"  personas: {len(personas)} sampled from Table 3 "
              f"(independent traits, paper-faithful)")

    # ---- careers ---------------------------------------------------------
    careers = [build_career(p, pool, as_of=args.as_of, seed=args.seed)
               for p in personas]
    n_pos = sum(len(c.positions) for c in careers)
    print(f"  careers: {len(careers)}, {n_pos} positions, "
          f"{n_pos / len(careers):.1f} per person")

    client = None
    if args.backend == "openai":
        from dotenv import load_dotenv
        from openai import OpenAI
        load_dotenv()
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            print("  ! OPENAI_API_KEY not set — put it in .env. Aborting.")
            sys.exit(1)
        client = OpenAI(api_key=key)

    # ---- render ----------------------------------------------------------
    years = sorted(set([args.as_of] + list(args.snapshots or [])))
    groups = GROUPS if args.identities else [None]
    rows, n_calls = [], 0

    for persona, career in zip(personas, careers):
        for year in years:
            if year < career.education_year + 1:
                continue
            for view in args.views:
                if args.backend == "openai":
                    body = R.render_openai(career, persona, year, client,
                                           view=view, model=args.model)
                    n_calls += 1
                else:
                    body = R.render_offline(career, persona, year, view=view)

                left = R.leftover_placeholders(body)
                snap = career.as_of(year)
                for g in groups:
                    if g is None:
                        text, name, email = body, None, None
                    else:
                        ip = assign_identity(persona, g, seed=args.seed)
                        text = R.insert_identity(body, ip)
                        name, email = ip.name, ip.email
                    rows.append({
                        "persona_id": persona.persona_id,
                        "source": persona.source,
                        "year": year,
                        "view": view,
                        "group": g,
                        "name": name,
                        "email": email,
                        "n_positions": len([p for p in snap.positions
                                            if p.employment_type != "gap"]),
                        "current_title": (snap.current().title if snap.current() else None),
                        "current_company": (snap.current().company if snap.current() else None),
                        "current_salary": snap.current_salary(),
                        "level": (snap.current().level if snap.current() else None),
                        "years_experience": snap.years_experience(year),
                        "n_promotions": snap.n_promotions(),
                        "salary_multiplier": career.salary_multiplier,
                        "latent_ability": round(persona.latent_ability, 4),
                        "chars": len(text),
                        "leftover_placeholders": ";".join(left),
                        "resume": text,
                        **count_markers(text),
                    })

    # ---- write -----------------------------------------------------------
    pdf = pd.DataFrame([p.to_dict() for p in personas])
    pdf.to_csv(os.path.join(OUT, "personas.csv"), index=False)

    with open(os.path.join(OUT, "careers.jsonl"), "w") as f:
        for c in careers:
            f.write(json.dumps(c.to_dict()) + "\n")

    rdf = pd.DataFrame(rows)
    rdf.to_csv(os.path.join(OUT, "resumes.csv"), index=False)

    # readable sample
    sample = rdf.sort_values("current_salary").iloc[
        [0, len(rdf) // 2, len(rdf) - 1]]
    with open(os.path.join(OUT, "resumes.md"), "w") as f:
        f.write(f"# Sample resumes ({args.backend}, {args.source}, {args.as_of})\n\n")
        for label, (_, r) in zip(["Lowest paid", "Median", "Highest paid"],
                                 sample.iterrows()):
            f.write(f"## {label} — persona {r.persona_id}, "
                    f"${int(r.current_salary):,}, level {r.level}, "
                    f"{r.years_experience} yrs exp\n\n```\n{r.resume}\n```\n\n")

    print(f"\nwrote {len(rdf)} resumes to resumegen/out/")
    if n_calls:
        print(f"  {n_calls} API calls")
    bad = rdf[rdf.leftover_placeholders != ""]
    print(f"  leftover placeholders: {len(bad)} rows")
    print(f"  salary: median ${rdf.current_salary.median():,.0f}, "
          f"p10 ${rdf.current_salary.quantile(.1):,.0f}, "
          f"p90 ${rdf.current_salary.quantile(.9):,.0f}")
    return rdf


if __name__ == "__main__":
    main()
