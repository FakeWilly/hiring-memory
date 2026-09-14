"""
Quality checks on the generated database.

Every synthetic dataset is wrong in ways nobody notices until a reviewer does.
These are the checks worth running before showing anyone: does the salary
distribution resemble the market it was drawn from, do careers progress the way
careers do, and is anything visibly impossible.

    python -m resumegen.qa
"""

from __future__ import annotations

import json
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from resumegen.career import LEVEL_TITLE

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")


def load():
    r = pd.read_csv(os.path.join(OUT, "resumes.csv"))
    careers = [json.loads(l) for l in open(os.path.join(OUT, "careers.jsonl"))]
    return r, careers


def main():
    r, careers = load()
    print("=" * 66)
    print(f"QA on {len(r)} resumes / {len(careers)} careers")
    print("=" * 66)

    # --- 1. salary realism -------------------------------------------------
    print("\n1. SALARY")
    jobs_path = "data/swe_jobs_small.csv"
    if os.path.exists(jobs_path):
        j = pd.read_csv(jobs_path, low_memory=False)
        j = j[(j.avg_amount >= 40_000) & (j.avg_amount <= 400_000)]
        print(f"   postings   p10 ${j.avg_amount.quantile(.1):>9,.0f}  "
              f"median ${j.avg_amount.median():>9,.0f}  "
              f"p90 ${j.avg_amount.quantile(.9):>9,.0f}")
    print(f"   generated  p10 ${r.current_salary.quantile(.1):>9,.0f}  "
          f"median ${r.current_salary.median():>9,.0f}  "
          f"p90 ${r.current_salary.quantile(.9):>9,.0f}")
    print(f"   sd of log salary: {np.log(r.current_salary).std():.3f}")

    # --- 2. seniority ------------------------------------------------------
    print("\n2. SENIORITY MIX")
    lv = r.level.value_counts().sort_index()
    for k, v in lv.items():
        print(f"   L{int(k)} {LEVEL_TITLE[int(k)]:<30} {v:>4}  ({v/len(r):>5.1%})")
    print(f"   median years experience: {r.years_experience.median():.0f}")
    print(f"   median promotions:       {r.n_promotions.median():.0f}")

    # --- 3. within-career salary paths ------------------------------------
    print("\n3. CAREER PATHS")
    cuts, jumps, spans, drops_after_gap = [], [], [], 0
    for c in careers:
        pos = [p for p in c["positions"] if p["employment_type"] != "gap"]
        gapped = {i for i, p in enumerate(c["positions"])
                  if p["employment_type"] == "gap"}
        sals = [p["salary"] for p in pos]
        spans.append(len(pos))
        for a, b in zip(sals, sals[1:]):
            ch = np.log(b / a)
            jumps.append(ch)
            if ch < 0:
                cuts.append(ch)
        if gapped:
            drops_after_gap += 1
    jumps = np.array(jumps)
    print(f"   positions per career: mean {np.mean(spans):.1f}, "
          f"max {max(spans)}")
    print(f"   job-to-job log salary change: mean {jumps.mean():+.3f}, "
          f"sd {jumps.std(ddof=1):.3f}")
    print(f"   fraction that are pay cuts: {(jumps < 0).mean():.1%}   "
          f"(all should follow a career break)")
    print(f"   careers containing a break: {drops_after_gap} "
          f"({drops_after_gap/len(careers):.0%})")

    # --- 4. merit vs luck --------------------------------------------------
    print("\n4. HOW MUCH IS ABILITY?")
    latest = r[r.year == r.year.max()]
    sub = latest.drop_duplicates("persona_id")
    for col in ("current_salary", "level", "n_promotions"):
        rho = np.corrcoef(sub.latent_ability, sub[col])[0, 1]
        print(f"   corr(latent_ability, {col:<16}) = {rho:+.3f}")
    print("   (deliberately modest: ability tilts promotion odds and pay band,")
    print("    everything else is tenure and luck)")

    # --- 5. impossible people ---------------------------------------------
    print("\n5. SANITY")
    problems = []
    if (r.years_experience < 0).any():
        problems.append("negative experience")
    bad_exp = r[(r.level >= 3) & (r.years_experience < 8)]
    if len(bad_exp):
        problems.append(f"{len(bad_exp)} staff+ engineers with <8 yrs experience")
    if r.leftover_placeholders.fillna("").ne("").any():
        n = r.leftover_placeholders.fillna("").ne("").sum()
        problems.append(f"{n} resumes with unfilled placeholders")
    dup = r.resume.duplicated().sum()
    if dup:
        problems.append(f"{dup} duplicate resume texts")
    lens = r.chars
    print(f"   resume length: median {lens.median():.0f} chars, "
          f"min {lens.min()}, max {lens.max()}")
    junk = r.current_company.fillna("").str.contains(
        r"intern|co-?op|new grad|staffing|careers page", case=False, regex=True)
    if junk.any():
        problems.append(f"{junk.sum()} resumes with job-board cruft in the employer")
    print("   " + ("no problems found" if not problems
                   else "PROBLEMS: " + "; ".join(problems)))

    # --- 6. length vs seniority (the mediator warning) --------------------
    print("\n6. LENGTH IS A MEDIATOR, NOT A NUISANCE")
    rho = np.corrcoef(r.chars, r.current_salary)[0, 1]
    print(f"   corr(resume length, current salary) = {rho:+.3f}")
    print("   Resumes get longer as careers progress, so length is CAUSED by")
    print("   success. LLM judges favour longer inputs, so any measured")
    print("   advantage partly runs through length. Cannot be controlled away")
    print("   without deleting part of the effect.")
    return r


if __name__ == "__main__":
    main()
