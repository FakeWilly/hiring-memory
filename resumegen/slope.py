"""
The question: "I would expect that the wage difference would increase over
time, in the standard case. But this depends on salary trajectories in
general. Do higher paying ones have a greater slope?"

Three parts, all offline:

  1. In the shipped model, does pay predict later growth?  (It does not, by
     construction: growth depends on LEVEL, via wage_level_bonus, not on pay.
     So the percentage gap between twins persists but does not grow.)
  2. The gap DOES grow in dollars even when it is flat in percent -- 12% of a
     bigger number is a bigger number. Report both, because "does the gap
     increase" has a different answer depending on the unit.
  3. Make slope depend on pay (wage_salary_elasticity > 0) and show how much
     of that it takes for the percentage gap to widen.  Then check the
     direction of the effect in the PSID panel we have -- with the caveats
     that belong on an 88-person file.

    python -m resumegen.slope --pairs 400 --years 20
"""

from __future__ import annotations

import argparse
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from resumegen.traits import personas_from_psid
from resumegen.career import PostingPool
from resumegen.twins import make_pair
from resumegen.generate import load_jobs
from resumegen.divergence import run_pair, DEFAULT_CFG

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")


def run_arm(pairs, pool, years, seed, cfg):
    recs = []
    for i, pair in enumerate(pairs):
        recs += run_pair(pair, pool, years, seed, cfg,
                         winner="A" if i % 2 == 0 else "B")
    return pd.DataFrame(recs)


def gaps(df, years):
    """Per-pair winner-minus-loser gap at t=0 and t=years, in log and dollars."""
    out = {}
    for t in (0, years):
        s = df[df.t == t]
        w = s[s.who == s.winner].set_index("pair_id")["salary"]
        l = s[s.who != s.winner].set_index("pair_id")["salary"]
        out[t] = {"log": np.log(w / l), "usd": w - l}
    return out


def slope_by_start(df, years):
    """Annual log growth over the horizon vs starting log salary, per person."""
    s0 = df[df.t == 0].set_index(["pair_id", "who"])["salary"]
    s1 = df[df.t == years].set_index(["pair_id", "who"])["salary"]
    g = (np.log(s1) - np.log(s0)) / years
    l0 = np.log(s0)
    q = pd.qcut(l0, 5, labels=["Q1 low", "Q2", "Q3", "Q4", "Q5 high"])
    by_q = g.groupby(q).mean()
    return float(l0.corr(g)), by_q


def psid_check(path):
    """Does higher pay predict steeper later growth in the PSID file we have?

    Head's labour income, individuals aged 25-44 at the window start, heads
    in both years, income > $5k in both. Two versions: naive (rank on the
    same year growth is measured from -- biased toward mean reversion), and
    split-sample (rank on income two years EARLIER, so a transitory dip in
    the base year cannot manufacture a negative slope).
    """
    d = pd.read_csv(path, low_memory=False)

    def yr(s):
        y = int(s)
        return 1900 + y if y >= 60 else 2000 + y

    def cols(pat):
        return {yr(re.search(r"(\d+)$", c).group(1)): c
                for c in d.columns if re.match(pat, c)}

    inc, age, rel = (cols(r"HEAD INCOME \d+$"),
                     cols(r"AGE OF INDIVIDUAL \d+$"),
                     cols(r"RELATIONSHIP TO HEAD DETAILED \d+$"))
    num = lambda c: pd.to_numeric(d[c], errors="coerce")

    naive, split = [], []
    for y0 in range(1997, 2010, 2):
        y1, yb = y0 + 10, y0 - 2
        if y1 not in inc:
            continue
        a, i0, i1 = d[age[y0]], num(inc[y0]), num(inc[y1])
        base = ((d[rel[y0]] == "head") & (d[rel[y1]] == "head")
                & a.between(25, 44) & (i0 > 5000) & (i1 > 5000))
        g = (np.log(i1) - np.log(i0)) / 10
        for idx in d.index[base]:
            naive.append((idx, np.log(i0[idx]), g[idx]))
        if yb in inc:
            ib = num(inc[yb])
            m = base & (d[rel[yb]] == "head") & (ib > 5000)
            for idx in d.index[m]:
                split.append((idx, np.log(ib[idx]), g[idx]))

    res = {}
    for name, rows in (("naive", naive), ("split-sample", split)):
        p = pd.DataFrame(rows, columns=["pid", "l0", "g"])
        p["tercile"] = pd.qcut(p.l0, 3, labels=["low", "mid", "high"])
        res[name] = {"windows": len(p), "people": p.pid.nunique(),
                     "corr": p.l0.corr(p.g),
                     "by_tercile": p.groupby("tercile", observed=True).g.mean()}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=int, default=400)
    ap.add_argument("--years", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--psid", default="data/psid_small.csv")
    ap.add_argument("--jobs", default="data/swe_jobs_small.csv")
    args = ap.parse_args()
    Y = args.years

    pool = PostingPool(load_jobs(args.jobs))
    psid = pd.read_csv(args.psid, low_memory=False)
    bases = personas_from_psid(psid, n=args.pairs, seed=args.seed)
    pairs = [make_pair(b, pool, i, seed=args.seed) for i, b in enumerate(bases)]

    # ---- 1. slope vs starting pay in the model as shipped ----------------
    base = run_arm(pairs, pool, Y, args.seed, DEFAULT_CFG)
    corr, by_q = slope_by_start(base, Y)
    print("1. In the model as shipped: annual log-salary growth by starting-pay quintile")
    print(by_q.round(4).to_string())
    print(f"   corr(log starting pay, {Y}-yr slope) = {corr:+.3f}")
    print("   -> growth is tied to LEVEL (wage_level_bonus), not to pay; the loser")
    print("      catches up in level, so the percentage gap holds but does not widen.\n")

    # ---- 2. percent gap vs dollar gap ------------------------------------
    g = gaps(base, Y)
    print("2. The same baseline run, gap between twins in two units")
    print(f"   t=0   {np.expm1(g[0]['log'].mean())*100:5.1f}%   "
          f"${g[0]['usd'].mean():>9,.0f}")
    print(f"   t={Y:<3} {np.expm1(g[Y]['log'].mean())*100:5.1f}%   "
          f"${g[Y]['usd'].mean():>9,.0f}   "
          f"(dollar gap x{g[Y]['usd'].mean()/g[0]['usd'].mean():.2f})")
    print("   -> flat in percent, growing in dollars. 'Does the gap increase'")
    print("      needs a unit before it has an answer.\n")

    # ---- 3. let pay buy slope ---------------------------------------------
    print("3. Give pay itself a slope: growth += elasticity * log(pay / Engineer band)")
    rows = []
    for el in (0.0, 0.01, 0.02, 0.03, 0.05):
        cfg = {**DEFAULT_CFG, "wage_salary_elasticity": el}
        df = run_arm(pairs, pool, Y, args.seed, cfg)
        gg = gaps(df, Y)
        c, _ = slope_by_start(df, Y)
        rows.append({
            "elasticity": el,
            "corr(pay, slope)": c,
            "gap t0 %": np.expm1(gg[0]["log"].mean()) * 100,
            f"gap t{Y} %": np.expm1(gg[Y]["log"].mean()) * 100,
            "amplification": gg[Y]["log"].mean() / gg[0]["log"].mean(),
            "winner ahead %": (gg[Y]["log"] > 0).mean() * 100,
            f"dollar gap t{Y}": gg[Y]["usd"].mean(),
        })
    res = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(res.round(3).to_string(index=False))
    res.to_csv(os.path.join(OUT, "slope_elasticity.csv"), index=False)
    print("   -> the percentage gap widens once, and only once, pay predicts")
    print("      growth. That is the parameter the expectation lives in.\n")

    # ---- 4. what does the PSID file say about that parameter? ------------
    print("4. Direction check in the PSID file (88 people; heads aged 25-44, 10-yr windows)")
    pr = psid_check(args.psid)
    for name, r in pr.items():
        print(f"   {name:13s} n={r['windows']} windows / {r['people']} people   "
              f"corr(log pay, later growth) = {r['corr']:+.3f}   "
              + "  ".join(f"{k}: {v*100:.1f}%/yr" for k, v in r["by_tercile"].items()))
    print("   -> in this panel higher earners grow SLOWER, even after ranking on")
    print("      income two years earlier to blunt regression to the mean. Nominal,")
    print("      total head income, tiny n -- a direction, not an estimate. It does")
    print("      not support putting a positive elasticity in by default.")


if __name__ == "__main__":
    main()
