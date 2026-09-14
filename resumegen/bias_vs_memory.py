"""
Bias sets the direction. Memory sets the magnitude.

The symmetry argument: matched twins are exchangeable -- statistically
identical in every merit-relevant respect. So the process that unfolds after
the hiring decision is the same process whichever twin was picked. A selection
rule, however biased, chooses WHO gets the winner's path; it cannot change what
the winner's path IS. Therefore:

    magnitude of divergence between equals   <- market memory only
    which group the divergence favours       <- the selection rule only

This script verifies the code respects that symmetry, with zero API calls.
Four arms, crossing the selection rule with market memory:

    fair coin x memory on      the earlier baseline
    always-A  x memory on      a MAXIMALLY biased decision-maker
    fair coin x memory off
    always-A  x memory off

Predictions, stated before running:
    winner-vs-loser gap:  identical across selection rules (symmetry),
                          ~12.5% with memory, ~0 without
    A-vs-B group gap:     ~0 under the fair coin, equal to the full
                          winner gap under always-A with memory,
                          and ~0 under always-A WITHOUT memory --
                          maximal bias, no lasting group disparity

    python -m resumegen.bias_vs_memory --pairs 400 --years 20
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from resumegen.traits import personas_from_psid
from resumegen.career import PostingPool
from resumegen.twins import make_pair, verify_pair
from resumegen.generate import load_jobs
from resumegen.divergence import run_pair, DEFAULT_CFG

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")

NO_MEMORY = {**DEFAULT_CFG, "anchoring": False, "ladder": False,
             "wage": False, "job_is_step_up": False}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=int, default=400)
    ap.add_argument("--years", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--psid", default="data/psid_small.csv")
    ap.add_argument("--jobs", default="data/swe_jobs_small.csv")
    args = ap.parse_args()

    pool = PostingPool(load_jobs(args.jobs))
    psid = pd.read_csv(args.psid, low_memory=False)
    bases = personas_from_psid(psid, n=args.pairs, seed=args.seed)
    pairs = [make_pair(b, pool, i, seed=args.seed) for i, b in enumerate(bases)]
    n_ok = sum(verify_pair(p)["equivalent"] for p in pairs)
    print(f"{len(pairs)} matched pairs; {n_ok} pass the equivalence audit\n")

    arms = [
        ("fair coin",        "memory on",  DEFAULT_CFG, "alternate"),
        ("always picks A",   "memory on",  DEFAULT_CFG, "A"),
        ("fair coin",        "memory off", NO_MEMORY,   "alternate"),
        ("always picks A",   "memory off", NO_MEMORY,   "A"),
    ]

    rows = []
    for rule_name, mem_name, cfg, rule in arms:
        recs = []
        for i, pair in enumerate(pairs):
            winner = "A" if (rule == "A" or i % 2 == 0) else "B"
            recs += run_pair(pair, pool, args.years, args.seed, cfg,
                             winner=winner)
        df = pd.DataFrame(recs)
        last = df[df.t == args.years]

        w = last[last.who == last.winner].set_index("pair_id")["salary"]
        l = last[last.who != last.winner].set_index("pair_id")["salary"]
        wl = np.log(w / l)

        a = last[last.who == "A"].set_index("pair_id")["salary"]
        b = last[last.who == "B"].set_index("pair_id")["salary"]
        ab = np.log(a / b)

        rows.append({
            "selection rule": rule_name,
            "market memory": mem_name,
            "winner-vs-loser gap %": np.expm1(wl.mean()) * 100,
            "winner ahead %": (wl > 0).mean() * 100,
            "A-vs-B group gap %": np.expm1(ab.mean()) * 100,
            "A ahead %": (ab > 0).mean() * 100,
            "sd of gap": wl.std(ddof=1),
        })

    res = pd.DataFrame(rows).round(2)
    res.to_csv(os.path.join(OUT, "bias_vs_memory.csv"), index=False)
    pd.set_option("display.width", 200)
    print(res.to_string(index=False))

    print("""
Reading:
  Column 3 (magnitude of individual divergence) depends only on the MEMORY
  column, never on the selection rule -- as the exchangeability of the twins
  requires. Column 5 (which group the divergence favours) depends only on the
  RULE column, and even maximal bias leaves no lasting group gap once memory
  is off. Bias decides who wins; memory decides whether winning matters.""")


if __name__ == "__main__":
    main()
