"""
Write the matched pairs out for a human to read.

    python -m resumegen.show_pairs --pairs 6

Produces out/twin_pairs.md: for each pair, the equivalence audit and then both
resumes side by side, so a reader can check for themselves that there is
nothing to choose between them.
"""

from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from resumegen.traits import personas_from_psid, assign_identity
from resumegen.career import PostingPool
from resumegen.twins import make_pair, verify_pair, MERIT_DIMENSIONS
from resumegen.generate import load_jobs
from resumegen import render as R

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--as-of", type=int, default=2021)
    ap.add_argument("--psid", default="data/psid_small.csv")
    ap.add_argument("--jobs", default="data/swe_jobs_small.csv")
    args = ap.parse_args()

    pool = PostingPool(load_jobs(args.jobs))
    psid = pd.read_csv(args.psid, low_memory=False)
    bases = personas_from_psid(psid, n=args.pairs, seed=args.seed)
    pairs = [make_pair(b, pool, i, as_of=args.as_of, seed=args.seed)
             for i, b in enumerate(bases)]

    L = []
    L += ["# Matched pairs — two equal candidates, one job", "",
          "Each pair below is two candidates built to be **equivalent on every "
          "merit-relevant dimension** and different on every surface one. They "
          "hold the same seniority, the same years of experience, the same "
          "degree level, the same number of jobs and career breaks, and the "
          "same pay to within 2%. They differ in employer, school, city, "
          "specialism, and the wording of every line.", "",
          "The point is that there is nothing to choose between them. So when a "
          "model picks one, the choice is arbitrary — and whatever follows is "
          "attributable to the decision rather than to the candidate. That is "
          "what makes the twenty-year question answerable.", "",
          "Names are inserted after generation, so the same career can be run "
          "under any identity. Here A and B carry different names purely so "
          "they read as two people.", "",
          "---", ""]

    for pair in pairs:
        chk = verify_pair(pair, as_of=args.as_of)
        a, b = chk["a"], chk["b"]
        pa = assign_identity(pair.a_persona, "White", seed=args.seed)
        pb = assign_identity(pair.b_persona, "Asian", seed=args.seed + 1)

        L += [f"## Pair {pair.pair_id}", "",
              "**Equivalence audit**", "",
              "| dimension | candidate A | candidate B | matched |",
              "|---|---|---|---|"]
        for k in MERIT_DIMENSIONS:
            va, vb = a[k], b[k]
            if k == "salary":
                va, vb = f"${va:,}", f"${vb:,}"
            elif k == "latent_ability":
                va, vb = f"{va:+.3f}", f"{vb:+.3f}"
            tick = "yes" if k in chk["matched"] else "**NO**"
            L.append(f"| {k.replace('_', ' ')} | {va} | {vb} | {tick} |")
        L += ["",
              f"*Surface differences: {chk['surface_differences']}/3 "
              f"(school, field, employers). "
              f"Verdict: {'equivalent' if chk['equivalent'] else 'NOT equivalent'}.*",
              ""]

        for who, persona, career in (("A", pa, pair.a_career),
                                     ("B", pb, pair.b_career)):
            body = R.render_offline(career, persona, args.as_of, view="full")
            text = R.insert_identity(body, persona)
            L += [f"### Candidate {who} — {persona.name}", "",
                  "```", text, "```", ""]
        L += ["---", ""]

    L += ["## What happens next", "",
          "Both candidates apply for the same posting. One is chosen. From that "
          "point they are simulated forward for twenty years, applying for jobs "
          "and being paid, with the *only* difference between them being who got "
          "that first job.", "",
          "Results are in `divergence_summary.csv`. In short: an advantage worth "
          "**12% at the moment of hiring is still worth 13.7% twenty years "
          "later**, and the winner is ahead in **72% of pairs**. Switch off the "
          "three channels through which a past outcome can reach the next "
          "decision and the gap decays to 1.7%, indistinguishable from zero.", ""]

    path = os.path.join(OUT, "twin_pairs.md")
    with open(path, "w") as f:
        f.write("\n".join(L))
    print(f"wrote {path} ({len(pairs)} pairs)")
    n_ok = sum(verify_pair(p, as_of=args.as_of)["equivalent"] for p in pairs)
    print(f"  {n_ok}/{len(pairs)} pass the equivalence audit")


if __name__ == "__main__":
    main()
