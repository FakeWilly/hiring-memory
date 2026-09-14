"""
The experiment: two equal candidates, one job, twenty years.

    Year 0   twins A and B, matched on every merit dimension, both apply
             for the same posting. One is chosen. By construction the choice
             is arbitrary -- there is nothing to choose on.
    Years 1..20
             both continue to work and to apply for jobs. The ONLY difference
             between them is which one got that first job.
    Measure  how far apart they are, and through which channel.

The point is that the branch is exogenous by construction. In an observational
setting you can never separate "he was hired because he was better" from "he is
better off because he was hired". Here the first is ruled out by design, so
whatever gap opens up is the causal effect of one arbitrary decision.

Two things are separated deliberately:

  ADVANTAGE   the winner's own path relative to the loser's
  MECHANISM   which of the three channels produced it --
                anchoring  (a higher current salary makes the next job better)
                ladder     (a higher title makes the next job better)
                wage       (a higher salary compounds on its own)

Run with `--channels` to switch each off in turn. A gap that survives all three
being disabled is not cumulative advantage; it is a bug.

    python -m resumegen.divergence --pairs 100 --years 20
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from resumegen.traits import personas_from_psid
from resumegen.career import PostingPool, LEVEL_MIN_YOE, MAX_LEVEL
from resumegen.twins import make_pair, verify_pair
from resumegen.generate import load_jobs

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")


def _rng(seed, pair_id, who, t):
    """Deterministic per-(pair, person, period) stream.

    An earlier version keyed this on Python's hash() of a tuple containing a
    string. Python randomises string hashing per process, so every run drew
    different path noise and the headline moved a couple of points between
    invocations of the same command. SeedSequence is stable across processes.
    """
    return np.random.default_rng(
        np.random.SeedSequence([int(seed), int(pair_id), 0 if who == "A" else 1,
                                int(t)]))


def step_career(state: dict, pool: PostingPool, rng, year: int, cfg: dict) -> dict:
    """One period for one person.

    Each channel is a form of MEMORY -- a way today's standing shapes tomorrow's.
    Switch them all off and a job move redraws you from the population band,
    erasing your history. That is the null: the initial advantage persists only
    as long as you stay put, and washes out as you move.

      anchoring  your current salary is a floor on your next offer
      ladder     your current level determines which band you are drawn from
      wage       staying put compounds faster the higher your level
    """
    level, salary = state["level"], state["salary"]
    moved = rng.random() < cfg["p_move"]

    if moved:
        if cfg["ladder"]:
            reach = int(np.clip(level + (1 if rng.random() < 0.40 else 0),
                                0, MAX_LEVEL))
        else:
            # no ladder memory: drawn from the middle of the market
            reach = int(np.clip(rng.integers(1, 3), 0, MAX_LEVEL))
        offer = pool.band(reach) * state["multiplier"] * rng.normal(1.0, 0.06)
        if cfg["anchoring"]:
            # your own pay is a floor -- the single most familiar way a past
            # outcome reaches into the next negotiation
            offer = max(offer, salary * rng.normal(cfg.get("anchor_floor", 1.06), 0.03))
        offer = min(offer, salary * 1.40)
        salary, level = float(offer), reach
    else:
        growth = cfg["wage_base"] + (cfg["wage_level_bonus"] * level
                                     if cfg["wage"] else 0.0)
        # Optional: do higher-paid trajectories have a steeper slope?
        # Default 0.0 -- in the shipped model growth depends on LEVEL, not
        # on pay. Set >0 to let pay itself buy growth: +elasticity per log
        # point above the reference salary (an Engineer-band median).
        el = cfg.get("wage_salary_elasticity", 0.0)
        if el:
            growth += el * np.log(salary / cfg.get("ref_salary", pool.band(1)))
        salary *= (1 + rng.normal(growth, 0.030))

    state["yoe"] += 1
    if level < MAX_LEVEL and state["yoe"] >= LEVEL_MIN_YOE[level + 1]:
        if rng.random() < (0.10 if cfg["ladder"] else 0.06):
            level += 1
            if cfg["ladder"]:
                salary *= 1.05

    state["level"], state["salary"] = level, float(salary)
    state["moves"] += int(moved)
    return state


def run_pair(pair, pool, years: int, seed: int, cfg: dict,
             winner: str = "A") -> dict:
    """Run one twin pair forward. `winner` took the year-0 job."""
    rows = []
    states = {}
    for who, (persona, career) in (("A", (pair.a_persona, pair.a_career)),
                                   ("B", (pair.b_persona, pair.b_career))):
        cur = career.current()
        base = float(cur.salary)
        # the winner starts the clock in the new job; the loser stays put
        if who == winner:
            start = base * cfg["job_premium"]
            lvl = min(cur.level + (1 if cfg["job_is_step_up"] else 0), MAX_LEVEL)
        else:
            start = base
            lvl = cur.level
        states[who] = {"salary": start, "level": lvl,
                       "yoe": career.years_experience(2021),
                       "multiplier": career.salary_multiplier,
                       "moves": 0}

    for t in range(years + 1):
        for who in ("A", "B"):
            rows.append({"pair_id": pair.pair_id, "who": who, "t": t,
                         "winner": winner,
                         "salary": states[who]["salary"],
                         "level": states[who]["level"],
                         "moves": states[who]["moves"]})
        if t == years:
            break
        for who in ("A", "B"):
            states[who] = step_career(states[who], pool,
                                      _rng(seed, pair.pair_id, who, t),
                                      2021 + t, cfg)

    return rows


DEFAULT_CFG = dict(anchoring=True, ladder=True, wage=True,
                   wage_base=0.021, wage_level_bonus=0.006,
                   wage_salary_elasticity=0.0,
                   p_move=0.30, job_premium=1.12, job_is_step_up=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=int, default=100)
    ap.add_argument("--years", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--psid", default="data/psid_small.csv")
    ap.add_argument("--jobs", default="data/swe_jobs_small.csv")
    ap.add_argument("--channels", action="store_true",
                    help="also run with each channel disabled in turn")
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    jobs = load_jobs(args.jobs)
    pool = PostingPool(jobs)
    psid = pd.read_csv(args.psid, low_memory=False)
    bases = personas_from_psid(psid, n=args.pairs, seed=args.seed)

    pairs = [make_pair(b, pool, i, seed=args.seed) for i, b in enumerate(bases)]

    checks = [verify_pair(p) for p in pairs]
    ok = sum(c["equivalent"] for c in checks)
    print(f"built {len(pairs)} matched pairs; {ok} pass the equivalence audit")
    if ok < len(pairs):
        bad = [c for c in checks if not c["equivalent"]][:3]
        for c in bad:
            print(f"   pair {c['pair_id']} unmatched on: {c['unmatched']}")

    arms = {"baseline": dict(DEFAULT_CFG)}
    if args.channels:
        arms["no anchoring"] = {**DEFAULT_CFG, "anchoring": False}
        arms["no ladder"] = {**DEFAULT_CFG, "ladder": False}
        arms["no wage compounding"] = {**DEFAULT_CFG, "wage": False}
        arms["all three off"] = {**DEFAULT_CFG, "anchoring": False,
                                 "ladder": False, "wage": False,
                                 "job_is_step_up": False}

    all_rows, summary = [], []
    for arm, cfg in arms.items():
        rows = []
        for i, pair in enumerate(pairs):
            # alternate who wins, so the result cannot be an artefact of
            # twin A being systematically different from twin B
            rows += run_pair(pair, pool, args.years, args.seed, cfg,
                             winner="A" if i % 2 == 0 else "B")
        df = pd.DataFrame(rows)
        df["arm"] = arm
        all_rows.append(df)

        last = df[df.t == args.years]
        w = last[last.who == last.winner].set_index("pair_id")["salary"]
        l = last[last.who != last.winner].set_index("pair_id")["salary"]
        gap = np.log(w / l)
        first = df[df.t == 0]
        w0 = first[first.who == first.winner].set_index("pair_id")["salary"]
        l0 = first[first.who != first.winner].set_index("pair_id")["salary"]
        gap0 = np.log(w0 / l0)

        summary.append({
            "arm": arm,
            "gap_t0_log": gap0.mean(),
            "gap_t0_pct": np.expm1(gap0.mean()) * 100,
            f"gap_t{args.years}_log": gap.mean(),
            f"gap_t{args.years}_pct": np.expm1(gap.mean()) * 100,
            "amplification": gap.mean() / gap0.mean() if gap0.mean() else np.nan,
            "winner_ahead_pct": (gap > 0).mean() * 100,
            "sd_gap": gap.std(ddof=1),
            "t_stat": gap.mean() / (gap.std(ddof=1) / np.sqrt(len(gap))),
        })

    out = pd.concat(all_rows, ignore_index=True)
    out.to_csv(os.path.join(OUT, "divergence_paths.csv"), index=False)
    s = pd.DataFrame(summary)
    s.to_csv(os.path.join(OUT, "divergence_summary.csv"), index=False)

    pd.set_option("display.width", 200)
    print()
    print(s.round(3).to_string(index=False))
    print()
    b = s.iloc[0]
    print(f"At the branch point the winner is {b['gap_t0_pct']:.1f}% ahead.")
    print(f"After {args.years} years that gap is "
          f"{b[f'gap_t{args.years}_pct']:.1f}% "
          f"({b['amplification']:.1f}x the initial advantage), and the winner "
          f"is ahead in {b['winner_ahead_pct']:.0f}% of pairs.")
    return s


if __name__ == "__main__":
    main()
