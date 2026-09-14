"""
Twenty years with the model in the loop.

The coin-flip simulation (divergence.py) asked what the MARKET does with one
arbitrary decision. This asks what happens when the decision-maker is an LLM
that reads both resumes every time the twins compete -- because then the
model's own memory (how much of a candidate's past it re-transmits) multiplies
the market's.

  Year 0        twins A and B apply for posting J0. The model chooses. The
                winner takes it on the same terms as in hiring.py and
                divergence.py: one level up, +12% pay. The loser gets the
                year's in-place raise. All three experiments share this
                branch point.
  Years 1..20   each year both careers advance (stay-put growth, occasional
                promotion in place), and each twin may move on their OWN
                with probability p_move -- an uncontested offer under the
                market's rules, exactly as in divergence.py. With probability
                p_event a CONTESTED posting appears at the level of the more
                senior twin (one step up if they are level and eligible);
                both apply with resumes rendered from their current careers;
                the model chooses; the winner takes the job.

  The coin control runs the identical code with a coin in the model's seat,
  so it should reproduce the divergence.py dynamics (~14% gap, ~70% winner
  ahead) up to the contested events; the LLM arms differ from it only in who
  wins the contests.

Every decision is made in BOTH presentation orders. If the two orders agree,
that candidate wins; if they disagree the model has no position-robust
preference and a coin decides. Position bias is therefore measured, not
inherited by the trajectory.

Cells cross what the model can SEE (view) with what the market REMEMBERS:
    view    full | no_salary | blind
    market  default (anchoring, ladder, compounding) | no_anchoring | no_memory
plus a `coin` decision rule run through the identical code path as the control.

Pre-registered comparison: re-selection rate and 20-year gap under
(default, full) vs (default, no_salary) vs coin.

    python -m resumegen.trajectory --model mock --pairs 20 --years 20
    python -m resumegen.trajectory --model gpt-4o-mini --pairs 50 --dry-run
"""

from __future__ import annotations

import argparse
import copy
import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from resumegen.career import (PostingPool, Position, Career, LEVEL_MIN_YOE, MAX_LEVEL)
from resumegen.render import render_offline
from resumegen.hiring import (load_pairs, anonymise, pair_specialism, _fresh_highlights,
                              YEAR0)
from resumegen.twins import pair_specialise
from resumegen import prompts as P
from resumegen.llm import make_client, guard_budget, BudgetExceeded

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")

MARKETS = {
    "default":      dict(anchoring=True,  ladder=True,  wage=True),
    "no_anchoring": dict(anchoring=False, ladder=True,  wage=True),
    "no_memory":    dict(anchoring=False, ladder=False, wage=False),
}
WAGE_BASE, WAGE_LEVEL_BONUS = 0.021, 0.006
P_MOVE = 0.20      # own, uncontested moves per person-year
P_EVENT = 0.20     # contested postings per pair-year  (0.20 + 0.20/2 = 0.30 moves/yr, as divergence.py)
JOB_PREMIUM = 1.12 # the year-0 branch point, shared with hiring.py and divergence.py


def _rng(*parts):
    return np.random.default_rng(np.random.SeedSequence([int(x) for x in parts]))


# ---------------------------------------------------------------------------

class LiveCareer:
    """A Career that can be advanced year by year and rendered at any point."""

    def __init__(self, career: Career, persona, specialism: str, pool: PostingPool,
                 specialise: bool = True):
        self.c = copy.deepcopy(career)
        self.persona, self.spec, self.pool = persona, specialism, pool
        self.specialise = specialise      # shared by both twins of a pair
        self.moves = 0

    @property
    def cur(self) -> Position:
        return self.c.current()

    @property
    def salary(self) -> int:
        return int(self.cur.salary)

    @property
    def level(self) -> int:
        return int(self.cur.level)

    def yoe(self, year: int) -> int:
        return self.c.years_experience(year)

    def stay(self, growth: float):
        self.cur.salary = int(round(self.cur.salary * (1 + growth), -2))

    def promote(self, rng, raise_pct: float):
        self.cur.level = min(self.cur.level + 1, MAX_LEVEL)
        self.cur.title = self.pool.title(self.cur.level, rng, self.spec, specialise=self.specialise)
        self.cur.is_promotion = True
        self.cur.salary = int(round(self.cur.salary * (1 + raise_pct), -2))

    def move(self, year: int, level: int, company: str, location: str, salary: float, rng):
        self.cur.year_end = year
        self.c.positions.append(Position(
            year_start=year, year_end=None,
            title=self.pool.title(level, rng, self.spec, specialise=self.specialise),
            company=company,
            level=level, salary=int(round(salary, -2)), location=location,
            highlights=_fresh_highlights(self.c, level, rng)))
        self.moves += 1

    def render(self, year: int, view: str) -> str:
        return anonymise(render_offline(self.c, self.persona, year, view))


# ---------------------------------------------------------------------------

def advance_year(live: LiveCareer, market: dict, rng, year: int):
    """Stay-put growth, or an uncontested move of one's own; then a chance of
    promotion in place. Mirrors divergence.step_career."""
    if rng.random() < P_MOVE:
        if market["ladder"]:
            eligible = live.level < MAX_LEVEL and live.yoe(year) >= LEVEL_MIN_YOE[live.level + 1]
            reach = live.level + (1 if (eligible and rng.random() < 0.40) else 0)
        else:
            reach = int(np.clip(rng.integers(1, 3), 0, MAX_LEVEL))
        live.move(year, reach, live.pool.company(rng), live.pool.location(rng),
                  offer_for(live, reach, market, rng), rng)
    else:
        growth = WAGE_BASE + (WAGE_LEVEL_BONUS * live.level if market["wage"] else 0.0)
        live.stay(rng.normal(growth, 0.030))
    if live.level < MAX_LEVEL and live.yoe(year) >= LEVEL_MIN_YOE[live.level + 1]:
        if rng.random() < (0.10 if market["ladder"] else 0.06):
            live.promote(rng, 0.05 if market["ladder"] else 0.0)


def offer_for(live: LiveCareer, level: int, market: dict, rng) -> float:
    offer = live.pool.band(level) * live.c.salary_multiplier * rng.normal(1.0, 0.06)
    if market["anchoring"]:
        offer = max(offer, live.salary * rng.normal(1.06, 0.03))
    return float(min(offer, live.salary * 1.40))


def posting_level(a: LiveCareer, b: LiveCareer, year: int) -> int:
    top = max(a.level, b.level)
    if a.level == b.level and top < MAX_LEVEL and \
            min(a.yoe(year), b.yoe(year)) >= LEVEL_MIN_YOE[top + 1]:
        return top + 1
    return top


def decide(client, prompt, posting: str, ra: str, rb: str, coin_rng, rule: str, tag: str):
    """Both orders. Agreement wins; disagreement -> coin. Returns a record."""
    rec = {"rule": rule}
    if rule == "coin":
        rec.update(choice_ab=None, choice_ba=None, p_first_ab=None, p_first_ba=None,
                   winner="A" if coin_rng.random() < 0.5 else "B", resolved="coin")
        return rec
    fmt = P.JSON_FORMAT if prompt.answer_mode == "json" else None
    picks = {}
    for order, (r1, r2) in (("ab", (ra, rb)), ("ba", (rb, ra))):
        rep = client.chat(prompt.messages(posting, r1, r2), max_tokens=prompt.max_tokens,
                          logprobs=(prompt.answer_mode == "direct"), response_format=fmt,
                          tag=tag)
        ch, status = (None, "dry") if client.dry_run else P.parse_choice(rep.content, prompt.answer_mode)
        rec[f"choice_{order}"] = ch
        rec[f"p_first_{order}"] = P.p_first_from_logprobs(rep.first_token_logprobs)
        if ch in (1, 2):
            # translate position into candidate
            picks[order] = ("A" if ch == 1 else "B") if order == "ab" else ("B" if ch == 1 else "A")
        else:
            picks[order] = None
    if picks["ab"] and picks["ab"] == picks["ba"]:
        rec.update(winner=picks["ab"], resolved="agree")
    else:
        rec.update(winner="A" if coin_rng.random() < 0.5 else "B",
                   resolved="disagree" if all(picks.values()) else "unparsed")
    return rec


# ---------------------------------------------------------------------------

def run_pair(pair, pool, client, prompt, years: int, seed: int, market_name: str,
             view: str, rule: str):
    market = MARKETS[market_name]
    spec, spz = pair_specialism(pair.pair_id), pair_specialise(pair.pair_id)
    live = {"A": LiveCareer(pair.a_career, pair.a_persona, spec, pool, spz),
            "B": LiveCareer(pair.b_career, pair.b_persona, spec, pool, spz)}
    cell = f"{rule}|{market_name}|{view}"
    paths, decisions = [], []
    first_winner = None

    def snapshot(t, year):
        for who, lc in live.items():
            paths.append({"cell": cell, "rule": rule, "market": market_name, "view": view,
                          "pair_id": pair.pair_id, "t": t, "year": year, "who": who,
                          "salary": lc.salary, "level": lc.level, "moves": lc.moves,
                          "first_winner": first_winner})

    def contest(t, year):
        nonlocal first_winner
        lvl = posting_level(live["A"], live["B"], year)
        prng = _rng(seed, pair.pair_id, t, 1)
        company, location = pool.company(prng), pool.location(prng)
        posting = P.posting_text(lvl, company, location, spec)
        ra, rb = live["A"].render(year, view), live["B"].render(year, view)
        shown = {"salary_A": live["A"].salary, "salary_B": live["B"].salary,
                 "level_A": live["A"].level, "level_B": live["B"].level}
        rec = decide(client, prompt, posting, ra, rb, _rng(seed, pair.pair_id, t, 2), rule,
                     tag=f"traj|{cell}")
        w = rec["winner"]
        l = "B" if w == "A" else "A"
        orng = _rng(seed, pair.pair_id, t, 3)
        if t == 0:
            # the shared branch point: +12% and the step up; the loser's year in place
            live[w].move(year, lvl, company, location, live[w].salary * JOB_PREMIUM, orng)
            live[l].stay(0.025)
        else:
            live[w].move(year, lvl, company, location, offer_for(live[w], lvl, market, orng), orng)
        if first_winner is None:
            first_winner = w
        decisions.append({"cell": cell, "rule": rule, "market": market_name, "view": view,
                          "pair_id": pair.pair_id, "t": t, "year": year, "level": lvl,
                          **shown, **rec, "first_winner": first_winner,
                          "reselected": None if t == 0 else int(w == first_winner)})

    contest(0, YEAR0)
    snapshot(0, YEAR0)
    for t in range(1, years + 1):
        year = YEAR0 + t
        for who in ("A", "B"):
            advance_year(live[who], market, _rng(seed, pair.pair_id, t, 4 if who == "A" else 5), year)
        if _rng(seed, pair.pair_id, t, 6).random() < P_EVENT:
            contest(t, year)
        snapshot(t, year)
    return paths, decisions


def summarise(paths: pd.DataFrame, decisions: pd.DataFrame, years: int) -> pd.DataFrame:
    rows = []
    for cell, sub in paths.groupby("cell", sort=False):
        last = sub[sub.t == years]
        w = last[last.who == last.first_winner].set_index("pair_id").salary
        l = last[last.who != last.first_winner].set_index("pair_id").salary
        gap = np.log(w / l)
        first = sub[sub.t == 0]
        w0 = first[first.who == first.first_winner].set_index("pair_id").salary
        l0 = first[first.who != first.first_winner].set_index("pair_id").salary
        gap0 = np.log(w0 / l0)
        d = decisions[decisions.cell == cell]
        later = d[d.t > 0]
        rows.append({
            "cell": cell,
            "pairs": len(gap),
            "gap_t0_pct": np.expm1(gap0.mean()) * 100,
            f"gap_t{years}_pct": np.expm1(gap.mean()) * 100,
            "winner_ahead_pct": (gap > 0).mean() * 100,
            "amplification": gap.mean() / gap0.mean() if gap0.mean() else np.nan,
            "sd_gap": gap.std(ddof=1),
            "contests_per_pair": len(d) / max(1, d.pair_id.nunique()),
            "reselect_rate": later.reselected.mean() if len(later) else np.nan,
            "A_wins_year0_pct": (d[d.t == 0].winner == "A").mean() * 100,
            "agree_rate": (d.resolved == "agree").mean() if "resolved" in d else np.nan,
            "pick_first_rate": pd.concat([(d.choice_ab == 1), (d.choice_ba == 1)]).mean()
                               if d.choice_ab.notna().any() else np.nan,
        })
    return pd.DataFrame(rows)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mock")
    ap.add_argument("--pairs", type=int, default=50)
    ap.add_argument("--years", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--prompt", default="neutral", choices=list(P.DECISION_PROMPTS))
    ap.add_argument("--cells", default="default:full,default:no_salary,no_anchoring:full,no_anchoring:no_salary",
                    help="market:view pairs for the LLM arms; coin controls are added automatically")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--budget", type=float, default=25.0)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--workers", type=int, default=8, help="pair trajectories in flight at once")
    ap.add_argument("--psid", default="data/psid_small.csv")
    ap.add_argument("--jobs", default="data/swe_jobs_small.csv")
    args = ap.parse_args(argv)

    pool, pairs, dropped = load_pairs(args.pairs, args.seed, args.psid, args.jobs)
    print(f"{len(pairs)} matched pairs (from a fixed pool; {dropped} failed the equivalence "
          f"audit or sit at the top of the ladder)")
    prompt = P.DECISION_PROMPTS[args.prompt]
    cells = [tuple(c.split(":")) for c in args.cells.split(",")]
    markets = sorted({m for m, _ in cells})

    def everything(client):
        from concurrent.futures import ThreadPoolExecutor, as_completed
        all_paths, all_dec = [], []
        for market in markets:                           # coin controls: no API calls
            for pair in pairs:
                p, d = run_pair(pair, pool, client, prompt, args.years, args.seed, market,
                                "full", "coin")
                all_paths += p
                all_dec += d
        jobs = [(pair, market, view) for market, view in cells for pair in pairs]
        workers = 1 if client.dry_run else max(1, args.workers)
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(run_pair, pair, pool, client, prompt, args.years, args.seed,
                              market, view, "llm") for pair, market, view in jobs]
            try:
                for i, fut in enumerate(as_completed(futs), 1):
                    p, d = fut.result()
                    all_paths += p
                    all_dec += d
                    if not client.dry_run and i % 20 == 0:
                        print(f"  {i}/{len(jobs)} pair-trajectories  {client.meter.summary()}",
                              flush=True)
            except BudgetExceeded as e:
                print(f"\n!! {e}")
                for f in futs:
                    f.cancel()
        return all_paths, all_dec

    dry = make_client(args.model, temperature=args.temperature, dry_run=True)
    everything(dry)
    guard_budget(dry, args.budget)
    if args.dry_run:
        return
    client = make_client(args.model, temperature=args.temperature, budget_usd=args.budget)
    all_paths, all_dec = everything(client)

    paths, dec = pd.DataFrame(all_paths), pd.DataFrame(all_dec)
    prefix = f"trajectory_{args.model}"
    paths.to_csv(os.path.join(OUT, f"{prefix}_paths.csv"), index=False)
    dec.to_csv(os.path.join(OUT, f"{prefix}_decisions.csv"), index=False)
    s = summarise(paths, dec, args.years)
    s.to_csv(os.path.join(OUT, f"{prefix}_summary.csv"), index=False)
    pd.set_option("display.width", 250)
    print()
    print(s.round(3).to_string(index=False))
    print("""
Reading:
  reselect_rate   share of later contests won by the year-0 winner. 0.5 = the
                  decision-maker carries nothing forward; above 0.5 = the
                  model re-transmits the first outcome. Compare full vs
                  no_salary vs coin under the same market.
  agree_rate      share of decisions where both presentation orders agreed;
                  the rest were decided by coin (position bias, not preference).
  A_wins_year0    should be ~50 if the model treats the twins as equals.""")
    print("\n" + client.meter.summary())


if __name__ == "__main__":
    main()
