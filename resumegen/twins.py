"""
Matched pairs: the design the research question actually needs.

The question is: two equally qualified candidates apply for one job, an LLM
picks one, and twenty years later how far apart are they?

To answer that you cannot use two arbitrary people. You need TWINS -- two
candidates constructed to be equivalent on every merit-relevant dimension, but
who are visibly different people. Then the LLM's choice between them is, by
construction, arbitrary, and everything that follows is attributable to the
choice rather than to the candidates.

What is held equal (by construction, and asserted):
    seniority level, years of experience, degree level, number of positions,
    number of career breaks, current salary (within a tight band), latent
    ability

What is allowed to differ (surface only):
    employer names, city, field of study, specialism, project details,
    the wording of every bullet, and the name inserted afterwards

The equivalence is CHECKABLE -- `verify_pair` returns the matched and unmatched
dimensions, so the claim "these two are equal" is auditable rather than
asserted.

One caveat worth stating up front: this guarantees equivalence in the
*construct*, not in the model's perception of it. If an LLM systematically
prefers twin A over twin B across many pairs, that is measurement bias, not
cumulative advantage -- and the twin design is what makes the two separable.
`--swap-order` exists for exactly that test.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import copy
import hashlib

import numpy as np

from resumegen.career import (Career, Position, PostingPool, build_career,
                              FIELDS, SCHOOLS, SPECIALISMS, HIGHLIGHTS,
                              LEVEL_TITLE, MAX_LEVEL, _fill)


def _rng(*parts) -> np.random.Generator:
    h = hashlib.sha256("|".join(map(str, parts)).encode()).digest()
    return np.random.default_rng(int.from_bytes(h[:8], "big"))


@dataclass
class TwinPair:
    pair_id: int
    a_persona: object
    b_persona: object
    a_career: Career
    b_career: Career

    def salaries(self) -> tuple[int, int]:
        return self.a_career.current_salary(), self.b_career.current_salary()


# ---------------------------------------------------------------------------

def pair_specialise(pair_id: int) -> bool:
    """Whether this pair's titles carry the specialism. One draw, shared."""
    return bool(_rng("spec_coin", pair_id).random() < 0.6)


def make_twin(career: Career, persona, pair_id: int, which: str,
              pool: PostingPool, as_of: int,
              specialism: Optional[str] = None) -> tuple[object, Career]:
    """Re-skin a career: same structure, entirely different surface."""
    rng = _rng("twin", pair_id, which)
    c = copy.deepcopy(career)
    p = copy.deepcopy(persona)

    p.persona_id = pair_id * 2 + (0 if which == "A" else 1)
    p.traits = dict(p.traits)

    # different school and field, same degree level
    # A and B must land on different schools and fields, or the "twins" are
    # only nominally distinguishable and the audit rightly fails them.
    offset = 0 if which == "A" else len(FIELDS) // 2
    c.field_of_study = FIELDS[(int(rng.integers(len(FIELDS))) + offset) % len(FIELDS)]
    soff = 0 if which == "A" else len(SCHOOLS) // 2
    c.institution = SCHOOLS[(int(rng.integers(len(SCHOOLS))) + soff) % len(SCHOOLS)]

    # Twins share a specialism on purpose: a Data Engineer and a Frontend
    # Engineer are not interchangeable, so letting these diverge would smuggle
    # a real difference into a pair that claims to have none.
    specialism = specialism or str(rng.choice(SPECIALISMS))
    for pos in c.positions:
        if pos.employment_type == "gap":
            continue
        pos.company = pool.company(rng)
        pos.location = pool.location(rng)
        pos.title = pool.title(pos.level, rng, specialism,
                               specialise=pair_specialise(pair_id))
        n = len(pos.highlights)
        bucket = list(HIGHLIGHTS[min(pos.level, MAX_LEVEL)])
        rng.shuffle(bucket)
        pos.highlights = [_fill(h, rng) for h in bucket[:n]]

    # surface traits differ; nothing merit-bearing does
    for k in ("Personal Time", "Hobbies", "Defining Quirks",
              "Communication Style", "Place of Birth"):
        from resumegen.traits import TABLE3
        if k in TABLE3:
            p.traits[k] = str(rng.choice(TABLE3[k]))

    return p, c


def make_pair(base_persona, pool: PostingPool, pair_id: int,
              as_of: int = 2021, seed: int = 0,
              salary_tolerance: float = 0.02) -> TwinPair:
    """Build one matched pair from a base persona.

    Both twins inherit the same career SHAPE -- same levels, same dates, same
    number of moves -- so experience and seniority match exactly. Salaries are
    then equalised to within `salary_tolerance`, because a visible pay gap at
    the branch point would be the very advantage we are trying to create.
    """
    base = build_career(base_persona, pool, as_of=as_of, seed=seed)

    shared_specialism = str(_rng("spec", pair_id).choice(SPECIALISMS))
    pa, ca = make_twin(base, base_persona, pair_id, "A", pool, as_of,
                       specialism=shared_specialism)
    pb, cb = make_twin(base, base_persona, pair_id, "B", pool, as_of,
                       specialism=shared_specialism)

    # equalise pay exactly at the branch point, with a hair of jitter so the
    # two resumes are not trivially identical on the number
    rng = _rng("pairsalary", pair_id)
    for posa, posb in zip(ca.positions, cb.positions):
        if posa.employment_type == "gap":
            continue
        target = (posa.salary + posb.salary) / 2
        posa.salary = int(round(target * rng.uniform(1 - salary_tolerance / 2,
                                                     1 + salary_tolerance / 2), -2))
        posb.salary = int(round(target * rng.uniform(1 - salary_tolerance / 2,
                                                     1 + salary_tolerance / 2), -2))

    # identical latent ability: they really are equally good
    pa.latent_ability = pb.latent_ability = float(base_persona.latent_ability)
    return TwinPair(pair_id, pa, pb, ca, cb)


# ---------------------------------------------------------------------------

MERIT_DIMENSIONS = ["level", "current_title", "years_experience", "degree",
                    "n_positions", "n_breaks", "n_promotions", "latent_ability",
                    "salary"]


def _profile(career: Career, persona, as_of: int) -> dict:
    real = [p for p in career.positions if p.employment_type != "gap"]
    return {
        "level": career.current().level if career.current() else None,
        "current_title": career.current().title if career.current() else None,
        "years_experience": career.years_experience(as_of),
        "degree": career.degree,
        "n_positions": len(real),
        "n_breaks": len(career.positions) - len(real),
        "n_promotions": career.n_promotions(),
        "latent_ability": round(persona.latent_ability, 6),
        "salary": career.current_salary(),
    }


def verify_pair(pair: TwinPair, as_of: int = 2021,
                salary_tolerance: float = 0.03) -> dict:
    """Audit the equivalence claim. Returns matched / unmatched dimensions."""
    a = _profile(pair.a_career, pair.a_persona, as_of)
    b = _profile(pair.b_career, pair.b_persona, as_of)

    matched, unmatched = [], []
    for k in MERIT_DIMENSIONS:
        if k == "salary":
            lo, hi = sorted((a[k], b[k]))
            ok = hi <= lo * (1 + salary_tolerance)
        else:
            ok = a[k] == b[k]
        (matched if ok else unmatched).append(k)

    surface_differs = sum([
        pair.a_career.institution != pair.b_career.institution,
        pair.a_career.field_of_study != pair.b_career.field_of_study,
        {p.company for p in pair.a_career.positions}
        != {p.company for p in pair.b_career.positions},
    ])
    return {"pair_id": pair.pair_id, "a": a, "b": b,
            "matched": matched, "unmatched": unmatched,
            "surface_differences": surface_differs,
            "equivalent": not unmatched and surface_differs >= 2}
