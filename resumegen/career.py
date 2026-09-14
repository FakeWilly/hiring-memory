"""
Careers: the two things the FAccT'25 generator does not produce.

The paper generates ONE resume at ONE moment, with no salary attached
("Income Bracket" is a persona trait, not a number on the CV). A cumulative
advantage study needs neither of those to be true: it needs a resume that grows,
and a salary attached to every step, because how a career responds to a good or
bad year IS the mechanism under study.

A Career is a latent object. Rendering it at year Y gives the resume that person
would have submitted in year Y, so the same person can be shown to a
decision-maker at 2027 and again at 2035 without regenerating anything.

Salaries are drawn from REAL postings (the ASEE software dataset already in the
repo) so bands, titles and companies stay grounded. A per-person multiplier
persists across the whole career, which is what makes some people consistently
paid above band -- the thing a Matthew effect would act on.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional
import hashlib
import re

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Seniority ladder
# ---------------------------------------------------------------------------

LADDER = [
    (0, "Junior Software Engineer",    0),
    (1, "Software Engineer",           2),
    (2, "Senior Software Engineer",    5),
    (3, "Staff Software Engineer",     9),
    (4, "Principal Software Engineer", 13),
]
LEVEL_TITLE = {i: t for i, t, _ in LADDER}
LEVEL_MIN_YOE = {i: y for i, _, y in LADDER}
MAX_LEVEL = max(LEVEL_TITLE)

# Fallback salary bands (2021 USD) if no postings file is supplied.
FALLBACK_BANDS = {0: 85_000, 1: 118_000, 2: 155_000, 3: 195_000, 4: 240_000}

TITLE_PATTERNS = {
    0: ["junior", "associate", " i ", "entry"],
    2: ["senior", "sr."],
    3: ["staff", "architect"],
    4: ["principal", "distinguished", "fellow"],
}


def _rng(*parts) -> np.random.Generator:
    h = hashlib.sha256("|".join(map(str, parts)).encode()).digest()
    return np.random.default_rng(int.from_bytes(h[:8], "big"))


# ---------------------------------------------------------------------------

@dataclass
class Position:
    year_start: int
    year_end: Optional[int]          # None == current
    title: str
    company: str
    level: int
    salary: int
    location: str = "New York, NY"
    is_promotion: bool = False
    employment_type: str = "full-time"   # or "gap"
    highlights: list[str] = field(default_factory=list)

    def tenure(self, as_of: int) -> int:
        return max(1, (self.year_end or as_of) - self.year_start)


@dataclass
class Career:
    persona_id: int
    positions: list[Position]
    education_year: int
    field_of_study: str
    institution: str
    degree: str
    salary_multiplier: float = 1.0

    def as_of(self, year: int) -> "Career":
        """A copy truncated to what existed in `year`."""
        import copy
        c = copy.deepcopy(self)
        c.positions = [p for p in c.positions if p.year_start <= year]
        for p in c.positions:
            if p.year_end is not None and p.year_end > year:
                p.year_end = None
        return c

    def current(self) -> Optional[Position]:
        real = [p for p in self.positions if p.employment_type != "gap"]
        return real[-1] if real else None

    def current_salary(self) -> Optional[int]:
        p = self.current()
        return p.salary if p else None

    def years_experience(self, as_of: int) -> int:
        return sum(p.tenure(as_of) for p in self.positions
                   if p.employment_type != "gap")

    def n_promotions(self) -> int:
        return sum(1 for p in self.positions if p.is_promotion)

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Posting pool
# ---------------------------------------------------------------------------

SPECIALISMS = ["Backend", "Full Stack", "Platform", "Frontend", "Data",
               "Infrastructure", "Systems"]

# Company strings in the ASEE file carry recruiting cruft. Drop these.
_BAD_COMPANY = re.compile(
    r"intern|co-?op|new grad|careers page|staffing|recruit|award winning|"
    r"growth mode|confidential|\bhiring\b", re.I)


def clean_company(name: str) -> Optional[str]:
    """Keep real employer names, reject job-board artefacts."""
    c = str(name or "").strip()
    if not c or len(c) > 45 or _BAD_COMPANY.search(c):
        return None
    if c.isupper() and len(c) > 18:          # shouty staffing agencies
        return None
    c = re.sub(r"^[0-9]{2,6}\s+", "", c)          # "0001 Applied Materials"
    c = re.sub(r"\s*\((?:ABSc|Inc\.?|LLC|USA)\)\s*$", "", c)
    if len(c) < 3:
        return None
    return c


class PostingPool:
    """Real postings, used for two things only: real COMPANY names, and the
    empirical salary BAND at each seniority level.

    Deliberately NOT used for job titles. Posting titles are adverts for
    current openings and carry things a resume never would -- start dates,
    "Remote - Home Based Worker", "for New Grads, Co-Ops & Internships". Using
    them verbatim produced mid-career people whose current role was an
    internship. Titles come from the ladder instead, with a specialism.
    """

    def __init__(self, jobs: Optional[pd.DataFrame] = None):
        self.by_level: dict[int, pd.DataFrame] = {}
        self.bands: dict[int, float] = dict(FALLBACK_BANDS)
        self.companies: list[str] = []
        self.locations: list[str] = ["New York, NY", "Jersey City, NJ",
                                     "Stamford, CT", "Brooklyn, NY"]
        if jobs is None or len(jobs) == 0:
            self.empty = True
            return
        self.empty = False
        j = jobs.copy()
        j["_level"] = [self._infer_level(t, s)
                       for t, s in zip(j["title"], j["avg_amount"])]
        for lvl in LEVEL_TITLE:
            sub = j[j["_level"] == lvl]
            if len(sub) >= 5:
                self.by_level[lvl] = sub.reset_index(drop=True)
                self.bands[lvl] = float(sub["avg_amount"].median())
        # enforce a monotone ladder of bands
        for lvl in sorted(self.bands):
            if lvl > 0:
                self.bands[lvl] = max(self.bands[lvl],
                                      self.bands[lvl - 1] * 1.12)
        seen = {clean_company(c) for c in j["company"].dropna().unique()}
        self.companies = sorted(c for c in seen if c)
        metro = re.compile(r",\s*(NY|NJ|CT)\b")
        locs = [str(l) for l in j["location"].dropna().unique()
                if isinstance(l, str) and 3 < len(l) < 30 and metro.search(l)]
        if len(locs) >= 4:
            self.locations = sorted(set(locs))

    @staticmethod
    def _infer_level(title: str, salary: float) -> int:
        t = f" {str(title).lower()} "
        for lvl in (4, 3, 2, 0):
            if any(p in t for p in TITLE_PATTERNS[lvl]):
                return lvl
        for cut, lvl in ((235_000, 4), (190_000, 3), (150_000, 2), (110_000, 1)):
            if salary >= cut:
                return lvl
        return 0

    def band(self, level: int) -> float:
        return self.bands[int(np.clip(level, 0, MAX_LEVEL))]

    def company(self, rng) -> str:
        if not self.companies:
            return "Confidential"
        return str(self.companies[int(rng.integers(len(self.companies)))])

    def location(self, rng) -> str:
        return str(self.locations[int(rng.integers(len(self.locations)))])

    def title(self, level: int, rng, specialism: Optional[str] = None,
              specialise: Optional[bool] = None) -> str:
        """Canonical ladder title, optionally specialised.

        `specialise` fixes whether the specialism appears in the title. Twins
        must share it (drawn once per pair), otherwise "Senior Data Engineer"
        vs "Senior Software Engineer" is a real difference between two people
        who are supposed to have none.
        """
        base = LEVEL_TITLE[int(np.clip(level, 0, MAX_LEVEL))]
        if specialise is None:
            specialise = rng.random() < 0.6
        if specialism and level >= 1 and specialise:
            return base.replace("Software Engineer",
                                f"{specialism} Engineer") \
                       if specialism in ("Backend", "Frontend", "Platform",
                                         "Data", "Infrastructure", "Systems") \
                       else base
        return base


# ---------------------------------------------------------------------------
# Career synthesis
# ---------------------------------------------------------------------------

FIELDS = ["Computer Science", "Information Systems", "Electrical Engineering",
          "Mathematics", "Statistics", "Software Engineering"]
SCHOOLS = ["Rutgers University", "Stony Brook University", "CUNY City College",
           "Northeastern University", "Penn State University",
           "University of Connecticut", "Drexel University",
           "Rochester Institute of Technology", "Baruch College"]

HIGHLIGHTS = {
    0: ["Shipped features across the {area} codebase with mentorship from senior engineers.",
        "Wrote and maintained unit and integration tests for {area}.",
        "Resolved production issues in {area} during business-hours rotation."],
    1: ["Owned the {area} service end to end, including releases and on-call.",
        "Cut {metric} by {pct}% by {action}.",
        "Partnered with product and design to deliver {feature}."],
    2: ["Led design of {feature}, adopted by {n} teams.",
        "Reduced {metric} by {pct}% through {action}.",
        "Mentored {n} engineers and ran the team's design review."],
    3: ["Set technical direction for {area} across {n} teams.",
        "Drove the migration from {old} to {new} with no customer-facing downtime.",
        "Cut {metric} by {pct}% at platform scale."],
    4: ["Authored the multi-year roadmap for {area}.",
        "Chaired org-wide architecture review across {n} teams.",
        "Led the {old}-to-{new} programme spanning {n} teams."],
}

FILL = {
    "area": ["payments", "identity", "search", "data platform", "billing",
             "notifications", "deployment"],
    "feature": ["a self-serve reporting tool", "the onboarding flow",
                "an internal experimentation platform", "a real-time alerting pipeline"],
    "metric": ["p99 latency", "infrastructure cost", "build time",
               "incident volume", "onboarding time"],
    "action": ["introducing request-level caching", "rewriting the hot path",
               "consolidating three services", "adding batch processing"],
    "old": ["a monolith", "self-managed VMs", "nightly batch jobs", "Python 2"],
    "new": ["microservices", "Kubernetes", "streaming ingestion", "Python 3"],
}


def _fill(tpl: str, rng) -> str:
    out = tpl
    if "{old}" in out and "{new}" in out:
        # keep migrations coherent: a monolith becomes microservices, not Python 3
        i = int(rng.integers(len(FILL["old"])))
        out = out.replace("{old}", FILL["old"][i]).replace("{new}", FILL["new"][i])
    for k, opts in FILL.items():
        if "{" + k + "}" in out:
            out = out.replace("{" + k + "}", str(rng.choice(opts)))
    out = out.replace("{pct}", str(int(rng.integers(15, 55))))
    out = out.replace("{n}", str(int(rng.integers(3, 12))))
    return out


def _degree_years(education: str) -> int:
    return {"Associate's Degree": 14, "Bachelor's Degree": 16,
            "Master's Degree": 18}.get(education, 16)


def build_career(persona, pool: PostingPool, as_of: int = 2021,
                 seed: int = 0,
                 ability_on_promo: float = 0.45,
                 promo_intercept: float = 0.55,
                 level_penalty: float = 1.15,
                 overdue_slope: float = 0.11,
                 lateral_cut_prob: float = 0.10,
                 max_move_up: float = 1.35,
                 gap_prob: float = 0.06,
                 within_job_raise: float = 0.028,
                 move_premium: float = 0.045,
                 wage_drift: float = 0.022) -> Career:
    """Synthesise one plausible career ending in `as_of`.

    Salary rules, chosen so trajectories look like careers rather than random
    draws (the failure mode in the current pipeline):
      * each level has an empirical band from the real postings
      * a persistent per-person multiplier -- some people sit above band for
        their whole career, which is what a Matthew effect would act on
      * staying put earns a small annual raise
      * changing jobs earns a premium over current pay, so a voluntary move is
        never a pay cut
      * only a spell of unemployment can cut pay

    Ability enters in exactly two places, both modest and explicit: the chance
    of promotion, and the pay multiplier. Everything else is tenure and luck.
    Keeping the merit channel small and named is what makes it possible to ask
    later how much of the observed advantage is merit.
    """
    rng = _rng("career", seed, persona.persona_id)
    t = persona.traits
    age = int(t.get("Age", 32))
    education = str(t.get("Education", "Bachelor's Degree"))
    edu_years = _degree_years(education)

    grad_age = edu_years + 6
    years_working = max(0, age - grad_age)
    education_year = as_of - years_working

    ability = float(getattr(persona, "latent_ability", 0.0))
    multiplier = float(np.clip(np.exp(0.055 * ability + rng.normal(0, 0.045)),
                               0.80, 1.30))
    specialism = str(rng.choice(SPECIALISMS))

    positions: list[Position] = []
    used_highlights: set[str] = set()
    year = education_year
    level = 1 if (education == "Master's Degree" and years_working >= 1) else 0
    gap_after: set[int] = set()      # index of positions preceded by a gap

    # ---- pass 1: structure (dates, levels, employers) --------------------
    while year < as_of:
        tenure = int(rng.integers(2, 6))
        end = min(year + tenure, as_of)
        if end - year < 1:
            break

        want = 2 if level <= 1 else 3
        full = HIGHLIGHTS[min(level, MAX_LEVEL)]
        fresh = [h for h in full if h not in used_highlights]
        rng.shuffle(fresh)
        chosen = fresh[:want]
        if len(chosen) < want:                      # bucket exhausted; reuse
            spare = [h for h in full if h not in chosen]
            rng.shuffle(spare)
            chosen += spare[:want - len(chosen)]
        used_highlights.update(chosen)

        positions.append(Position(
            year_start=year, year_end=None if end >= as_of else end,
            title=pool.title(level, rng, specialism),
            company=pool.company(rng),
            level=level, salary=0, location=pool.location(rng),
            is_promotion=bool(positions) and level > positions[-1].level,
            highlights=[_fill(c, rng) for c in chosen],
        ))

        year = end
        if year >= as_of:
            break

        if rng.random() < gap_prob and year + 1 < as_of:
            positions.append(Position(
                year_start=year, year_end=year + 1, title="", company="",
                level=level, salary=0, employment_type="gap", highlights=[]))
            year += 1
            gap_after.add(len(positions))     # the NEXT position follows a gap

        yoe = year - education_year
        if level < MAX_LEVEL and yoe >= LEVEL_MIN_YOE[level + 1]:
            overdue = yoe - LEVEL_MIN_YOE[level + 1]
            # Promotion gets harder the higher you go -- the pyramid narrows.
            # Without the level penalty half the cohort ended up Staff+, which
            # no real engineering org looks like.
            logit = (promo_intercept - level_penalty * level
                     + overdue_slope * overdue + ability_on_promo * ability)
            if rng.random() < 1 / (1 + np.exp(-logit)):
                level += 1

    # ---- pass 2: salaries ------------------------------------------------
    # Anchor to the CURRENT level's empirical band, then back-date. The
    # postings are present-day adverts, so a 2008 role must be discounted or
    # the cross-section comes out far richer than the market it was drawn from.
    prev = None
    for i, p in enumerate(positions):
        if p.employment_type == "gap":
            continue
        deflate = (1 + wage_drift) ** (p.year_start - as_of)
        base = pool.band(p.level) * multiplier * deflate
        sal = base * rng.normal(1.0, 0.05)
        if prev is not None:
            if i in gap_after:
                # re-entry after a break: the main way pay falls
                sal = min(sal, prev * rng.uniform(0.82, 0.98))
                sal = max(sal, base * 0.80)
            elif rng.random() < lateral_cut_prob:
                # a move taken for something other than money -- smaller
                # company, better team, shorter commute. Real, and rarer than
                # the 40% of moves that were pay cuts in the original pipeline.
                sal = min(sal, prev * rng.uniform(0.92, 1.0))
            else:
                sal = max(sal, prev * (1 + move_premium) * rng.normal(1.0, 0.02))
            # No single move should double someone's pay. Without this, a
            # lateral pay cut followed by a snap back to band produced +57%.
            sal = min(sal, prev * max_move_up)
        p.salary = int(round(sal, -2))
        # within-job growth carried into the next comparison
        span = max(0, (p.year_end or as_of) - p.year_start - 1)
        prev = p.salary * (1 + within_job_raise) ** span

    if not positions:
        positions = [Position(
            year_start=as_of - 1, year_end=None,
            title=pool.title(0, rng, specialism), company=pool.company(rng),
            level=0, salary=int(round(pool.band(0) * multiplier, -2)),
            location=pool.location(rng),
            highlights=[_fill(HIGHLIGHTS[0][0], rng)])]

    return Career(
        persona_id=persona.persona_id,
        positions=positions,
        education_year=education_year,
        field_of_study=str(rng.choice(FIELDS)),
        institution=str(rng.choice(SCHOOLS)),
        degree=education,
        salary_multiplier=round(multiplier, 4),
    )
