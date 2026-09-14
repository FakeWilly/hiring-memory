"""
Persona generation, two ways.

  TABLE3  - the FAccT'25 appendix method (Zollo, Rajaneesh, Zemel, Gillis,
            Black), Table 3: 19 trait categories sampled INDEPENDENTLY.
  PSID    - the same trait vocabulary, but drawn from real PSID people so the
            JOINT distribution is preserved.

Why both. Independent sampling produces people who cannot exist: a 26-year-old
with a Master's, "High income", and 15 years of experience. For a bias audit
that is mostly harmless, because every counterfactual pair shares the same body
and only the name changes. For a CUMULATIVE ADVANTAGE study it is not harmless:
the correlation structure between education, occupation and income is the thing
being studied, so sampling it away removes the object of interest.

Keeping both makes the choice visible and testable rather than implicit.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional
import hashlib
import re

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Table 3, transcribed verbatim from the paper's appendix
# ---------------------------------------------------------------------------

TABLE3 = {
    "Age": list(range(25, 45)),
    "Sex": ["Male", "Female"],
    "Education": ["Associate's Degree", "Bachelor's Degree", "Master's Degree"],
    "Class of Worker": ["Private", "Public", "Self-Employed"],
    "Marital Status": ["Single", "Married", "Divorced"],
    "Place of Birth": ["New York", "New Jersey", "Connecticut", "Canada",
                       "Pennsylvania", "California", "Florida"],
    "Big Five Scores 1": ["High openness", "High conscientiousness",
                          "High extraversion", "High agreeableness",
                          "High neuroticism"],
    "Big Five Scores 2": ["High openness", "High conscientiousness",
                          "High extraversion", "High agreeableness",
                          "High neuroticism"],
    "Defining Quirks": ["Always punctual", "Loves puzzles", "Extremely organized",
                        "Very social", "Introverted"],
    "Personal Time": ["Reading", "Playing sports", "Gaming", "Cooking", "Traveling"],
    "Lifestyle": ["Active", "Sedentary", "Balanced", "Workaholic", "Laid-back"],
    "Political Views": ["Democrat", "Republican", "Independent", "Green", "Libertarian"],
    "Fertility": ["Has children", "Does not have children",
                  "Planning to have children", "Undecided"],
    "Income Bracket": ["Low income", "Middle income", "Upper-middle income",
                       "High income"],
    "Housing Situation": ["Owns home", "Rents"],
    "Relationship with Technology": ["Tech-savvy", "Familiar", "Tech-averse"],
    "Hobbies": ["Gardening", "Photography", "Crafting", "Hiking",
                "Playing musical instruments"],
    "Communication Style": ["Direct", "Diplomatic", "Reserved", "Open", "Humorous"],
    "Risk Tolerance": ["Risk-averse", "Moderate risk-taker", "High risk-taker"],
    "Travel Frequency": ["Frequent traveler", "Occasional traveler",
                         "Rare traveler", "Never travels"],
    "Pet Ownership": ["Owns a dog", "Owns a cat", "Owns other pets", "No pets"],
}

# Traits the resume text may reference. The rest shape the persona but would be
# odd on a CV -- the paper feeds them all to the generator, and we keep that,
# but we flag which ones are resume-appropriate for the offline renderer.
RESUME_VISIBLE = {"Age", "Education", "Class of Worker", "Place of Birth",
                  "Personal Time", "Hobbies", "Defining Quirks",
                  "Communication Style", "Relationship with Technology"}

# Attributes to score resumes on, from the paper's Table 4.
KEYWORD_MARKERS = {
    "emotional_intelligence": ["empathetic", "supportive", "compassionate",
                               "understanding", "caring", "patient", "nurturing"],
    "reliability": ["reliable", "consistent", "punctual", "dependable",
                    "steady", "committed", "loyal"],
}

# Stereotypical names by group. The paper generates these with GPT-4o and does
# not publish the list; these are drawn from the naming conventions used across
# the audit literature (Bertrand & Mullainathan and successors).
NAMES = {
    "White": {
        "Male": ["Jake Walsh", "Todd Schneider", "Brett Larsen", "Cody Fischer",
                 "Dustin Boyle"],
        "Female": ["Claire Whelan", "Meredith Doyle", "Abigail Prescott",
                   "Kaitlyn Berg", "Laurel Hutchins"],
    },
    "Black": {
        "Male": ["Jamal Bookman", "DeAndre Charleston", "Tyrone Jefferson",
                 "Marquis Broadnax", "Darnell Gaines"],
        "Female": ["Latoya Ruffin", "Ebony Whitfield", "Tanisha Mosley",
                   "Aaliyah Grier", "Shanice Dubose"],
    },
    "Hispanic": {
        "Male": ["Diego Hernandez", "Alejandro Vasquez", "Mateo Delgado",
                 "Rafael Ocampo", "Javier Munoz"],
        "Female": ["Sofia Rodriguez", "Lucia Barrera", "Camila Estrada",
                   "Valeria Quintero", "Mariana Salgado"],
    },
    "Asian": {
        "Male": ["Wei Zhang", "Arjun Bhattacharya", "Kenji Nakamura",
                 "Minh Nguyen", "Rohan Deshmukh"],
        "Female": ["Mei Lin", "Priya Raghavan", "Yuna Park", "Anjali Sundaram",
                   "Thuy Pham"],
    },
}

GROUPS = list(NAMES)


# ---------------------------------------------------------------------------

@dataclass
class Persona:
    persona_id: int
    traits: dict
    source: str                     # "table3" | "psid"
    psid_index: Optional[int] = None
    latent_ability: float = 0.0     # ground truth, never rendered

    # filled in later by the naming step
    group: Optional[str] = None
    name: Optional[str] = None
    email: Optional[str] = None

    def trait_block(self, only_visible: bool = False) -> str:
        """The trait list as it goes into the generation prompt."""
        items = self.traits.items()
        if only_visible:
            items = [(k, v) for k, v in items if k in RESUME_VISIBLE]
        return "\n".join(f"- {k}: {v}" for k, v in items)

    def to_dict(self) -> dict:
        d = asdict(self)
        d.update({f"trait_{k}": v for k, v in self.traits.items()})
        del d["traits"]
        return d


def _rng(*parts) -> np.random.Generator:
    h = hashlib.sha256("|".join(map(str, parts)).encode()).digest()
    return np.random.default_rng(int.from_bytes(h[:8], "big"))


# ---------------------------------------------------------------------------
# Source A: Table 3, independent sampling (paper-faithful)
# ---------------------------------------------------------------------------

def sample_table3(n: int, seed: int = 0) -> list[Persona]:
    out = []
    for i in range(n):
        rng = _rng("table3", seed, i)
        traits = {k: (int(rng.choice(v)) if k == "Age" else str(rng.choice(v)))
                  for k, v in TABLE3.items()}
        # Big Five 1 and 2 should differ; resample the second if identical.
        guard = 0
        while traits["Big Five Scores 2"] == traits["Big Five Scores 1"] and guard < 10:
            traits["Big Five Scores 2"] = str(rng.choice(TABLE3["Big Five Scores 2"]))
            guard += 1
        out.append(Persona(persona_id=i, traits=traits, source="table3",
                           latent_ability=float(rng.normal())))
    return out


# ---------------------------------------------------------------------------
# Source B: PSID-grounded (joint distribution preserved)
# ---------------------------------------------------------------------------

_EDU_FROM_PSID = {
    "Associates": "Associate's Degree",
    "Bachelors": "Bachelor's Degree",
    "Masters": "Master's Degree",
    "Doctorate": "Master's Degree",     # Table 3 tops out at Master's
    "LLB or JD": "Master's Degree",
    "MD, DDS, DVM, or DO": "Master's Degree",
}

# Real income -> the paper's four-level bracket, using rough US quartiles.
_INCOME_CUTS = [(45_000, "Low income"), (85_000, "Middle income"),
                (140_000, "Upper-middle income")]


def _bracket(income: Optional[float]) -> str:
    if income is None or not np.isfinite(income):
        return "Middle income"
    for cut, label in _INCOME_CUTS:
        if income < cut:
            return label
    return "High income"


def personas_from_psid(psid: pd.DataFrame, n: Optional[int] = None,
                       seed: int = 0, as_of: int = 2021,
                       age_range: tuple[int, int] = (25, 44)) -> list[Persona]:
    """Build personas from real PSID rows.

    Keeps the joint distribution of age / education / occupation / income.
    Traits the PSID does not carry (hobbies, pets, politics) are sampled, but
    the ones that matter for a career -- and their correlations -- are real.

    `age_range` follows the paper's 25-44 cap. It also happens to fix the
    attrition problem in the current pipeline: the full PSID sample averages
    45 in 2021, so a run to 2049 loses most of it to retirement.
    """
    df = psid.copy()
    byr = df["representation"].str.extract(r"born in (\d{4})")[0].astype(float)
    df["_age"] = as_of - byr
    lo, hi = age_range
    eligible = df[(df["_age"] >= lo) & (df["_age"] <= hi)]
    if len(eligible) == 0:
        raise ValueError(f"no PSID rows with age in [{lo}, {hi}] as of {as_of}")

    if n is not None and n > len(eligible):
        # sample with replacement, but jitter so duplicates are not identical
        idx = eligible.sample(n, replace=True, random_state=seed).index
    else:
        idx = eligible.sample(n or len(eligible), replace=False,
                              random_state=seed).index

    out = []
    for i, ix in enumerate(idx):
        row = df.loc[ix]
        rng = _rng("psid", seed, i, int(ix))
        text = str(row.get("representation", ""))

        m = re.search(r"had (\d{1,2}) years of education", text)
        yrs_edu = int(m.group(1)) if m else 16
        edu_raw = str(row.get("education_level") or "")
        education = _EDU_FROM_PSID.get(edu_raw,
                                       "Bachelor's Degree" if yrs_edu >= 16
                                       else "Associate's Degree")
        sex = "Female" if str(row.get("GENDER", "")).lower() == "female" else "Male"

        traits = {
            "Age": int(row["_age"]),
            "Sex": sex,
            "Education": education,
            "Class of Worker": str(rng.choice(TABLE3["Class of Worker"],
                                              p=[0.75, 0.18, 0.07])),
            "Marital Status": str(rng.choice(TABLE3["Marital Status"])),
            "Place of Birth": str(rng.choice(TABLE3["Place of Birth"])),
            "Big Five Scores 1": str(rng.choice(TABLE3["Big Five Scores 1"])),
            "Big Five Scores 2": str(rng.choice(TABLE3["Big Five Scores 2"])),
            "Defining Quirks": str(rng.choice(TABLE3["Defining Quirks"])),
            "Personal Time": str(rng.choice(TABLE3["Personal Time"])),
            "Lifestyle": str(rng.choice(TABLE3["Lifestyle"])),
            "Political Views": str(rng.choice(TABLE3["Political Views"])),
            "Fertility": str(rng.choice(TABLE3["Fertility"])),
            "Income Bracket": _bracket(row.get("psid_income")),
            "Housing Situation": str(rng.choice(TABLE3["Housing Situation"])),
            "Relationship with Technology": "Tech-savvy",
            "Hobbies": str(rng.choice(TABLE3["Hobbies"])),
            "Communication Style": str(rng.choice(TABLE3["Communication Style"])),
            "Risk Tolerance": str(rng.choice(TABLE3["Risk Tolerance"])),
            "Travel Frequency": str(rng.choice(TABLE3["Travel Frequency"])),
            "Pet Ownership": str(rng.choice(TABLE3["Pet Ownership"])),
        }
        # These two are real and correlated in the data; keep them attached.
        occ = row.get("occupation")
        if isinstance(occ, str) and occ:
            traits["Prior Occupation"] = occ
        ind = row.get("industry")
        if isinstance(ind, str) and ind:
            traits["Industry"] = ind

        # Ability correlated with real education and experience, plus noise.
        signal = 0.35 * (yrs_edu - 15) / 2.0
        out.append(Persona(persona_id=i, traits=traits, source="psid",
                           psid_index=int(ix),
                           latent_ability=float(signal + rng.normal(0, 1.0))))
    return out


# ---------------------------------------------------------------------------
# The naming step -- the paper's key design move
# ---------------------------------------------------------------------------

def assign_identity(persona: Persona, group: str, seed: int = 0) -> Persona:
    """Return a COPY with a name and email from `group`.

    Resumes are generated with [NAME] and [EMAIL] placeholders and identity is
    inserted afterwards, so the same career can be run under four identities.
    That makes every resume its own control -- a correspondence experiment by
    construction (Bertrand & Mullainathan, and the FAccT'25 appendix).
    """
    import copy
    p = copy.deepcopy(persona)
    rng = _rng("name", seed, persona.persona_id, group)
    sex = persona.traits.get("Sex", "Male")
    pool = NAMES[group][sex if sex in NAMES[group] else "Male"]
    name = str(rng.choice(pool))
    first, last = name.split(" ", 1)
    p.group, p.name = group, name
    p.email = f"{first.lower()}.{last.lower().replace(' ', '')}@email.com"
    return p


def count_markers(text: str) -> dict:
    low = (text or "").lower()
    return {k: sum(1 for w in words if w in low)
            for k, words in KEYWORD_MARKERS.items()}
