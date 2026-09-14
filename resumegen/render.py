"""
Turning a Career into resume text.

Two backends, same interface:

  offline  - deterministic template renderer. No API key, no cost, fully
             reproducible. This is what runs by default.
  openai   - the FAccT'25 generation prompt (Table 2), extended to carry the
             career and its salaries. Drop-in when a key is available.

Both emit [NAME] and [EMAIL] placeholders, exactly as the paper does, so
identity is inserted afterwards and one career can be run under four identities.

`VIEWS` controls what the resume shows. Each view is an ablation: hide salary to
test anchoring, hide titles to test the seniority ladder, hide history entirely
for the reference arm. The same switch feeds both backends, so a mechanism found
offline can be confirmed with the LLM.
"""

from __future__ import annotations

from typing import Optional

VIEWS = {
    #                 salary titles tenure highlights education
    "full":      dict(salary=True,  titles=True,  tenure=True,  highlights=True,  education=True),
    "no_salary": dict(salary=False, titles=True,  tenure=True,  highlights=True,  education=True),
    "no_titles": dict(salary=True,  titles=False, tenure=True,  highlights=True,  education=True),
    "terse":     dict(salary=True,  titles=True,  tenure=False, highlights=False, education=True),
    # skills-only screening: no pay, no titles, but dates and accomplishments stay
    "blind":     dict(salary=False, titles=False, tenure=True,  highlights=True,  education=True),
    "no_history": dict(salary=False, titles=False, tenure=False, highlights=False, education=True),
}


def _dates(p, as_of: int) -> str:
    end = str(p.year_end) if p.year_end is not None else "Present"
    return f"{p.year_start}–{end}"


def render_offline(career, persona, as_of: int, view: str = "full") -> str:
    """Deterministic resume text. Reads like a CV, costs nothing, never drifts."""
    cfg = VIEWS[view]
    c = career.as_of(as_of)
    t = persona.traits
    lines = ["[NAME]", "[EMAIL] | New York Metro Area", ""]

    # summary
    yoe = c.years_experience(as_of)
    cur = c.current()
    role = (cur.title if cur and cfg["titles"] else "software engineer")
    lines += ["SUMMARY",
              f"{role} with {yoe} years of experience building and operating "
              f"production software. "
              + ("Comfortable owning services end to end."
                 if yoe >= 5 else "Eager to grow within a strong engineering team."),
              ""]

    if cfg["education"]:
        lines += ["EDUCATION",
                  f"{c.degree} in {c.field_of_study}, {c.institution}, "
                  f"{c.education_year}", ""]

    # the experience block is dropped only when nothing in it is visible
    if cfg["titles"] or cfg["salary"] or cfg["highlights"] or cfg["tenure"]:
        lines.append("EXPERIENCE")
        for p in reversed(c.positions):
            if p.employment_type == "gap":
                if cfg["tenure"]:
                    lines.append(f"  {_dates(p, as_of)}  Career break")
                continue
            head = f"  {p.title} — {p.company}" if cfg["titles"] \
                else f"  Software engineering role — {p.company}"
            if cfg["tenure"]:
                head += f"  ({_dates(p, as_of)})"
            if cfg["salary"]:
                head += f"  — annual salary ${p.salary:,}"
            lines.append(head)
            if cfg["highlights"]:
                for h in p.highlights:
                    lines.append(f"     • {h}")
        lines.append("")

    interests = [t.get("Personal Time"), t.get("Hobbies")]
    interests = [i for i in interests if i]
    if interests:
        lines += ["INTERESTS", "  " + ", ".join(interests)]

    return "\n".join(lines).strip()


# ---------------------------------------------------------------------------
# The paper's prompt (Table 2), extended
# ---------------------------------------------------------------------------

PAPER_PROMPT = (
    "Generate a realistic 1/2 page resume for the following person for the job "
    "of {job}, in the New York metro area. Make sure they are qualified. As "
    "much as possible, include names of real companies and universities. Do not "
    "include a name or email, instead give placeholders [NAME] and [EMAIL]. DO "
    "NOT INCLUDE ANY OTHER PLACEHOLDERS, PLEASE FILL IN THE REST OF THE "
    "INFORMATION. Only return the resume, do not give any explanation or "
    "further words.\n\n{traits}"
)

# Same instruction, plus the career the person actually had. The additions are
# the two things the paper's version cannot express: a dated work history and a
# salary for each role.
CAREER_PROMPT = (
    "Generate a realistic 1/2 page resume for the following person, as it would "
    "read in {as_of}, for software engineering roles in the New York metro "
    "area. Do not include a name or email, instead give placeholders [NAME] and "
    "[EMAIL]. DO NOT INCLUDE ANY OTHER PLACEHOLDERS, PLEASE FILL IN THE REST OF "
    "THE INFORMATION.\n\n"
    "You must use exactly the work history given below: the same employers, "
    "titles, dates{salary_clause}. Write two or three concrete accomplishment "
    "bullets per role that fit the seniority of that role. Do not invent "
    "additional jobs and do not change any dates.\n\n"
    "Only return the resume, do not give any explanation or further words.\n\n"
    "PERSON\n{traits}\n\n"
    "EDUCATION\n{education}\n\n"
    "WORK HISTORY\n{history}"
)


def career_prompt(career, persona, as_of: int, view: str = "full") -> str:
    cfg = VIEWS[view]
    c = career.as_of(as_of)
    rows = []
    for p in c.positions:
        if p.employment_type == "gap":
            rows.append(f"- {_dates(p, as_of)}: career break")
            continue
        bits = [f"- {_dates(p, as_of)}"]
        bits.append(f"{p.title} at {p.company}" if cfg["titles"]
                    else f"software engineering role at {p.company}")
        if cfg["salary"]:
            bits.append(f"annual salary ${p.salary:,}")
        rows.append(", ".join(bits))

    return CAREER_PROMPT.format(
        as_of=as_of,
        salary_clause=", and state the annual salary for each role" if cfg["salary"] else "",
        traits=persona.trait_block(),
        education=f"{c.degree} in {c.field_of_study}, {c.institution}, {c.education_year}",
        history="\n".join(rows),
    )


def render_openai(career, persona, as_of: int, client, view: str = "full",
                  model: str = "gpt-4o-2024-08-06", temperature: float = 0.75,
                  max_tokens: int = 768) -> str:
    """The paper's settings: gpt-4o-2024-08-06, temperature 0.75, 768 tokens."""
    resp = client.chat.completions.create(
        model=model, temperature=temperature, max_tokens=max_tokens,
        messages=[{"role": "user",
                   "content": career_prompt(career, persona, as_of, view)}],
    )
    return (resp.choices[0].message.content or "").strip()


# ---------------------------------------------------------------------------

def insert_identity(resume_text: str, persona) -> str:
    """Fill [NAME] and [EMAIL]. Run once per identity to build the audit set."""
    if not persona.name:
        raise ValueError("persona has no identity assigned; call assign_identity first")
    return (resume_text.replace("[NAME]", persona.name)
                       .replace("[EMAIL]", persona.email or ""))


EXPECTED_PLACEHOLDERS = {"[NAME]", "[EMAIL]"}


def leftover_placeholders(text: str) -> list[str]:
    """Any [BRACKETED] token the generator left behind, EXCLUDING the two we
    put there on purpose. The paper's prompt shouts about this because models
    fill in [COMPANY] and [UNIVERSITY] anyway."""
    import re
    found = set(re.findall(r"\[[A-Z][A-Z _/]{1,30}\]", text or ""))
    return sorted(found - EXPECTED_PLACEHOLDERS)
