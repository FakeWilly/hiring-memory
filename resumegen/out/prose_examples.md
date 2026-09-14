# What the LLM backend will produce

**Provenance, please read.** These three were written by **Claude**, following
`render.CAREER_PROMPT` exactly, from careers that are really in
`careers.jsonl`. They are here to show the target quality and the shape of the
output — they are **not** GPT-4o output, and they are not in the shipped
dataset. Running `--backend openai` produces the GPT-4o versions with
`gpt-4o-2024-08-06`, temperature 0.75, 768 max tokens (the paper's settings).

Every one keeps the `[NAME]` / `[EMAIL]` placeholders, so identity is inserted
afterwards and the same career can be run under all four name groups.

The structured career underneath each one is printed first, so you can check
that the prose invents nothing: same employers, same dates, same salaries.

---

## 1. Early career — persona 9, 2021

```
Bachelor's Degree in Computer Science, University of Connecticut, 2017
2017–2020  Junior Software Engineer, NavitsPartners, Old Westbury NY   $88,400
2020–now   Software Engineer, Acuity Brands, Melville NY              $118,700
Traits: 26, female, high neuroticism, always punctual, plays sports, crafts,
humorous, tech-savvy, prior occupation computer systems analyst
```

```
[NAME]
Melville, NY · [EMAIL]

SOFTWARE ENGINEER

Software engineer with four years of experience across embedded controls and
web services. Known for shipping on schedule and for keeping the on-call
runbook something other people can actually follow.

EXPERIENCE

Software Engineer — Acuity Brands, Melville, NY
2020 – Present · Annual salary $118,700
· Own the device-provisioning service used by the connected lighting product
  line, from build through deployment and on-call rotation.
· Cut provisioning failures by 31% by adding retry logic and a dead-letter
  queue, eliminating the most common weekly support escalation.
· Wrote the integration test suite now gating every release of the service.

Junior Software Engineer — NavitsPartners, Old Westbury, NY
2017 – 2020 · Annual salary $88,400
· Built REST endpoints and internal tooling for a client-facing asset tracking
  platform, working under the direction of two senior engineers.
· Migrated a nightly reporting job from cron to a managed scheduler, removing a
  recurring source of silent failures.

EDUCATION

B.S. Computer Science, University of Connecticut, 2017

SKILLS
Python · Java · REST APIs · PostgreSQL · Docker · CI/CD · unit and integration
testing

INTERESTS
Recreational soccer league; hand-bookbinding and paper craft.
```

---

## 2. Senior — persona 107, 2021

```
Master's Degree in Electrical Engineering, CUNY City College, 2007
2007–2012  Infrastructure Engineer, OptimHire, Franklin Lakes NJ      $103,700
2012–2017  Senior Software Engineer, OptimHire, Buffalo NY            $155,900
2017–2020  Staff Infrastructure Engineer, Skynight Solutions, NYC     $221,500
2020–now   Staff Software Engineer, DocuSign, Albany NY               $250,600
Traits: 38, male, high agreeableness, high conscientiousness, introverted,
reads, crafts, open communicator, tech-savvy, prior occupation life scientist
```

```
[NAME]
Albany, NY · [EMAIL]

STAFF SOFTWARE ENGINEER

Staff engineer with fourteen years in infrastructure and platform work. Spent
the first half of my career building the systems and the second half making
them something a large organisation can run without heroics.

EXPERIENCE

Staff Software Engineer — DocuSign, Albany, NY
2020 – Present · Annual salary $250,600
· Set technical direction for the document-processing platform across six
  teams, including the design review process now used org-wide.
· Led the migration from self-managed VMs to Kubernetes across 40+ services,
  completed over four quarters with no customer-facing downtime.
· Reduced platform infrastructure spend by 24% through capacity right-sizing
  and consolidating three overlapping ingestion paths.

Staff Infrastructure Engineer — Skynight Solutions, New York, NY
2017 – 2020 · Annual salary $221,500
· Owned the deployment and observability stack for the company's core product,
  supporting a tenfold increase in customer volume without new headcount.
· Authored the incident-response process and ran postmortems; median
  time-to-mitigate fell from 47 to 18 minutes.
· Mentored five engineers, two of whom were promoted to senior during my tenure.

Senior Software Engineer — OptimHire, Buffalo, NY
2012 – 2017 · Annual salary $155,900
· Designed and built the multi-tenant scheduling service underpinning the
  company's largest enterprise accounts.
· Introduced load and soak testing to the release process, catching three
  capacity regressions before they reached production.

Infrastructure Engineer — OptimHire, Franklin Lakes, NJ
2007 – 2012 · Annual salary $103,700
· Built and maintained the internal build and deploy pipeline used by the whole
  engineering team.
· Automated environment provisioning, cutting new-environment setup from two
  days to under an hour.

EDUCATION

M.S. Electrical Engineering, CUNY City College, 2007

SKILLS
Distributed systems · Kubernetes · Terraform · Go · Python · capacity planning ·
incident command · platform architecture

INTERESTS
Woodworking and furniture restoration; long-form nonfiction.
```

---

## 3. The same career, five years earlier — persona 107, 2016

Rendering the same `Career` object at an earlier year. Nothing is regenerated;
the object is truncated. This is what makes the trajectory usable — the same
person can be shown to a decision-maker in 2016 and again in 2021.

```
[NAME]
Buffalo, NY · [EMAIL]

SENIOR SOFTWARE ENGINEER

Senior engineer with nine years of experience in infrastructure and backend
services, currently focused on multi-tenant systems for enterprise customers.

EXPERIENCE

Senior Software Engineer — OptimHire, Buffalo, NY
2012 – Present · Annual salary $155,900
· Designed and built the multi-tenant scheduling service underpinning the
  company's largest enterprise accounts.
· Introduced load and soak testing to the release process, catching three
  capacity regressions before they reached production.
· Mentor two junior engineers and review designs for the scheduling team.

Infrastructure Engineer — OptimHire, Franklin Lakes, NJ
2007 – 2012 · Annual salary $103,700
· Built and maintained the internal build and deploy pipeline used by the whole
  engineering team.
· Automated environment provisioning, cutting new-environment setup from two
  days to under an hour.

EDUCATION

M.S. Electrical Engineering, CUNY City College, 2007

SKILLS
Distributed systems · Linux · Python · Bash · CI/CD · monitoring and alerting

INTERESTS
Woodworking and furniture restoration; long-form nonfiction.
```

---

## What to look at

**The placeholder design does the heavy lifting.** One generated body, four
identities inserted afterwards, so a race or gender contrast holds the entire
career fixed. That is a correspondence experiment by construction, and it is the
single most reusable idea in the FAccT appendix.

**Length grows with seniority** — 1,050 characters at four years of experience,
2,400 at fourteen. That is realistic and it is also a problem: LLM judges favour
longer inputs, and here length is *caused by* career success. It is a mediator,
not a confounder, so controlling for it would delete part of the effect being
measured. Worth deciding deliberately rather than discovering later.

**The prose adds nothing the structure did not already contain.** Same
employers, same dates, same salaries. That is the point — the structured object
stays the source of truth, and prose is one rendering of it. It means an
ablation is a flag rather than a regeneration, and offline and LLM runs stay
directly comparable.
