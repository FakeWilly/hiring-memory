# Data (not redistributed)

The code expects two files here:

| file | what | why it is not in the repository |
|---|---|---|
| `psid_small.csv` | a small extract of the Panel Study of Income Dynamics (individual-level panel: age, education, income by year) | PSID data are free but require registration and may not be redistributed. Obtain an extract at https://psidonline.isr.umich.edu and save it here; `traits.personas_from_psid` reads the standard column names (`HEAD INCOME 19`, `AGE OF INDIVIDUAL 19`, …). |
| `swe_jobs_small.csv` | ~2,000 software-engineering job postings with employer, location and salary range | scraped job-board content; used only for employer names and salary bands per level. Any postings file with columns `title, company, location, avg_amount` will do; `generate.load_jobs` filters it. |

Without `swe_jobs_small.csv`, `career.PostingPool` falls back to built-in
salary bands. Without `psid_small.csv`, personas can be sampled from the FAccT
'25 Table 3 instead (`--source table3` in `generate.py`); the twin experiments
currently assume the PSID path.

The 840 GPT-written resumes in `resumegen/out/prose/` and the personas in
`resumegen/out/personas.csv` are synthetic, but their age, education and
income-bracket traits were drawn from PSID respondents. Treat them as derived
data: keep the repository private until the PSID terms have been checked for
this use, or regenerate them from Table 3.
