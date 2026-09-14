"""
Export the verbatim call log for sharing.

`out/prompt_log.jsonl` grows with every run and re-logs cached calls, so it is
too large for git. This writes `out/prompt_log.jsonl.gz`: one record per
distinct call (first occurrence by call_id), gzipped, with the same fields --
model served, parameters, both resumes, the answer, token usage, latency. That
file IS committed, so the repository carries every call that produced a
number.

    python -m resumegen.export_log
"""

from __future__ import annotations

import gzip
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")


def main():
    src = os.path.join(OUT, "prompt_log.jsonl")
    dst = os.path.join(OUT, "prompt_log.jsonl.gz")
    seen, kept, total = set(), 0, 0
    with open(src) as f, gzip.open(dst, "wt") as g:
        for line in f:
            total += 1
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            cid = rec.get("call_id")
            if cid in seen:
                continue
            seen.add(cid)
            rec.pop("cached", None)          # every kept record is the original call
            g.write(json.dumps(rec, ensure_ascii=False) + "\n")
            kept += 1
    mb = os.path.getsize(dst) / 1048576
    print(f"{total:,} log lines -> {kept:,} distinct calls -> {dst} ({mb:.1f} MB gzipped)")


if __name__ == "__main__":
    main()
