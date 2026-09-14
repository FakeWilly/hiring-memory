"""
The only file that talks to a language model.

Design rules, all of which exist because of something that went wrong in a
hiring-audit pipeline before:

  * The key is read from the environment (OPENAI_API_KEY). It is never a
    flag, never in a file in the repo, never printed.
  * Every call is logged verbatim -- model, parameters, full messages, raw
    response, token usage, latency -- to out/prompt_log.jsonl. A record of the prompts
    was requested; this is it, per call, not per experiment.
  * Every call is cached on a hash of (model, parameters, messages). Re-running
    an experiment costs nothing and returns the same answers. Delete the cache
    to force fresh calls.
  * A cost meter runs against a price table and refuses to start a run whose
    projection exceeds --budget. Prices are per million tokens and were
    checked on 2026-09-07 against a third-party mirror of OpenAI's page; edit
    PRICES if your dashboard disagrees.
  * --dry-run counts tokens and prints the projected cost per cell without
    sending anything.
  * Reasoning models (gpt-5.x) reject `temperature`, `logprobs` and
    `max_tokens`; the wrapper retries without them and records that it did,
    because a silently-dropped parameter is how a run ends up with no
    comparisons in it.
  * A MockLLM with the same interface lets the whole pipeline run offline with
    a KNOWN preference built in, so the analysis code can be checked against a
    truth before it ever meets a real model.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")

# ---------------------------------------------------------------------------
# Prices, USD per 1M tokens (input, output). Checked 2026-09-07.
# ---------------------------------------------------------------------------

PRICES = {
    "gpt-4o":              (2.50, 10.00),
    "gpt-4o-2024-08-06":   (2.50, 10.00),   # the FAccT '25 snapshot
    "gpt-4o-mini":         (0.15, 0.60),
    "gpt-4.1":             (2.00, 8.00),
    "gpt-4.1-mini":        (0.40, 1.60),
    "gpt-4.1-nano":        (0.10, 0.40),
    "gpt-5-mini":          (0.25, 2.00),
    "gpt-5-nano":          (0.05, 0.40),
    "gpt-5.4":             (2.50, 15.00),
    "gpt-5.4-mini":        (0.75, 4.50),
    "gpt-5.4-nano":        (0.20, 1.25),
    "gpt-5.6-luna":        (1.00, 6.00),
    "gpt-5.6-terra":       (2.50, 15.00),
    "gpt-5.6-sol":         (5.00, 30.00),
    "mock":                (0.00, 0.00),
}

REASONING_PREFIXES = ("gpt-5", "o1", "o3", "o4")


def price_for(model: str) -> tuple[float, float]:
    if model in PRICES:
        return PRICES[model]
    # dated snapshots: strip the date
    base = re.sub(r"-\d{4}-\d{2}-\d{2}$", "", model)
    if base in PRICES:
        return PRICES[base]
    raise KeyError(f"no price on file for {model!r}; add it to llm.PRICES")


def is_reasoning_model(model: str) -> bool:
    return model.startswith(REASONING_PREFIXES)


def parse_retry_after(msg: str):
    """Seconds the server asked us to wait, from messages like
    'Please try again in 270ms.' / '1.2s' / '6m12s' / '2h30m'. None if absent."""
    m = re.search(r"try again in ([0-9hms.]+)", msg)
    if not m:
        return None
    t = re.fullmatch(r"(?:(\d+)h)?(?:(\d+)m(?!s))?(?:([\d.]+)ms|([\d.]+)s)?", m.group(1).rstrip("."))
    if not t or not any(t.groups()):
        return None
    h, mi, ms, sec = t.groups()
    return ((int(h) * 3600 if h else 0) + (int(mi) * 60 if mi else 0)
            + (float(ms) / 1000 if ms else 0) + (float(sec) if sec else 0))


# ---------------------------------------------------------------------------
# Token counting (for --dry-run). tiktoken if it can load its encoding,
# otherwise a chars/4 heuristic, labelled as such.
# ---------------------------------------------------------------------------

_ENC = None
_ENC_TRIED = False


def count_tokens(text: str) -> tuple[int, bool]:
    """(tokens, exact). exact=False means the chars/4 fallback was used."""
    global _ENC, _ENC_TRIED
    if not _ENC_TRIED:
        _ENC_TRIED = True
        try:
            import tiktoken
            _ENC = tiktoken.get_encoding("o200k_base")
        except Exception:
            _ENC = None
    if _ENC is not None:
        return len(_ENC.encode(text)), True
    return max(1, math.ceil(len(text) / 4)), False


def messages_tokens(messages: list[dict]) -> tuple[int, bool]:
    n, exact = 0, True
    for m in messages:
        t, e = count_tokens(str(m.get("content", "")))
        n += t + 4                      # per-message framing overhead
        exact = exact and e
    return n + 3, exact


# ---------------------------------------------------------------------------
# Cost meter
# ---------------------------------------------------------------------------

@dataclass
class CostMeter:
    budget_usd: float = 25.0
    spent_usd: float = 0.0
    calls: int = 0
    cached: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    by_tag: dict = field(default_factory=dict)

    def add(self, model: str, prompt_tokens: int, completion_tokens: int,
            tag: str = "", was_cached: bool = False):
        pin, pout = price_for(model)
        cost = 0.0 if was_cached else (prompt_tokens * pin + completion_tokens * pout) / 1e6
        self.spent_usd += cost
        self.calls += 1
        self.cached += int(was_cached)
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens
        t = self.by_tag.setdefault(tag, {"calls": 0, "usd": 0.0})
        t["calls"] += 1
        t["usd"] += cost
        if self.spent_usd > self.budget_usd:
            raise BudgetExceeded(
                f"spent ${self.spent_usd:.2f} > budget ${self.budget_usd:.2f}; "
                f"stopping. Raise --budget to continue (cache keeps what is done).")

    def summary(self) -> str:
        return (f"{self.calls} calls ({self.cached} from cache), "
                f"{self.prompt_tokens:,} in / {self.completion_tokens:,} out, "
                f"${self.spent_usd:.2f} spent of ${self.budget_usd:.2f}")


class BudgetExceeded(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# Result object shared by the real and mock clients
# ---------------------------------------------------------------------------

@dataclass
class Reply:
    content: str
    prompt_tokens: int
    completion_tokens: int
    cached: bool
    model: str
    first_token_logprobs: Optional[dict] = None   # {token: logprob} for the first token
    finish_reason: Optional[str] = None
    dropped_params: list = field(default_factory=list)
    latency_s: float = 0.0
    call_id: str = ""


def _hash_request(model: str, params: dict, messages: list[dict]) -> str:
    blob = json.dumps({"model": model, "params": params, "messages": messages},
                      sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode()).hexdigest()[:24]


# ---------------------------------------------------------------------------
# Real client
# ---------------------------------------------------------------------------

class LLM:
    """OpenAI chat completions with logging, caching and metering."""

    def __init__(self, model: str, temperature: float = 0.0,
                 budget_usd: float = 25.0, cache_dir: Optional[str] = None,
                 log_path: Optional[str] = None, dry_run: bool = False,
                 max_retries: int = 5, cache_only: bool = False):
        self.model = model
        self.temperature = temperature
        self.dry_run = dry_run
        self.cache_only = cache_only     # answer from cache; skip (never call) anything else
        self.meter = CostMeter(budget_usd=budget_usd)
        self.cache_dir = cache_dir or os.path.join(OUT, "llm_cache")
        self.log_path = log_path or os.path.join(OUT, "prompt_log.jsonl")
        self.max_retries = max_retries
        os.makedirs(self.cache_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        self._client = None
        self.projected = []          # dry-run: list of (tag, prompt_tokens, exact)
        self._lock = threading.Lock()   # meter, log and projection are shared across workers
        self.throttle_events = 0        # 429s seen (diagnostic, printed by the heartbeat)
        self.last_throttle = ""         # the server's last rate-limit message, trimmed

    # -- key handling -------------------------------------------------------
    def _get_client(self):
        if self._client is None:
            key = os.environ.get("OPENAI_API_KEY", "")
            if not key:
                raise RuntimeError(
                    "OPENAI_API_KEY is not set. In your terminal:\n"
                    "    export OPENAI_API_KEY=sk-...   (mac/linux, this shell only)\n"
                    "The key is read from the environment and never written anywhere.")
            from openai import OpenAI
            # 60 s per request, not the SDK's 10-minute default: a laptop that
            # sleeps mid-run leaves sockets hanging, and eight hung workers
            # can stall a run for an hour. Our own loop handles the retries.
            self._client = OpenAI(api_key=key, timeout=60.0, max_retries=1)
        return self._client

    # -- the call -----------------------------------------------------------
    def chat(self, messages: list[dict], max_tokens: int = 16,
             logprobs: bool = False, top_logprobs: int = 5,
             response_format: Optional[dict] = None, tag: str = "",
             temperature: Optional[float] = None, seed: Optional[int] = 0) -> Reply:
        temp = self.temperature if temperature is None else temperature
        params = {"temperature": temp, "max_tokens": max_tokens,
                  "logprobs": logprobs, "top_logprobs": top_logprobs if logprobs else None,
                  "response_format": response_format, "seed": seed}
        h = _hash_request(self.model, params, messages)
        cache_file = os.path.join(self.cache_dir, h + ".json")

        if self.dry_run:
            n, exact = messages_tokens(messages)
            with self._lock:
                self.projected.append((tag, n, max_tokens, exact))
            return Reply(content="", prompt_tokens=n, completion_tokens=0,
                         cached=False, model=self.model, call_id=h)

        if os.path.exists(cache_file):
            with open(cache_file) as f:
                rec = json.load(f)
            r = Reply(**{k: rec[k] for k in Reply.__dataclass_fields__ if k in rec})
            r.cached = True
            with self._lock:
                self._log(messages, params, r, tag)
                self.meter.add(self.model, r.prompt_tokens, r.completion_tokens,
                               tag=tag, was_cached=True)
            return r

        if self.cache_only:
            return Reply(content="", prompt_tokens=0, completion_tokens=0, cached=False,
                         model=self.model, finish_reason="skipped", call_id=h)

        reply = self._send(messages, params, tag, h)
        tmp = cache_file + f".{threading.get_ident()}.tmp"
        with open(tmp, "w") as f:
            json.dump(reply.__dict__, f, ensure_ascii=False)
        os.replace(tmp, cache_file)
        with self._lock:
            self._log(messages, params, reply, tag)
            self.meter.add(self.model, reply.prompt_tokens, reply.completion_tokens, tag=tag)
        return reply

    def _send(self, messages, params, tag, h) -> Reply:
        client = self._get_client()
        kwargs = {"model": self.model, "messages": messages}
        dropped = []
        reasoning = is_reasoning_model(self.model)
        # reasoning models: no temperature, no logprobs, max_completion_tokens
        # must leave room for the hidden reasoning tokens or content is empty
        if reasoning:
            kwargs["max_completion_tokens"] = max(params["max_tokens"], 1024)
            dropped += ["temperature", "logprobs"]
        else:
            kwargs["temperature"] = params["temperature"]
            kwargs["max_tokens"] = params["max_tokens"]
            if params["logprobs"]:
                kwargs["logprobs"] = True
                kwargs["top_logprobs"] = params["top_logprobs"]
        if params.get("seed") is not None:
            kwargs["seed"] = params["seed"]
        if params["response_format"]:
            kwargs["response_format"] = params["response_format"]

        delay = 2.0
        attempt, drops, throttled = 0, 0, 0
        resp = None
        while resp is None:
            t0 = time.time()
            try:
                resp = client.chat.completions.create(**kwargs)
            except Exception as e:                          # noqa: BLE001
                msg = str(e)
                # rate limit: wait what the server asks for (or back off) and do not
                # spend a retry on it -- Tier-1 keys hit this constantly on gpt-4o
                if re.search(r"rate.?limit|429", msg, re.I):
                    throttled += 1
                    with self._lock:
                        self.throttle_events += 1
                        self.last_throttle = re.sub(r"\s+", " ", msg)[:160]
                    if throttled > 200:
                        raise RuntimeError(f"rate-limited 200 times in a row: {msg}") from e
                    # "try again in 270ms" / "in 1.2s" / "in 6m12s" / "in 2h30m"
                    wait = parse_retry_after(msg)
                    if wait is None or wait <= 0:
                        wait = delay
                    # a daily cap asks for hours: stop and say so instead of sleeping in silence
                    if wait > 600:
                        raise RuntimeError(f"the server asked to wait {wait/60:.0f} minutes -- a daily "
                                           f"limit, not a burst; rerun later (cache keeps progress): {msg}") from e
                    time.sleep(min(max(wait, 0.5), 30) + 0.25)
                    delay = min(delay * 1.5, 30)
                    continue
                # things that will not get better with a retry
                if re.search(r"model_not_found|does not exist|invalid_api_key|Incorrect API key|"
                             r"insufficient_quota|exceeded your current quota", msg, re.I):
                    hint = ""
                    if re.search(r"model_not_found|does not exist", msg, re.I):
                        hint = f"\n  -> the model {self.model!r} is not available to this key; try --model gpt-4o or gpt-4o-mini"
                    elif re.search(r"quota", msg, re.I):
                        hint = "\n  -> no credit left on this key's project; check Billing on platform.openai.com"
                    else:
                        hint = "\n  -> check `export OPENAI_API_KEY=...` in this terminal"
                    raise RuntimeError(msg + hint) from e
                # parameter not supported -> drop it and retry immediately
                m = re.search(r"'(\w+)' is not supported|Unsupported parameter: '(\w+)'|"
                              r"Unsupported value: '(\w+)'", msg)
                bad = next((g for g in m.groups() if g), None) if m else None
                if bad and bad in kwargs and drops < 4:
                    kwargs.pop(bad, None)
                    if bad == "logprobs":
                        kwargs.pop("top_logprobs", None)
                    if bad == "max_tokens":
                        kwargs["max_completion_tokens"] = max(params["max_tokens"], 1024)
                    dropped.append(bad)
                    drops += 1
                    continue
                attempt += 1
                if attempt >= self.max_retries:
                    raise RuntimeError(f"giving up after {attempt} attempts: {msg}") from e
                time.sleep(delay)
                delay = min(delay * 2, 30)
        latency = time.time() - t0

        choice = resp.choices[0]
        content = (choice.message.content or "").strip()
        lp = None
        if getattr(choice, "logprobs", None) and choice.logprobs and choice.logprobs.content:
            first = choice.logprobs.content[0]
            lp = {first.token: first.logprob}
            for alt in (first.top_logprobs or []):
                lp[alt.token] = alt.logprob
        usage = resp.usage
        return Reply(content=content,
                     prompt_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
                     completion_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
                     cached=False, model=resp.model or self.model,
                     first_token_logprobs=lp, finish_reason=choice.finish_reason,
                     dropped_params=sorted(set(dropped)), latency_s=round(latency, 3),
                     call_id=h)

    def _log(self, messages, params, reply: Reply, tag: str):
        rec = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "tag": tag,
               "model_requested": self.model, "model_served": reply.model,
               "params": params, "messages": messages,
               "response": reply.content, "first_token_logprobs": reply.first_token_logprobs,
               "usage": {"prompt": reply.prompt_tokens, "completion": reply.completion_tokens},
               "finish_reason": reply.finish_reason, "dropped_params": reply.dropped_params,
               "latency_s": reply.latency_s, "call_id": reply.call_id, "cached": reply.cached}
        with open(self.log_path, "a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # -- dry-run report -----------------------------------------------------
    def projected_usd(self, per_call_out: Optional[int] = None) -> float:
        """Numeric total of the dry-run projection (same assumptions as projection())."""
        pin, pout = price_for(self.model)
        tot = 0.0
        for tag, n, max_out, exact in self.projected:
            if per_call_out is not None:
                out = per_call_out
            elif is_reasoning_model(self.model):
                out = max(max_out, 1024) // 2
            else:
                out = 3 if max_out <= 16 else int(max_out * 0.6)
            tot += (n * pin + out * pout) / 1e6
        return tot

    def projection(self, per_call_out: Optional[int] = None) -> str:
        """Projected cost by tag from the dry-run token counts.

        Output tokens are assumed to be 3 per call for single-token answers,
        60% of max_tokens for JSON answers, or `per_call_out` if given.
        """
        if not self.projected:
            return "nothing projected"
        pin, pout = price_for(self.model)
        by = {}
        exact_all = True
        for tag, n, max_out, exact in self.projected:
            d = by.setdefault(tag, {"calls": 0, "in": 0, "out": 0})
            d["calls"] += 1
            d["in"] += n
            if per_call_out is not None:
                d["out"] += per_call_out
            elif is_reasoning_model(self.model):
                # hidden reasoning tokens bill as output; assume half the cap
                d["out"] += max(max_out, 1024) // 2
            else:
                d["out"] += 3 if max_out <= 16 else int(max_out * 0.6)
            exact_all = exact_all and exact
        lines = [f"projected cost on {self.model} "
                 f"({'exact tokens' if exact_all else 'APPROXIMATE: chars/4, tiktoken could not load'})"]
        tot = 0.0
        for tag, d in by.items():
            usd = (d["in"] * pin + d["out"] * pout) / 1e6
            tot += usd
            lines.append(f"  {tag:44s} {d['calls']:6d} calls  {d['in']:>10,} in  "
                         f"{d['out']:>8,} out   ${usd:7.2f}")
        lines.append(f"  {'TOTAL':44s} {sum(d['calls'] for d in by.values()):6d} calls"
                     f"{'':36s}${tot:7.2f}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Mock client -- same interface, zero cost, KNOWN behaviour
# ---------------------------------------------------------------------------

class MockLLM(LLM):
    """A fake decision-maker for testing the pipeline and the analysis.

    Behaviour (deterministic per prompt hash):
      * If the prompt shows salaries, prefer the higher-paid candidate with
        probability `salary_pref`; otherwise pick by a hash (a fair coin).
      * A position bias of `first_bias` toward Candidate 1 is mixed in, so the
        order-balancing in the analysis has something to cancel.
      * JSON prompts get a valid JSON answer; tie prompts declare EQUAL with
        probability `tie_rate`.
    The point: the analysis must recover salary_pref in `full` and ~0.5 in
    `no_salary`, and must report the position bias separately. If it does
    not, the pipeline is wrong, and we found out for free.
    """

    def __init__(self, salary_pref: float = 0.75, first_bias: float = 0.10,
                 tie_rate: float = 0.15, **kw):
        # the mock's ground truth is part of its identity: separate cache per setting
        kw.setdefault("cache_dir", os.path.join(
            OUT, f"llm_cache_mock_{salary_pref}_{first_bias}_{tie_rate}"))
        super().__init__(model="mock", **kw)
        self.salary_pref, self.first_bias, self.tie_rate = salary_pref, first_bias, tie_rate

    def _send(self, messages, params, tag, h) -> Reply:
        text = "\n".join(m["content"] for m in messages)
        u = int(hashlib.sha256((h + "u").encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
        v = int(hashlib.sha256((h + "v").encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
        # salaries visible? find the current-salary figures for each candidate
        sal = self._salaries(text)
        if sal and sal[0] != sal[1]:
            richer = 1 if sal[0] > sal[1] else 2
            pick = richer if u < self.salary_pref else 3 - richer
        else:
            pick = 1 if u < 0.5 else 2
        if v < self.first_bias:
            pick = 1
        wants_json = bool(params["response_format"]) or "JSON" in text
        allow_tie = "answer 0" in text
        if allow_tie and (u * 7 % 1) < self.tie_rate:
            answer = "0"
        else:
            answer = str(pick)
        if wants_json:
            content = json.dumps({"reasoning": "mock", "choice": answer})
        else:
            content = answer
        p_pick = 0.5 + 0.3 * (1 if pick == 1 else -1)
        lp = ({"1": math.log(p_pick), "2": math.log(1 - p_pick)}
              if params.get("logprobs") else None)
        n, _ = messages_tokens(messages)
        return Reply(content=content, prompt_tokens=n, completion_tokens=len(content) // 4 + 1,
                     cached=False, model="mock", first_token_logprobs=lp,
                     finish_reason="stop", call_id=h)

    @staticmethod
    def _salaries(text: str):
        """Current salary of Candidate 1 and 2, from the rendered resumes."""
        out = []
        parts = re.split(r"\n(?:CANDIDATE|APPLICANT|RESUME) 2\n", text)
        for block in (parts[:2] if len(parts) == 2 else [text]):
            m = re.findall(r"Present\)?[^\n]*?\$([0-9,]+)", block)
            out.append(int(m[0].replace(",", "")) if m else None)
        if len(out) == 2 and all(o is not None for o in out):
            return out
        return None


# ---------------------------------------------------------------------------

def guard_budget(dry_client: "LLM", budget_usd: float, per_call_out: Optional[int] = None) -> float:
    """Print the dry-run projection and refuse to proceed if it exceeds the budget."""
    print(dry_client.projection(per_call_out=per_call_out))
    usd = dry_client.projected_usd(per_call_out=per_call_out)
    if usd > budget_usd:
        raise SystemExit(f"\nprojected ${usd:.2f} exceeds --budget ${budget_usd:.2f}; nothing was sent. "
                         f"Raise --budget, or shrink --pairs / the plan.")
    return usd


def make_client(model: str, temperature: float = 0.0, budget_usd: float = 25.0,
                dry_run: bool = False, cache_only: bool = False, **mock_kw) -> LLM:
    if model == "mock":
        return MockLLM(temperature=temperature, dry_run=dry_run, budget_usd=budget_usd,
                       cache_only=cache_only, **mock_kw)
    return LLM(model=model, temperature=temperature, budget_usd=budget_usd, dry_run=dry_run,
               cache_only=cache_only)
