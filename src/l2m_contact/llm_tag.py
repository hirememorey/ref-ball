"""Tag L2M comments with Sonnet via the Claude Code CLI (headless, no tools).

The model sees id, call type, committing player and comment. It never sees the
league decision. Outputs are cached per batch (resumable) and validated.

    python3 src/l2m_contact/llm_tag.py dev     # human-graded audit rows -> llm_dev.csv
    python3 src/l2m_contact/llm_tag.py audit   # all 90 audit rows -> llm_audit.csv
    python3 src/l2m_contact/llm_tag.py all     # every foul CNC/INC/CC/IC row -> events_llm.csv

Backends: LLM_BACKEND=claude (Claude Code CLI, default model sonnet) or LLM_BACKEND=codex
(Codex CLI, default model gpt-6-luna). LLM_MODEL, LLM_BATCH and LLM_WORKERS override.
On a Claude subscription, LLM_MAX_FIVE_HOUR / LLM_MAX_SEVEN_DAY (0-1, default 0.85 / 0.90)
pause the run before the plan usage windows fill; set them above 1 to disable.
"""

from __future__ import annotations

import csv
import hashlib
import os
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from audit_grader import GRADES, SAMPLE, row_id

from paths import HERE, WORK

ROOT = WORK
CACHE = WORK / "llm_cache"
PROMPT = (HERE / "llm_prompt.md").read_text()
SUFFIX = ""  # set below for non-default backends
PROMPT_HASH = hashlib.sha1(PROMPT.encode()).hexdigest()[:8]  # cache invalidates on prompt edits
BACKEND = os.environ.get("LLM_BACKEND", "claude")          # "claude" or "codex"
MODEL = os.environ.get("LLM_MODEL", "sonnet" if BACKEND == "claude" else "gpt-6-luna")
BATCH = int(os.environ.get("LLM_BATCH", "40"))
if BACKEND != "claude" or MODEL != "sonnet":
    SUFFIX = f"_{MODEL}_b{BATCH}"
WORKERS = int(os.environ.get("LLM_WORKERS", "6"))
MAX_FIVE_HOUR = float(os.environ.get("LLM_MAX_FIVE_HOUR", "0.85"))
MAX_SEVEN_DAY = float(os.environ.get("LLM_MAX_SEVEN_DAY", "0.90"))
STOP = False


class UsageGuard(Exception):
    pass
LEVELS = ["none", "marginal", "affecting", "legal", "not_contact_judgment", "unclear"]
SCHEMA = json.dumps({
    "type": "object",
    "properties": {"results": {"type": "array", "items": {
        "type": "object",
        "properties": {"id": {"type": "string"}, "level": {"type": "string", "enum": LEVELS},
                       "quote": {"type": "string"}},
        "required": ["id", "level", "quote"]}}},
    "required": ["results"]})


def event_rows() -> list[dict]:
    rows = []
    for r in csv.DictReader((ROOT / "events_tagged.csv").open()):
        if r["call_type"].startswith("Foul") and r["decision"] in ("CNC", "INC", "CC", "IC"):
            r["id"] = f"{r['game_id']}_{r['event_index']}"
            rows.append(r)
    return rows


def audit_rows(graded_only: bool) -> list[dict]:
    by_key = {(r["clip_url"], r["comment"]): r for r in event_rows()}
    graded = {g["row_id"] for g in csv.DictReader(GRADES.open())} if GRADES.exists() else set()
    out = []
    for s in csv.DictReader(SAMPLE.open()):
        rid = row_id(s)
        if graded_only and rid not in graded:
            continue
        e = by_key[(s["clip_url"], s["comment"])]
        out.append(e | {"audit_id": rid})
    return out


STRICT_SCHEMA = json.dumps({
    "type": "object", "additionalProperties": False, "required": ["results"],
    "properties": {"results": {"type": "array", "items": {
        "type": "object", "additionalProperties": False, "required": ["id", "level", "quote"],
        "properties": {"id": {"type": "string"}, "level": {"type": "string", "enum": LEVELS},
                       "quote": {"type": "string"}}}}}})


def call_codex(batch: list[dict], name: str, msg: str) -> list[dict]:
    """One codex exec call. Records token usage; returns validated results or raises."""
    schema = CACHE / "codex_schema.json"
    schema.write_text(STRICT_SCHEMA)
    last = CACHE / f"{name}.last.txt"
    p = subprocess.run(
        ["codex", "exec", "-m", MODEL, "--sandbox", "read-only", "--skip-git-repo-check", "--ephemeral",
         "--output-schema", str(schema), "--json", "-o", str(last), "-"],
        input=PROMPT + "\n\n" + msg, capture_output=True, text=True, cwd="/tmp", timeout=900)
    usage = {}
    for line in p.stdout.splitlines():
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue
        if e.get("type") == "turn.completed":
            usage = e.get("usage", {})
    with (CACHE / "codex_usage.jsonl").open("a") as f:
        f.write(json.dumps({"batch": name, "rows": len(batch), "model": MODEL, **usage}) + "\n")
    try:
        return json.loads(last.read_text())["results"]
    finally:
        last.unlink(missing_ok=True)


def call(batch: list[dict], name: str) -> list[dict]:
    global STOP
    path = CACHE / f"{name}.json"
    if path.exists():
        return json.loads(path.read_text())
    if STOP:
        raise UsageGuard(name)
    payload = [{"id": r["id"], "call_type": r["call_type"], "committing": r["committing"],
                "comment": r["comment"]} for r in batch]
    msg = "Label these rows:\n" + json.dumps(payload, ensure_ascii=False)
    for attempt in range(4):
        if BACKEND == "codex":
            try:
                results = call_codex(batch, name, msg)
                got, want = {x["id"] for x in results}, {r["id"] for r in batch}
                if got == want and all(x["level"] in LEVELS for x in results):
                    path.write_text(json.dumps(results))
                    return results
                err = f"id/level mismatch: missing {len(want - got)}, extra {len(got - want)}"
            except (json.JSONDecodeError, KeyError, TypeError, FileNotFoundError, subprocess.TimeoutExpired) as e:
                err = str(e)[:300]
            print(f"  {name} attempt {attempt + 1} failed: {err}", flush=True)
            time.sleep(10 * (attempt + 1))
            continue
        p = subprocess.run(
            ["claude", "-p", "--model", MODEL, "--tools", "", "--strict-mcp-config",
             "--no-session-persistence", "--system-prompt", PROMPT,
             "--output-format", "json", "--json-schema", SCHEMA],
            input=msg, capture_output=True, text=True, cwd="/tmp", timeout=600)
        try:
            events = json.loads(p.stdout)
            events = events if isinstance(events, list) else [events]
            result = next(e for e in events if e.get("type") == "result")
            for e in events:
                if e.get("type") == "rate_limit_event":
                    info = e.get("rate_limit_info") or {}
                    with (CACHE / "rate_limits.jsonl").open("a") as f:
                        f.write(json.dumps(info) + "\n")
                    win = info.get("unifiedWindows") or {}
                    five = (win.get("five_hour") or {}).get("utilization") or 0
                    week = (win.get("seven_day") or {}).get("utilization") or 0
                    # A full plan window reports "rejected" even while extra usage covers the call;
                    # only a rejection without overage means calls are actually failing.
                    blocked = info.get("status") == "rejected" and not info.get("isUsingOverage")
                    if (blocked
                            or five >= MAX_FIVE_HOUR or week >= MAX_SEVEN_DAY):
                        STOP = True
            if result.get("is_error"):
                raise KeyError(f"is_error: {str(result)[:200]}")
            results = result["structured_output"]["results"]
            with (CACHE / "costs.jsonl").open("a") as f:
                f.write(json.dumps({"batch": name, "rows": len(batch),
                                    "cost_usd": result.get("total_cost_usd"),
                                    "models": list((result.get("modelUsage") or {}).keys())}) + "\n")
            got = {x["id"] for x in results}
            want = {r["id"] for r in batch}
            if got == want and all(x["level"] in LEVELS for x in results):
                path.write_text(json.dumps(results))
                return results
            err = f"id/level mismatch: missing {len(want - got)}, extra {len(got - want)}"
        except (json.JSONDecodeError, KeyError, TypeError, StopIteration) as e:
            err = f"{e}: {p.stdout[:200]} {p.stderr[:200]}"
        print(f"  {name} attempt {attempt + 1} failed: {err}", flush=True)
        time.sleep(10 * (attempt + 1))
    raise RuntimeError(f"{name} failed")


def run(rows: list[dict], tag: str) -> dict:
    CACHE.mkdir(exist_ok=True)
    batches = [rows[i:i + BATCH] for i in range(0, len(rows), BATCH)]
    results, failed = {}, []
    with ThreadPoolExecutor(WORKERS) as ex:
        # Cache name = content hash of the batch's row ids, so a re-run with a different row
        # list can never pick up another batch's cached answers.
        def key(b):
            return hashlib.sha1(",".join(r["id"] for r in b).encode()).hexdigest()[:10]
        futs = {ex.submit(call, b, f"{tag}_{PROMPT_HASH}{SUFFIX}_{i:05d}_{key(b)}"): i for i, b in enumerate(batches)}
        for n, f in enumerate(as_completed(futs), 1):
            try:
                for x in f.result():
                    results[x["id"]] = x
            except UsageGuard:
                failed.append("usage guard")
            except RuntimeError as e:
                failed.append(str(e))
            if n % 25 == 0 or n == len(batches):
                print(f"  {tag}: {n}/{len(batches)} batches", flush=True)
    if STOP:
        print(f"  PAUSED by usage guard (5-hour >= {MAX_FIVE_HOUR:.0%}, 7-day >= {MAX_SEVEN_DAY:.0%}, overage, or not allowed). "
              f"{len(failed)} batches not run; rerun later, cache resumes.")
    elif failed:
        print(f"  {len(failed)} batches failed; rerun to retry (cache resumes)")
    return results


def write(rows: list[dict], results: dict, path: Path, extra: list[str]) -> None:
    fields = ["id"] + extra + ["decision", "call_type", "committing", "comment", "contact_level",
                               "llm_level", "llm_quote"]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            x = results.get(r["id"], {})
            w.writerow(r | {"llm_level": x.get("level", ""), "llm_quote": x.get("quote", "")})
    print(f"wrote {path.name}: {sum(1 for r in rows if r['id'] in results)}/{len(rows)} labeled")


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode in ("dev", "audit"):
        rows = audit_rows(graded_only=(mode == "dev"))
        write(rows, run(rows, mode), ROOT / f"llm_{mode}{SUFFIX}.csv", ["audit_id"])
    elif mode == "all":
        rows = event_rows()
        write(rows, run(rows, "all"), ROOT / f"events_llm{SUFFIX}.csv", ["season", "game_id", "event_index"])
