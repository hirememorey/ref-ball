"""Combine Luna (all rows) with a Sonnet re-check on rows selected by recheck_rule.

Reuses any Sonnet tags already cached for the current prompt. Writes events_hybrid.csv
with the final label and which model produced it.

    python3 src/l2m_contact/hybrid.py          # default (Sonnet) backend env
    python3 src/l2m_contact/hybrid.py export   # rebuild contact_tags.csv only

Also writes the tracked tag table data/l2m_contact/contact_tags.csv (ids and labels only,
no league comment text), which analyze_crews.py reads for TAGS=hybrid.
"""

from __future__ import annotations

import csv
import glob
import json

import llm_tag as t
from recheck_rule import needs_recheck

from paths import TRACKED

ROOT = t.ROOT


def cached(pattern: str) -> dict:
    out = {}
    for f in glob.glob(str(t.CACHE / pattern)):
        for x in json.load(open(f)):
            out[x["id"]] = x
    return out


def main() -> None:
    if t.BACKEND != "claude" or t.MODEL != "sonnet" or t.BATCH != 40:
        raise SystemExit("run hybrid.py without LLM_* overrides (Sonnet, batch 40)")
    rows = t.event_rows()
    # main Luna pass (batch 200) plus a repair pass at batch 50 for truncated batches
    luna = cached(f"all_{t.PROMPT_HASH}_gpt-6-luna_b200_*.json") | cached(f"repair_{t.PROMPT_HASH}_gpt-6-luna_b50_*.json")
    missing = [r for r in rows if r["id"] not in luna]
    if missing:
        raise SystemExit(f"{len(missing)} rows lack Luna tags; finish the Luna run first")
    sel = [r for r in rows if needs_recheck(luna[r["id"]]["level"], r["comment"])]
    sonnet = cached(f"all_{t.PROMPT_HASH}_0*.json") | cached(f"recheck_{t.PROMPT_HASH}_0*.json")
    todo = [r for r in sel if r["id"] not in sonnet]
    print(f"re-check {len(sel)} rows ({100 * len(sel) / len(rows):.1f}%); {len(todo)} need new Sonnet calls")
    if todo:
        sonnet |= t.run(todo, "recheck")
    sel_ids = {r["id"] for r in sel}
    with (ROOT / "events_hybrid.csv").open("w", newline="") as f:
        fields = ["id", "season", "game_id", "event_index", "decision", "call_type", "committing", "comment",
                  "luna_level", "sonnet_level", "final_level", "final_source", "final_quote"]
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        n_missing = 0
        for r in rows:
            lu, so = luna[r["id"]], sonnet.get(r["id"])
            use_sonnet = r["id"] in sel_ids and so is not None
            n_missing += r["id"] in sel_ids and so is None
            fin = so if use_sonnet else lu
            w.writerow(r | {"luna_level": lu["level"], "sonnet_level": so["level"] if so else "",
                            "final_level": fin["level"], "final_source": "sonnet" if use_sonnet else "luna",
                            "final_quote": fin["quote"]})
    print(f"wrote events_hybrid.csv ({len(rows)} rows; {n_missing} selected rows lacked Sonnet and kept Luna)")
    export()


def export() -> None:
    """Tracked copy of the tags: ids and labels only, no league comment text or quotes."""
    cols = ["season", "game_id", "event_index", "decision", "call_type", "luna_level", "sonnet_level",
            "final_level", "final_source"]
    with (ROOT / "events_hybrid.csv").open() as f, (TRACKED / "contact_tags.csv").open("w", newline="") as g:
        w = csv.DictWriter(g, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in csv.DictReader(f):
            w.writerow(r)
    print("wrote data/l2m_contact/contact_tags.csv")


if __name__ == "__main__":
    import sys
    export() if sys.argv[1:] == ["export"] else main()
