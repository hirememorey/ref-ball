"""Run a non-default backend on rows Sonnet (current prompt) already tagged, and compare.

    LLM_BACKEND=codex LLM_BATCH=40 python3 src/l2m_contact/compare_models.py
"""

from __future__ import annotations

import collections
import glob
import json

import llm_tag as t

sonnet = {}
for f in glob.glob(str(t.CACHE / f"all_{t.PROMPT_HASH}_0*.json")):
    for x in json.load(open(f)):
        sonnet[x["id"]] = x["level"]
rows = [r for r in t.event_rows() if r["id"] in sonnet]
print(f"{len(rows)} rows with current-prompt Sonnet tags; running {t.MODEL} batch {t.BATCH}")
other = {k: v["level"] for k, v in t.run(rows, "overlap").items()}
ids = [r["id"] for r in rows if r["id"] in other]
agree = sum(sonnet[i] == other[i] for i in ids)
print(f"agreement {agree}/{len(ids)} ({100 * agree / len(ids):.1f}%)")
print("disagreements (sonnet -> other):",
      collections.Counter((sonnet[i], other[i]) for i in ids if sonnet[i] != other[i]).most_common(10))
by = {r["id"]: r for r in rows}
dec = collections.defaultdict(collections.Counter)
for i in ids:
    dec[by[i]["decision"]][other[i]] += 1
print("other model by league decision:", {d: dict(c) for d, c in dec.items()})
out = t.ROOT / f"compare_sonnet_vs{t.SUFFIX}.json"
out.write_text(json.dumps([{"id": i, "sonnet": sonnet[i], "other": other[i], "decision": by[i]["decision"],
                             "committing": by[i]["committing"], "comment": by[i]["comment"]}
                            for i in ids if sonnet[i] != other[i]], indent=1))
print("wrote", out.name)
