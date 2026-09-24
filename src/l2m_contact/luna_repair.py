"""Re-tag rows missing from the Luna batch-200 pass, at batch 50.

    LLM_BACKEND=codex LLM_BATCH=50 LLM_WORKERS=8 python3 src/l2m_contact/luna_repair.py
"""

import glob
import json

import llm_tag as t

assert t.BACKEND == "codex" and t.BATCH == 50, "run with LLM_BACKEND=codex LLM_BATCH=50"
done = set()
for f in glob.glob(str(t.CACHE / f"all_{t.PROMPT_HASH}_gpt-6-luna_b200_*.json")):
    done |= {x["id"] for x in json.load(open(f))}
todo = [r for r in t.event_rows() if r["id"] not in done]
print(f"{len(todo)} rows missing from the batch-200 pass")
if todo:
    t.run(todo, "repair")
