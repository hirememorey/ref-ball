"""Fetch crews for L2M games missing from the existing crew_assignments.csv
(currently 2025-26), from the www.nba.com game page __NEXT_DATA__ JSON.
Same pacing/backoff as fetch_l2m.py. Writes crews_extra.csv (resumable).

    python3 src/l2m_contact/fetch_crews.py
"""

from __future__ import annotations

import csv
import json
import random
import re
import sys
import time
from pathlib import Path

from fetch_l2m_reports import DELAY, MAX_CONSECUTIVE_BLOCKS, RAW, Blocked, get, log

from paths import CREW as EXISTING, TRACKED

OUT = TRACKED / "crews_extra.csv"
FIELDS = ["game_id", "official_id", "official_name", "jersey_num", "order"]


def main() -> None:
    have = {r["game_id"].zfill(10) for r in csv.DictReader(EXISTING.open())}
    if OUT.exists():
        have |= {r["game_id"] for r in csv.DictReader(OUT.open())}
    games = sorted({p.stem for p in RAW.glob("*/00*.json")} - have)
    print(f"{len(games)} L2M games need crews", flush=True)
    new = not OUT.exists()
    f = OUT.open("a", newline="")
    w = csv.DictWriter(f, fieldnames=FIELDS)
    if new:
        w.writeheader()
    blocks = 0
    for i, gid in enumerate(games, 1):
        try:
            try:
                page = get(f"https://www.nba.com/game/{gid}/play-by-play", "https://www.nba.com/games",
                           backoff=[10]).decode("utf-8", "replace")
            except Blocked as e:
                if e.status and e.status >= 500:   # broken page, not a block: skip
                    log(gid=gid, error=f"page {e.status}")
                    print(f"  page error {e.status}: {gid} (skipped)", flush=True)
                    time.sleep(random.uniform(*DELAY))
                    continue
                # 403/429: retry once more on the long schedule before counting a block
                page = get(f"https://www.nba.com/game/{gid}/play-by-play", "https://www.nba.com/games").decode("utf-8", "replace")
            m = re.search(r'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>', page, re.S)
            officials = json.loads(m.group(1))["props"]["pageProps"]["game"]["officials"] if m else []
            if not officials:
                log(gid=gid, error="no officials on page")
                print(f"  no officials: {gid}", flush=True)
            for k, o in enumerate(officials, 1):
                w.writerow({"game_id": gid, "official_id": o.get("personId"), "official_name": o.get("name"),
                            "jersey_num": o.get("jerseyNum"), "order": k})
            f.flush()
            blocks = 0
        except Blocked:
            blocks += 1
            print(f"  blocked on {gid} ({blocks}/{MAX_CONSECUTIVE_BLOCKS})", flush=True)
            if blocks >= MAX_CONSECUTIVE_BLOCKS:
                sys.exit("Stopping: repeated blocks. Rerun later; output resumes.")
        except (KeyError, ValueError) as e:
            log(gid=gid, error=str(e))
            print(f"  parse error {gid}: {e}", flush=True)
        if i % 50 == 0:
            print(f"  {i}/{len(games)}", flush=True)
        time.sleep(random.uniform(*DELAY))
    f.close()


if __name__ == "__main__":
    main()
