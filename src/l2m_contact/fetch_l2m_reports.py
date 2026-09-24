"""Fetch NBA Last Two Minute reports (JSON) and build an events table.

Rate limits: one request at a time, jittered delay, on-disk cache (resumable),
exponential backoff on 403/429/5xx, hard stop after repeated blocks.

    python3 src/l2m_contact/fetch_l2m_reports.py json            # all seasons, resumable
    python3 src/l2m_contact/fetch_l2m_reports.py json 2023-24    # one season
    python3 src/l2m_contact/fetch_l2m_reports.py table           # raw JSON -> events.csv, games.csv
"""

from __future__ import annotations

import csv
import html
import json
import random
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from paths import RAW, WORK

LOG = WORK / "fetch_log.jsonl"
SEASONS = [f"{y}-{str(y + 1)[2:]}" for y in range(2018, 2026)]
UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126 Safari/537.36"
DELAY = (1.5, 3.0)
BACKOFF = [30, 60, 120, 300, 600]
MAX_CONSECUTIVE_BLOCKS = 4


class Blocked(Exception):
    def __init__(self, url: str, status=None):
        super().__init__(url)
        self.status = status


def log(**row) -> None:
    row["ts"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    with LOG.open("a") as f:
        f.write(json.dumps(row) + "\n")


def get(url: str, referer: str, backoff: list[int] = BACKOFF) -> bytes:
    """GET with backoff. Raises Blocked after exhausting retries on 403/429/5xx."""
    status = None
    for attempt, wait in enumerate([0] + backoff):
        if wait:
            print(f"    backoff {wait}s ({url})", flush=True)
            time.sleep(wait + random.uniform(0, 5))
        req = urllib.request.Request(url, headers={
            "User-Agent": UA,
            "Accept": "application/json, text/html, */*",
            "Referer": referer,
        })
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            status = e.code
            log(url=url, status=e.code, attempt=attempt)
            if e.code not in (403, 429, 500, 502, 503, 504):
                raise
        except (urllib.error.URLError, TimeoutError) as e:
            log(url=url, status=str(e), attempt=attempt)
    raise Blocked(url, status)


def discover(season: str) -> list[str]:
    cache = RAW / season / "_game_ids.json"
    if cache.exists():
        return json.loads(cache.read_text())
    url = f"https://official.nba.com/{season}-nba-officiating-last-two-minute-reports/"
    text = get(url, "https://official.nba.com/").decode("utf-8", "replace")
    ids = list(dict.fromkeys(re.findall(r"L2MReport\.html\?gameId=(\d{10})", text)))
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(ids))
    return ids


def fetch_json(seasons: list[str]) -> None:
    blocks = 0
    for season in seasons:
        ids = discover(season)
        out = RAW / season
        todo = [g for g in ids if not (out / f"{g}.json").exists()]
        print(f"{season}: {len(ids)} games, {len(todo)} to fetch", flush=True)
        for i, gid in enumerate(todo, 1):
            try:
                body = get(f"https://official.nba.com/l2m/json/{gid}.json",
                           f"https://official.nba.com/l2m/L2MReport.html?gameId={gid}")
                json.loads(body)  # validate before caching
                (out / f"{gid}.json").write_bytes(body)
                blocks = 0
            except Blocked:
                blocks += 1
                print(f"  blocked on {gid} ({blocks}/{MAX_CONSECUTIVE_BLOCKS})", flush=True)
                if blocks >= MAX_CONSECUTIVE_BLOCKS:
                    sys.exit("Stopping: repeated blocks. Rerun later; cache resumes.")
            except (urllib.error.HTTPError, json.JSONDecodeError) as e:
                log(gid=gid, error=str(e))
                print(f"  skip {gid}: {e}", flush=True)
            if i % 50 == 0:
                print(f"  {season} {i}/{len(todo)}", flush=True)
            time.sleep(random.uniform(*DELAY))


def clean(v):
    return html.unescape(str(v)).strip() if v is not None else ""


def build_table() -> None:
    events, games = [], []
    for season in SEASONS:
        for p in sorted((RAW / season).glob("00*.json")):
            d = json.loads(p.read_text())
            g = (d.get("game") or [{}])[0]
            gid = p.stem
            games.append({"season": season, "game_id": gid, "game_date": g.get("GameDate"),
                          "home": g.get("Home_team_abbr"), "away": g.get("Away_team_abbr"),
                          "home_score": g.get("HomeTeamScore"), "away_score": g.get("VisitorTeamScore"),
                          "n_events": len(d.get("l2m") or [])})
            for i, e in enumerate(d.get("l2m") or []):
                ev = clean(e.get("VideolLink"))
                events.append({
                    "season": season, "game_id": gid, "event_index": i,
                    "period": e.get("PeriodName"), "pc_time": e.get("PCTime"),
                    "call_type": clean(e.get("CallType")), "decision": clean(e.get("CallRatingName")).upper(),
                    "committing": clean(e.get("CP")), "disadvantaged": clean(e.get("DP")),
                    "difficulty": e.get("Difficulty"), "comment": clean(e.get("Comment")),
                    "pos_start": e.get("posStart"), "pos_end": e.get("posEnd"),
                    "pos_team_id": e.get("posTeamId"), "team_in_favor": e.get("teamIdInFavor"),
                    "video_event": ev,
                    "clip_url": (f"https://ak-static.cms.nba.com/wp-content/uploads/referee-clips/"
                                 f"{gid}_{ev}_DF%20BCAST_1509kbps.mp4") if ev else "",
                })
    for name, rows in (("events.csv", events), ("games.csv", games)):
        with (WORK / name).open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    print(f"{len(games)} games, {len(events)} events")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "json"
    if cmd == "json":
        fetch_json(sys.argv[2:] or SEASONS)
    elif cmd == "table":
        build_table()
