"""Download L2M referee clips for a sampled pilot set.

    python3 src/l2m_contact/fetch_clips.py select 2023-24 700   # all foul INC + 700 CNC fouls (stratified, seeded)
    python3 src/l2m_contact/fetch_clips.py download             # sequential, resumable, ffprobe-verified

Selection is written to pilot_manifest.csv with per-row selection probability.
"""

from __future__ import annotations

import csv
import hashlib
import json
import random
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path

from fetch_l2m_reports import UA, BACKOFF, MAX_CONSECUTIVE_BLOCKS, log

from paths import CLIPS, WORK

MANIFEST = WORK / "pilot_manifest.csv"
RESULTS = WORK / "clip_results.csv"
SEED = 20260923
DELAY = (1.0, 2.0)


def select(season: str, n_cnc: int) -> None:
    with (WORK / "events.csv").open() as f:
        rows = [r for r in csv.DictReader(f)
                if r["season"] == season and r["call_type"].startswith("Foul")
                and r["decision"] in ("CNC", "INC") and r["video_event"]]
    inc = [r for r in rows if r["decision"] == "INC"]
    cnc = [r for r in rows if r["decision"] == "CNC"]
    strata = defaultdict(list)
    for r in cnc:
        strata[r["call_type"]].append(r)
    rng = random.Random(SEED)
    picked = []
    for r in inc:
        picked.append(r | {"stratum": "INC", "sel_prob": "1.0"})
    for ct, group in sorted(strata.items()):
        k = round(n_cnc * len(group) / len(cnc))
        for r in rng.sample(group, min(k, len(group))):
            picked.append(r | {"stratum": f"CNC|{ct}", "sel_prob": f"{min(k, len(group)) / len(group):.6f}"})
    with MANIFEST.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(picked[0]))
        w.writeheader()
        w.writerows(picked)
    print(f"{season}: {len(inc)} INC + {len(picked) - len(inc)} of {len(cnc)} CNC fouls -> {MANIFEST.name}")


def probe(path: Path) -> dict:
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "stream=width,height:format=duration", "-of", "json", str(path)],
                         capture_output=True, text=True)
    try:
        d = json.loads(out.stdout)
        s = d["streams"][0]
        return {"width": s["width"], "height": s["height"], "duration": float(d["format"]["duration"])}
    except (KeyError, IndexError, ValueError, json.JSONDecodeError):
        return {}


def download() -> None:
    with MANIFEST.open() as f:
        rows = list(csv.DictReader(f))
    done = set()
    if RESULTS.exists():
        with RESULTS.open() as f:
            done = {r["clip_id"] for r in csv.DictReader(f) if r["status"] in ("ok", "missing")}
    new_file = not RESULTS.exists()
    out = RESULTS.open("a", newline="")
    w = csv.DictWriter(out, fieldnames=["clip_id", "status", "path", "bytes", "sha256",
                                        "width", "height", "duration", "http"])
    if new_file:
        w.writeheader()
    blocks = 0
    todo = [r for r in rows if f"{r['game_id']}_{r['video_event']}" not in done]
    print(f"{len(todo)} clips to fetch", flush=True)
    for i, r in enumerate(todo, 1):
        cid = f"{r['game_id']}_{r['video_event']}"
        dest = CLIPS / r["season"] / f"{cid}.mp4"
        dest.parent.mkdir(parents=True, exist_ok=True)
        status, http = "error", ""
        for attempt, wait in enumerate([0] + BACKOFF):
            if wait:
                print(f"    backoff {wait}s ({cid})", flush=True)
                time.sleep(wait + random.uniform(0, 5))
            req = urllib.request.Request(r["clip_url"], headers={
                "User-Agent": UA, "Referer": "https://official.nba.com/last-two-minute-report/"})
            try:
                with urllib.request.urlopen(req, timeout=120) as resp:
                    dest.write_bytes(resp.read())
                http, status = "200", "downloaded"
                break
            except urllib.error.HTTPError as e:
                http = str(e.code)
                log(clip=cid, status=e.code, attempt=attempt)
                # A 403 on this CDN also means "no such file"; retry once, then call it missing.
                if e.code == 404 or (e.code == 403 and attempt >= 1):
                    status = "missing"
                    break
            except (urllib.error.URLError, TimeoutError) as e:
                http = str(e)
        info = probe(dest) if status == "downloaded" else {}
        if status == "downloaded":
            ok = info and info["duration"] > 3 and info["height"] >= 360
            status = "ok" if ok else "bad_media"
            blocks = 0
        elif status == "missing":
            blocks += 1
        data = dest.read_bytes() if dest.exists() else b""
        w.writerow({"clip_id": cid, "status": status, "path": str(dest.relative_to(CLIPS.parent)) if data else "",
                    "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest() if data else "",
                    "width": info.get("width", ""), "height": info.get("height", ""),
                    "duration": info.get("duration", ""), "http": http})
        out.flush()
        if blocks >= MAX_CONSECUTIVE_BLOCKS * 5:
            sys.exit("Stopping: 20 consecutive missing/blocked clips. Check whether the CDN is blocking.")
        if i % 25 == 0:
            print(f"  {i}/{len(todo)}", flush=True)
        time.sleep(random.uniform(*DELAY))
    out.close()


if __name__ == "__main__":
    if sys.argv[1] == "select":
        select(sys.argv[2], int(sys.argv[3]))
    elif sys.argv[1] == "download":
        download()
