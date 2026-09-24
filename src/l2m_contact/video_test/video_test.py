"""Can a vision model separate league-labeled illegal vs marginal contact in L2M clips?

Two blinded tests, 2023-24 clips, labels from the league decision + hybrid comment tags:
  uncalled: INC (missed, comment says illegal) vs CNC (no-call, comment says marginal)
  called:   CC (called, comment says illegal)  vs IC (called in error, comment says not illegal)
Within each test both classes share whistle status, so the whistle cannot give the answer.

The model sees 20 frames from the middle of the clip plus foul type, the two players and
the game clock. It never sees the decision or the comment.

    python3 src/l2m_contact/video_test/video_test.py select       # frozen sample (seed below)
    python3 src/l2m_contact/video_test/video_test.py download     # fetch clips not in the pilot
    python3 src/l2m_contact/video_test/video_test.py frames       # 20 frames per clip
    python3 src/l2m_contact/video_test/video_test.py run sonnet   # or: run luna
    python3 src/l2m_contact/video_test/video_test.py score

The sample is tracked (data/l2m_contact/video_test_sample.csv) and is not redrawn if present.
"""

from __future__ import annotations

import csv
import hashlib
import json
import random
import subprocess
import sys
import time
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from paths import CLIPS as CLIP_ROOT, TRACKED, WORK  # noqa: E402

L2M = WORK
CLIPS = CLIP_ROOT / "2023-24"
FRAMES = WORK / "video_test" / "frames"
OUTS = WORK / "video_test" / "outputs"
SAMPLE = TRACKED / "video_test_sample.csv"
SCORES = TRACKED / "video_test_scores.json"
OUTPUTS_CSV = TRACKED / "video_test_outputs.csv"
SEED = 20260924
N_PER_CLASS = 50
FOUL_TYPES = {"Foul: Shooting", "Foul: Personal", "Foul: Offensive", "Foul: Loose Ball"}
FRAME_TIMES = [5.5 + 0.5 * i for i in range(20)]   # 5.5s .. 15.0s; plays sit mid-clip, not exactly centered
UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126 Safari/537.36"

PROMPT = """You are reviewing NBA broadcast frames for an officiating study.

The frames are in time order, about 0.5 seconds apart, from the moments around one play.
The broadcast game clock is visible in the score bug. The league reviewed this play for a
possible {call_type} by {committing} against {disadvantaged}, at {period} {clock} on the
game clock.

Find that moment and those two players. Judge only the physical contact by {committing}:

- none: no contact by {committing}
- marginal: contact occurs but does not affect the other player's rhythm, speed, balance,
  quickness or freedom of movement, or displace them
- illegal: contact that affects the other player's rhythm, speed, balance, quickness or
  freedom of movement, displaces them, or is otherwise foul-level
- cannot_see: the moment or players are not visible enough to judge

Also give p_illegal, your probability from 0 to 100 that the contact was illegal.
Do not guess what the officials called; judge the contact you can see."""

SCHEMA = {"type": "object", "additionalProperties": False,
          "required": ["level", "p_illegal", "reason"],
          "properties": {"level": {"type": "string", "enum": ["none", "marginal", "illegal", "cannot_see"]},
                         "p_illegal": {"type": "integer", "minimum": 0, "maximum": 100},
                         "reason": {"type": "string"}}}


def clip_id(r: dict) -> str:
    return f"{r['game_id']}_{r['video_event']}"


def select() -> None:
    if SAMPLE.exists():
        print(f"{SAMPLE.name} exists; sample is frozen and not redrawn")
        return
    tags = {(r["game_id"], r["event_index"]): r["final_level"]
            for r in csv.DictReader((L2M / "events_hybrid.csv").open())}
    ev = [r for r in csv.DictReader((L2M / "events_tagged.csv").open())
          if r["season"] == "2023-24" and r["call_type"] in FOUL_TYPES and r["video_event"]]
    for r in ev:
        r["level"] = tags.get((r["game_id"], r["event_index"]), "")
    # One label per clip: drop clips shared by several rated rows.
    shared = {k for k, n in Counter(clip_id(r) for r in ev).items() if n > 1}
    ev = [r for r in ev if clip_id(r) not in shared]
    pilot_ok = {r["clip_id"] for r in csv.DictReader((L2M / "clip_results.csv").open()) if r["status"] == "ok"}
    pools = {
        ("uncalled", "illegal"): [r for r in ev if r["decision"] == "INC" and r["level"] == "affecting"
                                  and clip_id(r) in pilot_ok],
        ("uncalled", "not_illegal"): [r for r in ev if r["decision"] == "CNC" and r["level"] == "marginal"
                                      and clip_id(r) in pilot_ok],
        ("called", "illegal"): [r for r in ev if r["decision"] == "CC" and r["level"] == "affecting"],
        ("called", "not_illegal"): [r for r in ev if r["decision"] == "IC" and r["level"] in ("marginal", "legal", "none")],
    }
    rng = random.Random(SEED)
    picked = []
    for test in ("uncalled", "called"):
        pos = pools[(test, "illegal")]
        neg = pools[(test, "not_illegal")]
        n = min(N_PER_CLASS, len(pos), len(neg))
        neg_s = rng.sample(neg, n)
        # Match the illegal class to the not-illegal class's foul-type mix where possible.
        want = Counter(r["call_type"] for r in neg_s)
        by_type = defaultdict(list)
        for r in pos:
            by_type[r["call_type"]].append(r)
        pos_s = []
        for ct, k in want.items():
            pos_s += rng.sample(by_type[ct], min(k, len(by_type[ct])))
        rest = [r for r in pos if r not in pos_s]
        pos_s += rng.sample(rest, n - len(pos_s))
        for label, grp in (("illegal", pos_s), ("not_illegal", neg_s)):
            for r in grp:
                picked.append({"test": test, "label": label, "clip_id": clip_id(r), "game_id": r["game_id"],
                               "event_index": r["event_index"], "decision": r["decision"], "call_type": r["call_type"],
                               "period": r["period"], "clock": r["pc_time"], "committing": r["committing"],
                               "disadvantaged": r["disadvantaged"], "clip_url": r["clip_url"]})
        print(f"{test}: {n} illegal vs {n} not_illegal (pools {len(pos)} / {len(neg)})")
    rng.shuffle(picked)
    with SAMPLE.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(picked[0]))
        w.writeheader()
        w.writerows(picked)


def rows() -> list[dict]:
    return list(csv.DictReader(SAMPLE.open()))


def download() -> None:
    for r in rows():
        dest = CLIPS / f"{r['clip_id']}.mp4"
        if dest.exists():
            continue
        req = urllib.request.Request(r["clip_url"], headers={"User-Agent": UA})
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                dest.write_bytes(resp.read())
            print("ok", r["clip_id"], flush=True)
        except Exception as e:  # noqa: BLE001 - record and move on
            print("fail", r["clip_id"], e, flush=True)
        time.sleep(random.uniform(1.0, 2.0))


def frames() -> None:
    for r in rows():
        src = CLIPS / f"{r['clip_id']}.mp4"
        out = FRAMES / r["clip_id"]
        if not src.exists() or (out.exists() and len(list(out.glob("*.jpg"))) == len(FRAME_TIMES)):
            continue
        out.mkdir(parents=True, exist_ok=True)
        for i, t in enumerate(FRAME_TIMES):
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", str(t), "-i", str(src), "-frames:v", "1",
                            "-vf", "scale=1280:-2", "-q:v", "3", str(out / f"f{i:02d}.jpg")], check=True)


def prompt_for(r: dict) -> str:
    return PROMPT.format(call_type=r["call_type"].replace("Foul: ", "").lower() + " foul",
                         committing=r["committing"], disadvantaged=r["disadvantaged"],
                         period=r["period"], clock=r["clock"])


def run_one(r: dict, model: str) -> dict:
    out_path = OUTS / model / f"{r['clip_id']}.json"
    if out_path.exists():
        return json.loads(out_path.read_text())
    imgs = sorted((FRAMES / r["clip_id"]).glob("*.jpg"))
    if len(imgs) != len(FRAME_TIMES):
        raise RuntimeError(f"missing frames for {r['clip_id']}")
    text = prompt_for(r)
    for attempt in range(3):
        if model == "luna":
            schema = WORK / "video_test" / "schema.json"
            schema.write_text(json.dumps(SCHEMA))
            last = OUTS / model / f"{r['clip_id']}.last"
            cmd = ["codex", "exec", "-m", "gpt-6-luna", "--sandbox", "read-only", "--skip-git-repo-check",
                   "--ephemeral", "--output-schema", str(schema), "-o", str(last)]
            for im in imgs:
                cmd += ["-i", str(im)]
            p = subprocess.run(cmd + ["-"], input=text, capture_output=True, text=True, cwd="/tmp", timeout=900)
            try:
                res = json.loads(last.read_text())
                last.unlink(missing_ok=True)
            except (FileNotFoundError, json.JSONDecodeError):
                res = None
        else:
            listing = "\n".join(str(im) for im in imgs)
            msg = text + "\n\nRead these 20 image files in order, then answer:\n" + listing
            p = subprocess.run(["claude", "-p", "--model", "sonnet", "--tools", "Read", "--allowedTools", "Read",
                                "--add-dir", str(FRAMES / r["clip_id"]), "--strict-mcp-config",
                                "--no-session-persistence", "--output-format", "json",
                                "--json-schema", json.dumps(SCHEMA)],
                               input=msg, capture_output=True, text=True, cwd="/tmp", timeout=900)
            res = None
            try:
                ev = json.loads(p.stdout)
                ev = ev if isinstance(ev, list) else [ev]
                result = next(e for e in ev if e.get("type") == "result")
                res = result.get("structured_output")
                if res is not None:
                    res["cost_usd"] = result.get("total_cost_usd")
            except (json.JSONDecodeError, StopIteration):
                pass
        if res and res.get("level") in SCHEMA["properties"]["level"]["enum"]:
            out_path.write_text(json.dumps(res))
            return res
        print(f"  {model} {r['clip_id']} attempt {attempt + 1} failed: {p.stderr[-200:]}", flush=True)
        time.sleep(10)
    raise RuntimeError(r["clip_id"])


def run(model: str) -> None:
    (OUTS / model).mkdir(parents=True, exist_ok=True)
    todo = [r for r in rows() if (FRAMES / r["clip_id"]).exists()]
    done = 0
    with ThreadPoolExecutor(6) as ex:
        futs = [ex.submit(run_one, r, model) for r in todo]
        for f in as_completed(futs):
            try:
                f.result()
            except RuntimeError as e:
                print("  gave up:", e, flush=True)
            done += 1
            if done % 20 == 0:
                print(f"  {model}: {done}/{len(todo)}", flush=True)


def auc(pos: list[float], neg: list[float]) -> float:
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def score() -> None:
    rs = rows()
    report = {}
    for model in ("sonnet", "luna", "gemini"):
        d = OUTS / model
        if not d.exists():
            continue
        for test in ("uncalled", "called"):
            res = []
            for r in rs:
                if r["test"] != test or not (d / f"{r['clip_id']}.json").exists():
                    continue
                res.append((r["label"], json.loads((d / f"{r['clip_id']}.json").read_text())))
            if not res:
                continue
            pos = [x["p_illegal"] for lab, x in res if lab == "illegal"]
            neg = [x["p_illegal"] for lab, x in res if lab == "not_illegal"]
            seen = [(lab, x) for lab, x in res if x["level"] != "cannot_see"]
            acc = sum((x["level"] == "illegal") == (lab == "illegal") for lab, x in seen) / max(len(seen), 1)
            # permutation p for AUC
            rng = random.Random(1)
            obs = auc(pos, neg)
            allp = pos + neg
            ge = 0
            for _ in range(2000):
                rng.shuffle(allp)
                ge += auc(allp[:len(pos)], allp[len(pos):]) >= obs
            levels = {lab: dict(Counter(x["level"] for l2, x in res if l2 == lab)) for lab in ("illegal", "not_illegal")}
            report[f"{model}/{test}"] = {"n": len(res), "auc": round(obs, 3), "auc_perm_p": round((ge + 1) / 2001, 4),
                                         "accuracy_excl_cannot_see": round(acc, 3),
                                         "cannot_see": sum(x["level"] == "cannot_see" for _, x in res),
                                         "levels_by_true_label": levels}
    SCORES.write_text(json.dumps(report, indent=2))
    # Tracked per-clip answers (so a fresh clone can compare models without the local outputs).
    with OUTPUTS_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["clip_id", "test", "label", "model", "level", "p_illegal", "reason"])
        w.writeheader()
        for model in ("sonnet", "luna", "gemini"):
            for r in rs:
                path = OUTS / model / f"{r['clip_id']}.json"
                if path.exists():
                    x = json.loads(path.read_text())
                    w.writerow({"clip_id": r["clip_id"], "test": r["test"], "label": r["label"], "model": model,
                                "level": x["level"], "p_illegal": x["p_illegal"], "reason": x.get("reason", "")})
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "select":
        select()
    elif cmd == "download":
        download()
    elif cmd == "frames":
        frames()
    elif cmd == "run":
        run(sys.argv[2])
    elif cmd == "score":
        score()
