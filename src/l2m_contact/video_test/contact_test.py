"""Contact-detection tests: can a model tell whether two players made contact, and is its
answer independent of whether a whistle was blown?

The model is asked only "did A make contact with B?", not whether it was legal. Labels come
from the league decision plus the LLM comment tags (data/l2m_contact/contact_tags.csv):

  Test A, no contact vs contact (both uncalled, so no whistle in either class):
      CNC tagged none (league: no contact)  vs  CNC tagged marginal (league: contact, no call)
  Test B, whistle bias (both classes had illegal contact per the league):
      CC tagged affecting (called)  vs  INC tagged affecting (missed)
      A good detector finds contact at the same rate in both.

Frames: the same 20-frame window as video_test.py (5.5-15 s). Sample frozen with its own seed.

VERSION=v2 (Sep 24): v1's "no contact" class mixed in comments that only say "no *illegal*
contact" (mostly screens), which allow legal contact. v2 uses on-ball shooting/personal plays
only, all seasons; negatives are CNC comments saying the defender made no contact with the
player at all (no "illegal", no body-part qualifier, no "contact occurs"); positives say contact
occurred and was marginal/incidental; and the prompt follows the ball (ball handler or shooter
vs the nearest defender).

    python3 src/l2m_contact/video_test/contact_test.py select
    python3 src/l2m_contact/video_test/contact_test.py download
    python3 src/l2m_contact/video_test/contact_test.py frames
    python3 src/l2m_contact/video_test/contact_test.py run luna
    python3 src/l2m_contact/video_test/contact_test.py score
"""

from __future__ import annotations

import csv
import json
import random
import subprocess
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

import os
import re

import video_test as v

VERSION = os.environ.get("VERSION", "v1")
SUFFIX = "" if VERSION == "v1" else f"_{VERSION}"
SEED = 20260925 if VERSION == "v1" else 20260926
N = 50
# v3 reuses the v2 sample and frames; only the prompt and answer fields change.
SAMPLE = v.TRACKED / f"contact_test{'_v2' if VERSION == 'v3' else SUFFIX}_sample.csv"
OUTS = v.WORK / "video_test" / f"contact_outputs{SUFFIX}"
SCORES = v.TRACKED / f"contact_test{SUFFIX}_scores.json"
OUTPUTS_CSV = v.TRACKED / f"contact_test{SUFFIX}_outputs.csv"

ONBALL = re.compile(r"\b(shot|shoot|shooting|layup|dunk|jump shot|drive|driving|dribbl|gather|release|attempt|"
                    r"ball handler|pass)\w*", re.I)
STRICT_NONE = re.compile(r"\b(does not make (any )?contact|doesn't make (any )?contact|avoids (making )?(any )?contact|"
                         r"no contact|without (making )?(any )?contact|does not contact)\b", re.I)
CONTAM = re.compile(r"illegal|contact occurs|absorb|brush|incidental|marginal|graz|engage|slight|minimal|hand on|"
                    r"hand-on|body contact|replay|challenge|that (affects|impacts|alters|causes)|"
                    r"(contact|avoids)[^.]{0,6}(with|to) [\w'.() -]{0,40}?(arm|hand|lower|upper|body|head|leg|wrist|"
                    r"elbow|shoulder|hip|foot|feet|half|back|side|torso|face|finger|front)", re.I)
MARGINAL_CONTACT = re.compile(r"\b(marginal|incidental|brush\w*|graz\w*)\b[^.]{0,40}contact|"
                              r"contact[^.]{0,30}\b(is|was) (marginal|incidental)", re.I)

PROMPT = """You are reviewing NBA broadcast frames for an officiating study.

The frames are in time order, about 0.5 seconds apart, from the moments around one play.
The broadcast game clock is visible in the score bug. Look at {period} {clock} on the game
clock, and at {committing} and {disadvantaged}.

Question: does {committing} make physical contact with {disadvantaged} at that moment?
Any touch counts, however light: hands, arms, bodies, feet. Contact with only the ball does
not count. Do not judge whether it was a foul or legal; only whether contact happened.

- contact: yes, no, or cannot_see (the moment or players are not visible enough)
- p_contact: your probability from 0 to 100 that contact happened
- reason: what you see, briefly"""

PROMPT_V2 = """You are reviewing NBA broadcast frames for an officiating study.

The frames are in time order, evenly spaced (about 0.25-0.5 seconds apart), from the moments
around one play. The broadcast game clock is visible in the score bug.

Follow the ball. At {period} {clock} on the game clock, {disadvantaged} is the offensive player
with the ball or shooting, and {committing} is the defender guarding that play. Find the ball
handler or shooter and the nearest opposing defender at that moment.

Question: does the defender make physical contact with the ball handler or shooter at that
moment? Any touch counts, however light: hands, arms, bodies, feet. Contact with only the ball
does not count. Do not judge whether it was a foul or legal; only whether contact happened.

- contact: yes, no, or cannot_see (the moment or players are not visible enough)
- p_contact: your probability from 0 to 100 that contact happened
- reason: what you see, briefly"""
# v3 (Sep 24): the contact point is often hidden behind a body from the broadcast angle (e.g. a
# defender in front of a shooter filmed from behind). Ask separately whether the contact point is
# visible, whether the offensive player's motion changes (judgeable even when the touch is hidden),
# and contact only when the contact point is visible.
PROMPT_V3 = """You are reviewing NBA broadcast footage for an officiating study.

The images are frames in time order, evenly spaced (about 0.25-0.5 seconds apart), from the
moments around one play. The broadcast game clock is visible in the score bug.

Follow the ball. At {period} {clock} on the game clock, {disadvantaged} is the offensive player
with the ball or shooting, and {committing} is the defender guarding that play. Find the ball
handler or shooter and the nearest opposing defender at that moment.

Answer three things separately:

1. contact_point_visible: can you actually see the place where the two players' bodies, arms or
   hands meet or come closest? Answer "hidden" if the camera angle or another body blocks it
   (for example, the defender is in front of the shooter and the camera is behind the shooter).
   Being close together is not the same as visibly touching.
2. motion_change: at that moment, does the ball handler's or shooter's motion visibly change in a
   way that suggests they were touched: a hitch or jolt in the rise, loss of balance, a change of
   path, an altered release, a lost dribble? Answer yes, no or unclear, and say what changed.
   This can be judged even when the contact point is hidden.
3. contact: yes or no only if contact_point_visible is "visible". If it is hidden, answer
   cannot_see. Any touch counts; contact with only the ball does not. Do not judge legality.

Also give p_contact (0-100), your probability that contact happened using everything you can
see, including motion change, and a brief reason."""
if VERSION == "v2":
    PROMPT = PROMPT_V2
elif VERSION == "v3":
    PROMPT = PROMPT_V3

SCHEMA = {"type": "object", "additionalProperties": False, "required": ["contact", "p_contact", "reason"],
          "properties": {"contact": {"type": "string", "enum": ["yes", "no", "cannot_see"]},
                         "p_contact": {"type": "integer", "minimum": 0, "maximum": 100},
                         "reason": {"type": "string"}}}
if VERSION == "v3":
    SCHEMA = {"type": "object", "additionalProperties": False,
              "required": ["contact_point_visible", "motion_change", "motion_detail", "contact", "p_contact", "reason"],
              "properties": {"contact_point_visible": {"type": "string", "enum": ["visible", "hidden"]},
                             "motion_change": {"type": "string", "enum": ["yes", "no", "unclear"]},
                             "motion_detail": {"type": "string"},
                             "contact": {"type": "string", "enum": ["yes", "no", "cannot_see"]},
                             "p_contact": {"type": "integer", "minimum": 0, "maximum": 100},
                             "reason": {"type": "string"}}}


def select_v2() -> None:
    tags = {(r["game_id"], r["event_index"]): r["final_level"]
            for r in csv.DictReader((v.TRACKED / "contact_tags.csv").open())}
    ev = [r for r in csv.DictReader((v.WORK / "events.csv").open())
          if r["call_type"] in ("Foul: Shooting", "Foul: Personal") and r["video_event"] and ONBALL.search(r["comment"])]
    shared = {k for k, n in Counter(v.clip_id(r) for r in ev).items() if n > 1}
    ev = [r for r in ev if v.clip_id(r) not in shared]
    for r in ev:
        r["level"] = tags.get((r["game_id"], r["event_index"]), "")
    pools = {
        ("A", "no_contact"): [r for r in ev if r["decision"] == "CNC" and STRICT_NONE.search(r["comment"])
                              and not CONTAM.search(r["comment"])],
        ("A", "contact"): [r for r in ev if r["decision"] == "CNC" and r["level"] == "marginal"
                           and MARGINAL_CONTACT.search(r["comment"])],
        ("B", "missed"): [r for r in ev if r["decision"] == "INC" and r["level"] == "affecting"],
        ("B", "called"): [r for r in ev if r["decision"] == "CC" and r["level"] == "affecting"],
    }
    rng = random.Random(SEED)
    picked = []
    for test, (a, b) in (("A", ("no_contact", "contact")), ("B", ("missed", "called"))):
        pa, pb = pools[(test, a)], pools[(test, b)]
        sa = rng.sample(pa, min(N, len(pa)))
        # match the second class to the first by season and foul type
        want = Counter((r["season"], r["call_type"]) for r in sa)
        sb = []
        for key, k in want.items():
            grp = [r for r in pb if (r["season"], r["call_type"]) == key]
            sb += rng.sample(grp, min(k, len(grp)))
        rest = [r for r in pb if r not in sb]
        sb += rng.sample(rest, len(sa) - len(sb))
        for label, grp in ((a, sa), (b, sb)):
            for r in grp:
                picked.append({"test": test, "label": label, "clip_id": v.clip_id(r), "season": r["season"],
                               "game_id": r["game_id"], "event_index": r["event_index"], "decision": r["decision"],
                               "call_type": r["call_type"], "period": r["period"], "clock": r["pc_time"],
                               "committing": r["committing"], "disadvantaged": r["disadvantaged"],
                               "clip_url": r["clip_url"]})
        print(f"test {test}: {len(sa)} {a} vs {len(sb)} {b} (pools {len(pa)} / {len(pb)})")
    rng.shuffle(picked)
    with SAMPLE.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(picked[0]))
        w.writeheader()
        w.writerows(picked)


def select() -> None:
    if SAMPLE.exists():
        print(f"{SAMPLE.name} exists; sample is frozen and not redrawn")
        return
    if VERSION == "v2":
        return select_v2()
    if SAMPLE.exists():
        print(f"{SAMPLE.name} exists; sample is frozen and not redrawn")
        return
    tags = {(r["game_id"], r["event_index"]): r["final_level"]
            for r in csv.DictReader((v.TRACKED / "contact_tags.csv").open())}
    ev = [r for r in csv.DictReader((v.WORK / "events_tagged.csv").open())
          if r["season"] == "2023-24" and r["call_type"] in v.FOUL_TYPES and r["video_event"]]
    for r in ev:
        r["level"] = tags.get((r["game_id"], r["event_index"]), "")
    shared = {k for k, n in Counter(v.clip_id(r) for r in ev).items() if n > 1}
    ev = [r for r in ev if v.clip_id(r) not in shared]
    pools = {
        ("A", "no_contact"): [r for r in ev if r["decision"] == "CNC" and r["level"] == "none"],
        ("A", "contact"): [r for r in ev if r["decision"] == "CNC" and r["level"] == "marginal"],
        ("B", "called"): [r for r in ev if r["decision"] == "CC" and r["level"] == "affecting"],
        ("B", "missed"): [r for r in ev if r["decision"] == "INC" and r["level"] == "affecting"],
    }
    rng = random.Random(SEED)
    picked = []
    for test, (a, b) in (("A", ("no_contact", "contact")), ("B", ("missed", "called"))):
        pa, pb = pools[(test, a)], pools[(test, b)]
        n = min(N, len(pa), len(pb))
        sa = rng.sample(pa, n)
        # match the second class's foul-type mix to the first where possible
        want = Counter(r["call_type"] for r in sa)
        sb = []
        for ct, k in want.items():
            grp = [r for r in pb if r["call_type"] == ct]
            sb += rng.sample(grp, min(k, len(grp)))
        rest = [r for r in pb if r not in sb]
        sb += rng.sample(rest, n - len(sb))
        for label, grp in ((a, sa), (b, sb)):
            for r in grp:
                picked.append({"test": test, "label": label, "clip_id": v.clip_id(r), "game_id": r["game_id"],
                               "event_index": r["event_index"], "decision": r["decision"],
                               "call_type": r["call_type"], "period": r["period"], "clock": r["pc_time"],
                               "committing": r["committing"], "disadvantaged": r["disadvantaged"],
                               "clip_url": r["clip_url"]})
        print(f"test {test}: {n} {a} vs {n} {b} (pools {len(pa)} / {len(pb)})")
    rng.shuffle(picked)
    with SAMPLE.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(picked[0]))
        w.writeheader()
        w.writerows(picked)


def rows() -> list[dict]:
    return list(csv.DictReader(SAMPLE.open()))


def clip_path(r: dict):
    """v1 clips are all 2023-24; v2 spans seasons, stored per season like the pilot."""
    return v.CLIP_ROOT / r.get("season", "2023-24") / f"{r['clip_id']}.mp4"


def download() -> None:
    import urllib.request
    ok = bad = 0
    for r in rows():
        dest = clip_path(r)
        if dest.exists():
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        req = urllib.request.Request(r["clip_url"], headers={"User-Agent": v.UA})
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                dest.write_bytes(resp.read())
            ok += 1
        except Exception as e:  # noqa: BLE001 - record and move on
            bad += 1
            print("fail", r["clip_id"], e, flush=True)
        time.sleep(random.uniform(1.0, 2.0))
    print(f"downloads: {ok} ok, {bad} failed", flush=True)


def frame_times(src) -> list[float]:
    """The standard 5.5-15 s window for ~20 s clips. Some older clips are shorter, so for those
    take the same relative stretch (about 27%-75% of the clip)."""
    dur = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
                                str(src)], capture_output=True, text=True).stdout.strip() or 0)
    if dur >= 16:
        return v.FRAME_TIMES
    n = len(v.FRAME_TIMES)
    return [round(dur * (0.275 + 0.475 * i / (n - 1)), 2) for i in range(n)]


def frames() -> None:
    skipped = []
    for r in rows():
        src, out = clip_path(r), v.FRAMES / r["clip_id"]
        if not src.exists() or (out.exists() and len(list(out.glob("*.jpg"))) == len(v.FRAME_TIMES)):
            continue
        out.mkdir(parents=True, exist_ok=True)
        try:
            for i, t in enumerate(frame_times(src)):
                subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", str(t), "-i", str(src), "-frames:v", "1",
                                "-vf", "scale=1280:-2", "-q:v", "3", str(out / f"f{i:02d}.jpg")], check=True)
        except subprocess.CalledProcessError:
            skipped.append(r["clip_id"])
            for f in out.glob("*.jpg"):
                f.unlink()
    print(f"frames: {len(skipped)} clips skipped {skipped}", flush=True)


def run_one(r: dict, model: str) -> dict:
    out = OUTS / model / f"{r['clip_id']}.json"
    if out.exists():
        return json.loads(out.read_text())
    imgs = sorted((v.FRAMES / r["clip_id"]).glob("*.jpg"))
    if len(imgs) != len(v.FRAME_TIMES):
        raise RuntimeError(f"missing frames for {r['clip_id']}")
    text = PROMPT.format(period=r["period"], clock=r["clock"], committing=r["committing"],
                         disadvantaged=r["disadvantaged"])
    schema = v.WORK / "video_test" / "contact_schema.json"
    schema.write_text(json.dumps(SCHEMA))
    for attempt in range(3):
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
        if res and res.get("contact") in ("yes", "no", "cannot_see"):
            out.write_text(json.dumps(res))
            return res
        print(f"  {r['clip_id']} attempt {attempt + 1} failed: {p.stderr[-200:]}", flush=True)
        time.sleep(10)
    raise RuntimeError(r["clip_id"])


GEMINI_SCHEMA = {"type": "OBJECT", "required": ["contact", "p_contact", "reason"],
                 "properties": {"contact": {"type": "STRING", "enum": ["yes", "no", "cannot_see"]},
                                "p_contact": {"type": "INTEGER"},
                                "reason": {"type": "STRING"}}}


def muted_clip(r: dict):
    """Video-only copy: in test B a whistle on the audio would reveal which plays were called."""
    out = v.WORK / "video_test" / "muted" / f"{r['clip_id']}.mp4"
    if not out.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(clip_path(r)), "-an", "-c:v", "copy", str(out)],
                       check=True)
    return out


def run_one_gemini(r: dict) -> dict:
    """Gemini native video (Vertex or Gemini API, per gemini_video.py env): whole clip, muted,
    10 fps, high media resolution, same question as the frame models."""
    import base64
    import urllib.error
    import urllib.request
    import gemini_video as g
    out = OUTS / "gemini" / f"{r['clip_id']}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        return json.loads(out.read_text())
    text = PROMPT.format(period=r["period"], clock=r["clock"], committing=r["committing"],
                         disadvantaged=r["disadvantaged"])
    text = text.replace("The frames are in time order, evenly spaced (about 0.25-0.5 seconds apart), from the moments\n"
                        "around one play. The broadcast game clock is visible in the score bug.",
                        "The video is an NBA broadcast clip around one play, with the audio removed. The broadcast\n"
                        "game clock is visible in the score bug.")
    body = {"contents": [{"role": "user", "parts": [
                {"inlineData": {"mimeType": "video/mp4", "data": base64.b64encode(muted_clip(r).read_bytes()).decode()},
                 "videoMetadata": {"fps": 10}},
                {"text": text}]}],
            "generationConfig": {"temperature": 0, "responseMimeType": "application/json",
                                 "responseSchema": GEMINI_SCHEMA, "mediaResolution": "MEDIA_RESOLUTION_HIGH"}}
    data = json.dumps(body).encode()
    for attempt in range(10):
        req = urllib.request.Request(g.URL, data=data, method="POST", headers=g.HEADERS())
        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                resp_json = json.loads(resp.read())
            res = json.loads(resp_json["candidates"][0]["content"]["parts"][0]["text"])
            if res.get("contact") in ("yes", "no", "cannot_see"):
                res["p_contact"] = max(0, min(100, int(res["p_contact"])))
                res["usage"] = resp_json.get("usageMetadata", {})
                out.write_text(json.dumps(res))
                return res
            err = f"bad answer {res}"
        except urllib.error.HTTPError as e:
            err = f"HTTP {e.code}"
            if e.code not in (429, 500, 503):
                raise RuntimeError(f"{r['clip_id']}: {err} {e.read()[:200]!r}")
        except (KeyError, IndexError, json.JSONDecodeError, OSError) as e:
            err = str(e)[:200]
        wait = min(180, 15 * 2 ** attempt) + random.uniform(0, 10)
        print(f"  {r['clip_id']} attempt {attempt + 1}: {err}; retry in {wait:.0f}s", flush=True)
        time.sleep(wait)
    raise RuntimeError(f"{r['clip_id']}: gave up")


def run(model: str) -> None:
    if model not in ("luna", "gemini"):
        raise SystemExit("models: luna, gemini")
    (OUTS / model).mkdir(parents=True, exist_ok=True)
    if model == "gemini":
        todo = [r for r in rows() if clip_path(r).exists()]
        fn, workers = (lambda r: run_one_gemini(r)), int(os.environ.get("GEMINI_WORKERS", "3"))
    else:
        todo = [r for r in rows() if len(list((v.FRAMES / r["clip_id"]).glob("*.jpg"))) == len(v.FRAME_TIMES)]
        fn, workers = (lambda r: run_one(r, model)), 6
    print(f"{len(todo)} clips for {model}", flush=True)
    done = 0
    with ThreadPoolExecutor(workers) as ex:
        for f in as_completed([ex.submit(fn, r) for r in todo]):
            try:
                f.result()
            except RuntimeError as e:
                print("  gave up:", e, flush=True)
            done += 1
            if done % 20 == 0:
                print(f"  {model}: {done}/{len(todo)}", flush=True)


def score() -> None:
    from scipy.stats import fisher_exact
    rs = rows()
    report = {}
    with OUTPUTS_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["clip_id", "test", "label", "model", "contact", "p_contact", "reason",
                                          "contact_point_visible", "motion_change", "motion_detail"], extrasaction="ignore")
        w.writeheader()
        for model in ("luna", "gemini"):
            res = {}
            for r in rs:
                path = OUTS / model / f"{r['clip_id']}.json"
                if path.exists():
                    res[r["clip_id"]] = json.loads(path.read_text())
                    x = res[r["clip_id"]]
                    w.writerow({"clip_id": r["clip_id"], "test": r["test"], "label": r["label"], "model": model,
                                **{k: x.get(k, "") for k in ("contact", "p_contact", "reason", "contact_point_visible",
                                                             "motion_change", "motion_detail")}})
            for test, (a, b) in (("A", ("no_contact", "contact")), ("B", ("missed", "called"))):
                grp = {lab: [res[r["clip_id"]] for r in rs if r["test"] == test and r["label"] == lab
                             and r["clip_id"] in res] for lab in (a, b)}
                if not grp[a] or not grp[b]:
                    continue
                counts = {lab: dict(Counter(x["contact"] for x in g)) for lab, g in grp.items()}
                yes = {lab: sum(x["contact"] == "yes" for x in g) for lab, g in grp.items()}
                n = {lab: len(g) for lab, g in grp.items()}
                entry = {"n": n, "contact_answers": counts,
                         "yes_rate": {lab: round(yes[lab] / n[lab], 3) for lab in grp}}
                if test == "A":
                    pos, neg = [x["p_contact"] for x in grp[b]], [x["p_contact"] for x in grp[a]]
                    entry["auc_contact_vs_none"] = round(v.auc(pos, neg), 3)
                    entry["fisher_p_yes_rate"] = round(fisher_exact(
                        [[yes[b], n[b] - yes[b]], [yes[a], n[a] - yes[a]]], alternative="greater")[1], 4)
                else:
                    entry["fisher_p_two_sided"] = round(fisher_exact(
                        [[yes[b], n[b] - yes[b]], [yes[a], n[a] - yes[a]]])[1], 4)
                if VERSION == "v3":
                    entry["visible_rate"] = {lab: round(sum(x["contact_point_visible"] == "visible" for x in g)
                                                        / len(g), 3) for lab, g in grp.items()}
                    entry["motion_change_yes_rate"] = {lab: round(sum(x["motion_change"] == "yes" for x in g)
                                                                  / len(g), 3) for lab, g in grp.items()}
                    vis = {lab: [x for x in g if x["contact_point_visible"] == "visible"] for lab, g in grp.items()}
                    entry["visible_n"] = {lab: len(g) for lab, g in vis.items()}
                    entry["visible_yes_rate"] = {lab: (round(sum(x["contact"] == "yes" for x in g) / len(g), 3)
                                                       if g else None) for lab, g in vis.items()}
                report[f"{model}/{test}"] = entry
    SCORES.write_text(json.dumps(report, indent=2))
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
