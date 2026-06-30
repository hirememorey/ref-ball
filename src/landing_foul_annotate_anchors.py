"""Browser-based manual annotator for landing-foul clip temporal anchors.

The fine-tuning run collapsed to a constant predictor because the temporal
window was the entire clip (0.0,1.0), diluting the ~400ms contact window across
8-12s of footage. `landing_foul_video_finetune.resolve_window` reads per-clip
anchors from `data/processed/landing_foul_clip_anchors.json`:

    {"{game_id}_{event_id}": {"foul_frac": 0.0-1.0, "half_width": 0.15}}

This script serves a local single-page annotator (stdlib only, no deps) so you
can scrub each clip in a browser, hit "mark foul here" at the contact frame,
tune the half-width, and save. State persists continuously, so you can resume.

Usage:
    python src/landing_foul_annotate_anchors.py
    python src/landing_foul_annotate_anchors.py --port 8765 --host 127.0.0.1

Then open the printed URL in your browser. Keyboard shortcuts:
    Space  play/pause      M      mark foul at current time
    Enter  save & next     S      skip (use global window)
    N / P  next / prev     , / .  seek 2s back / forward
    [ / ]  half-width -/+  F      filter pending/done/all
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

CLIPS_DIR = config.DATA_DIR / "clips" / "landing_foul"
GROUND_TRUTH_PATH = config.DATA_DIR / "landing_foul_ground_truth.csv"
ANCHORS_PATH = config.PROCESSED_DIR / "landing_foul_clip_anchors.json"
SKIPPED_PATH = config.PROCESSED_DIR / "landing_foul_clip_anchors_skipped.json"

DEFAULT_HALF_WIDTH = 0.15
CHUNK = 65536


# ---------------------------------------------------------------------------
# State (shared across request threads)
# ---------------------------------------------------------------------------


class AnnotationState:
    """Thread-safe anchor + skip state with atomic persistence."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.anchors: dict[str, dict[str, float]] = {}
        self.skipped: set[str] = set()
        self._load()

    def _load(self) -> None:
        if ANCHORS_PATH.exists():
            with open(ANCHORS_PATH) as f:
                self.anchors = json.load(f)
            logger.info("Loaded %d existing anchors from %s", len(self.anchors), ANCHORS_PATH)
        if SKIPPED_PATH.exists():
            with open(SKIPPED_PATH) as f:
                self.skipped = set(json.load(f))
            logger.info("Loaded %d skipped keys from %s", len(self.skipped), SKIPPED_PATH)

    def mark(self, key: str, foul_frac: float, half_width: float) -> None:
        with self._lock:
            self.anchors[key] = {
                "foul_frac": float(round(foul_frac, 4)),
                "half_width": float(round(half_width, 4)),
            }
            self.skipped.discard(key)
            self._persist_locked()

    def skip(self, key: str) -> None:
        with self._lock:
            self.skipped.add(key)
            # A skip means "no anchor"; drop any stale anchor so resolve_window
            # falls back to the global window for this clip.
            self.anchors.pop(key, None)
            self._persist_locked()

    def unskip(self, key: str) -> None:
        with self._lock:
            self.skipped.discard(key)
            self._persist_skipped_locked()

    def snapshot(self) -> tuple[dict[str, dict[str, float]], set[str]]:
        with self._lock:
            return dict(self.anchors), set(self.skipped)

    def _persist_locked(self) -> None:
        ANCHORS_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = ANCHORS_PATH.with_suffix(".json.tmp")
        with open(tmp, "w") as f:
            json.dump(self.anchors, f, indent=2, sort_keys=True)
        tmp.replace(ANCHORS_PATH)
        self._persist_skipped_locked()

    def _persist_skipped_locked(self) -> None:
        tmp = SKIPPED_PATH.with_suffix(".json.tmp")
        with open(tmp, "w") as f:
            json.dump(sorted(self.skipped), f, indent=2)
        tmp.replace(SKIPPED_PATH)


STATE = AnnotationState()


# ---------------------------------------------------------------------------
# Clip list (ground truth rows that have a downloaded MP4 on disk)
# ---------------------------------------------------------------------------


def build_clip_list() -> list[dict]:
    """Return ordered list of clips that exist on disk, with GT metadata."""
    import pandas as pd

    df = pd.read_csv(GROUND_TRUTH_PATH)
    df = df[df["landing_foul"].isin(["YES", "NO"])].copy()
    df["game_id"] = df["game_id"].astype(str).str.zfill(10)
    df["event_id"] = df["event_id"].astype(int)
    df = df.sort_values(["game_id", "event_id"]).reset_index(drop=True)

    clips = []
    for _, r in df.iterrows():
        gid, eid = r["game_id"], int(r["event_id"])
        key = f"{gid}_{eid}"
        fname = f"{key}.mp4"
        path = CLIPS_DIR / fname
        if not path.exists():
            continue
        clips.append({
            "key": key,
            "game_id": gid,
            "event_id": eid,
            "filename": fname,
            "label": r["landing_foul"],
            "description": str(r.get("description", "") or ""),
            "note": _clean_note(r.get("note", "")),
            "period": str(r.get("period", "") or ""),
            "clock": str(r.get("clock", "") or ""),
        })
    return clips


def _clean_note(val) -> str:
    if val is None:
        return ""
    if isinstance(val, float):
        import math
        if math.isnan(val):
            return ""
    s = str(val)
    return "" if s.lower() == "nan" else s


CLIP_LIST = build_clip_list()
CLIP_BY_KEY = {c["key"]: c for c in CLIP_LIST}
logger.info("Annotator ready: %d clips on disk", len(CLIP_LIST))


# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------


class Handler(BaseHTTPRequestHandler):
    # Quiet the default per-request logging; we log meaningful events ourselves.
    def log_message(self, fmt, *args):  # noqa: A003
        pass

    # --- routing ---
    def do_GET(self):  # noqa: N802
        path = urlparse(self.path).path
        if path == "/" or path == "/index.html":
            self._serve_html()
        elif path == "/api/list":
            self._serve_list()
        elif path.startswith("/clips/"):
            self._serve_clip(path[len("/clips/"):])
        else:
            self.send_error(404)

    def do_POST(self):  # noqa: N802
        path = urlparse(self.path).path
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length) if length else b""
        try:
            body = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            self.send_error(400, "invalid json")
            return

        if path == "/api/mark":
            self._handle_mark(body)
        elif path == "/api/skip":
            self._handle_skip(body)
        else:
            self.send_error(404)

    # --- api ---
    def _serve_list(self):
        anchors, skipped = STATE.snapshot()
        items = []
        for c in CLIP_LIST:
            k = c["key"]
            items.append({
                **c,
                "status": "done" if k in anchors else ("skipped" if k in skipped else "pending"),
                "anchor": anchors.get(k),
            })
        payload = {
            "clips": items,
            "total": len(items),
            "n_done": len(anchors),
            "n_skipped": len(skipped),
            "default_half_width": DEFAULT_HALF_WIDTH,
        }
        self._send_json(payload)

    def _handle_mark(self, body):
        key = body.get("key")
        frac = body.get("foul_frac")
        hw = body.get("half_width", DEFAULT_HALF_WIDTH)
        if key not in CLIP_BY_KEY or frac is None:
            self.send_error(400, "missing key/foul_frac")
            return
        try:
            frac = float(frac)
            hw = float(hw)
        except (TypeError, ValueError):
            self.send_error(400, "bad numbers")
            return
        frac = max(0.0, min(1.0, frac))
        hw = max(0.02, min(0.5, hw))
        STATE.mark(key, frac, hw)
        logger.info("mark %s  foul_frac=%.3f  half_width=%.3f", key, frac, hw)
        self._send_json({"ok": True})

    def _handle_skip(self, body):
        key = body.get("key")
        if key not in CLIP_BY_KEY:
            self.send_error(400, "unknown key")
            return
        STATE.skip(key)
        logger.info("skip %s", key)
        self._send_json({"ok": True})

    # --- static ---
    def _serve_clip(self, fname):
        # Reject anything that looks pathy; only serve bare mp4 filenames.
        if "/" in fname or "\\" in fname or ".." in fname:
            self.send_error(400)
            return
        path = CLIPS_DIR / fname
        if not path.exists() or path.suffix.lower() != ".mp4":
            self.send_error(404, "clip not found")
            return
        self._send_file_range(path, "video/mp4")

    def _send_file_range(self, path: Path, content_type: str):
        size = path.stat().st_size
        range_header = self.headers.get("Range")
        start, end = 0, size - 1
        if range_header:
            m = re.match(r"bytes=(\d+)-(\d*)", range_header)
            if m:
                start = int(m.group(1))
                if m.group(2):
                    end = int(m.group(2))
        if start > end or start >= size:
            self.send_response(416)
            self.send_header("Content-Range", f"bytes */{size}")
            self.end_headers()
            return
        length = end - start + 1
        self.send_response(206)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(length))
        self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        # HEAD requests (and clients that close early) are handled by write failures.
        if self.command == "HEAD":
            return
        try:
            with open(path, "rb") as f:
                f.seek(start)
                remaining = length
                while remaining > 0:
                    chunk = f.read(min(CHUNK, remaining))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _send_json(self, payload, status=200):
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _serve_html(self):
        body = PAGE_HTML.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------


PAGE_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Landing Foul Anchor Annotator</title>
<style>
  :root { --bg:#11141a; --panel:#181c24; --panel2:#1f2530; --txt:#e6e9ef;
          --muted:#8a93a3; --accent:#5b9dff; --yes:#4caf50; --no:#ef5350;
          --border:#2a3140; --done:#3a4250; }
  * { box-sizing:border-box; }
  body { margin:0; font:14px/1.4 -apple-system,Segoe UI,Roboto,sans-serif;
         background:var(--bg); color:var(--txt); }
  header { display:flex; gap:16px; align-items:center; padding:10px 16px;
           background:var(--panel); border-bottom:1px solid var(--border); }
  header h1 { font-size:15px; margin:0; font-weight:600; }
  #progress { color:var(--muted); font-variant-numeric:tabular-nums; }
  .filters { display:flex; gap:6px; margin-left:auto; }
  .filters button { background:var(--panel2); color:var(--muted); border:1px solid var(--border);
                    border-radius:6px; padding:4px 10px; cursor:pointer; font-size:12px; }
  .filters button.active { background:var(--accent); color:#fff; border-color:var(--accent); }
  .layout { display:grid; grid-template-columns:340px 1fr; height:calc(100vh - 47px); }
  #sidebar { overflow-y:auto; background:var(--panel); border-right:1px solid var(--border); }
  .item { padding:8px 12px; border-bottom:1px solid var(--border); cursor:pointer; }
  .item:hover { background:var(--panel2); }
  .item.active { background:#23304a; border-left:3px solid var(--accent); padding-left:9px; }
  .item .row1 { display:flex; gap:6px; align-items:center; }
  .badge { font-size:10px; padding:1px 6px; border-radius:10px; font-weight:600; }
  .badge.YES { background:var(--yes); color:#fff; }
  .badge.NO  { background:var(--no);  color:#fff; }
  .badge.SK  { background:var(--muted); color:#fff; }
  .item .key { font-variant-numeric:tabular-nums; color:var(--muted); font-size:11px; }
  .item .desc { font-size:12px; color:var(--txt); margin-top:2px;
                overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
  .status-dot { width:8px; height:8px; border-radius:50%; flex:none; }
  .status-dot.pending { background:#555; }
  .status-dot.done    { background:var(--yes); }
  .status-dot.skipped { background:var(--muted); }
  main { display:flex; flex-direction:column; padding:16px; gap:12px; overflow-y:auto; }
  .clip-head { background:var(--panel); border:1px solid var(--border);
               border-radius:8px; padding:12px 14px; }
  .clip-head .desc { font-size:15px; font-weight:500; }
  .clip-head .meta { color:var(--muted); font-size:12px; margin-top:4px; }
  .clip-head .note { margin-top:6px; color:#d8b25e; font-size:12px; }
  video { width:100%; max-height:62vh; background:#000; border-radius:8px; }
  .controls { display:flex; flex-wrap:wrap; gap:10px; align-items:center;
              background:var(--panel); border:1px solid var(--border);
              border-radius:8px; padding:12px 14px; }
  button.act { background:var(--accent); color:#fff; border:0; border-radius:6px;
               padding:8px 14px; cursor:pointer; font-size:13px; font-weight:500; }
  button.act.secondary { background:var(--panel2); color:var(--txt); border:1px solid var(--border); }
  button.act:disabled { opacity:.4; cursor:default; }
  .hw-wrap { display:flex; align-items:center; gap:8px; }
  .hw-wrap input { width:140px; }
  .anchor-info { color:var(--muted); font-size:12px; font-variant-numeric:tabular-nums; }
  .anchor-info b { color:var(--txt); }
  .kbd { color:var(--muted); font-size:11px; }
  .empty { padding:40px; text-align:center; color:var(--muted); }
</style>
</head>
<body>
<header>
  <h1>Landing Foul Anchor Annotator</h1>
  <div id="progress">loading…</div>
  <div class="filters">
    <button data-filter="pending">Pending</button>
    <button data-filter="done">Done</button>
    <button data-filter="skipped">Skipped</button>
    <button data-filter="all" class="active">All</button>
  </div>
</header>
<div class="layout">
  <div id="sidebar"></div>
  <main>
    <div id="empty" class="empty">Loading clips…</div>
    <div id="work" style="display:none">
      <div class="clip-head">
        <div id="c-desc" class="desc"></div>
        <div id="c-meta" class="meta"></div>
        <div id="c-note" class="note" style="display:none"></div>
      </div>
      <video id="v" controls preload="metadata"></video>
      <div class="controls">
        <button class="act" id="btn-mark">Mark foul here (M)</button>
        <div class="hw-wrap">
          <label>half-width <span id="hw-val">0.15</span></label>
          <input type="range" id="hw" min="0.05" max="0.40" step="0.01" value="0.15">
        </div>
        <button class="act" id="btn-save">Save &amp; next (Enter)</button>
        <button class="act secondary" id="btn-skip">Skip (S)</button>
        <button class="act secondary" id="btn-prev">Prev (P)</button>
        <button class="act secondary" id="btn-next">Next (N)</button>
        <div class="anchor-info" id="anchor-info"></div>
      </div>
      <div class="kbd">Shortcuts: Space play/pause · M mark · Enter save&amp;next · S skip · N/P next/prev · ,/. seek 2s · [/] half-width · F filter</div>
    </div>
  </main>
</div>

<script>
const $ = s => document.querySelector(s);
const video = $('#v');
let clips = [];
let idx = 0;
let filter = 'all';
let pendingFrac = null;   // mark set by user, not yet saved
let defaultHW = 0.15;

async function loadList() {
  const r = await fetch('/api/list');
  const d = await r.json();
  clips = d.clips;
  defaultHW = d.default_half_width;
  $('#hw').value = defaultHW;
  $('#hw-val').textContent = defaultHW.toFixed(2);
  renderSidebar();
  renderProgress(d);
  // jump to first pending if any
  const firstPending = clips.findIndex(c => c.status === 'pending');
  idx = firstPending >= 0 ? firstPending : 0;
  loadClip(idx);
}

function renderProgress(d) {
  const nDone = d ? d.n_done : clips.filter(c=>c.status==='done').length;
  const nSkip = d ? d.n_skipped : clips.filter(c=>c.status==='skipped').length;
  $('#progress').textContent = `${nDone} done · ${nSkip} skipped · ${clips.length} total`;
}

function visibleIndices() {
  if (filter === 'all') return clips.map((_,i)=>i);
  return clips.map((c,i)=>i).filter(i => clips[i].status === filter);
}

function renderSidebar() {
  const sb = $('#sidebar');
  sb.innerHTML = '';
  const vis = visibleIndices();
  if (!vis.length) {
    sb.innerHTML = '<div class="empty">No clips in this filter.</div>';
    return;
  }
  for (const i of vis) {
    const c = clips[i];
    const el = document.createElement('div');
    el.className = 'item' + (i === idx ? ' active' : '');
    el.dataset.i = i;
    const badge = c.status === 'skipped'
      ? '<span class="badge SK">SKIP</span>'
      : `<span class="badge ${c.label}">${c.label}</span>`;
    el.innerHTML = `
      <div class="row1">
        <span class="status-dot ${c.status}"></span>
        ${badge}
        <span class="key">${c.key}</span>
      </div>
      <div class="desc">${escapeHtml(c.description)}</div>`;
    el.onclick = () => { idx = i; loadClip(i); };
    sb.appendChild(el);
  }
}

function escapeHtml(s) {
  return (s||'').replace(/[&<>"']/g, ch => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[ch]));
}

function loadClip(i) {
  if (i < 0 || i >= clips.length) return;
  idx = i;
  const c = clips[i];
  $('#empty').style.display = 'none';
  $('#work').style.display = '';
  $('#c-desc').textContent = c.description || c.key;
  $('#c-meta').textContent =
    `${c.label} · ${c.period} ${c.clock} · game ${c.game_id} · event ${c.event_id}`;
  const noteEl = $('#c-note');
  if (c.note) { noteEl.textContent = 'Note: ' + c.note; noteEl.style.display = ''; }
  else { noteEl.style.display = 'none'; }
  video.src = '/clips/' + encodeURIComponent(c.filename);
  // reset pending frac + hw from existing anchor if present
  pendingFrac = null;
  if (c.anchor) {
    $('#hw').value = c.anchor.half_width;
    $('#hw-val').textContent = c.anchor.half_width.toFixed(2);
  } else {
    $('#hw').value = defaultHW;
    $('#hw-val').textContent = defaultHW.toFixed(2);
  }
  updateAnchorInfo();
  renderSidebar();
  video.focus();
}

function curFrac() {
  if (!video.duration) return null;
  return video.currentTime / video.duration;
}

function markHere() {
  const f = curFrac();
  if (f === null) return;
  pendingFrac = f;
  updateAnchorInfo();
}

function hwValue() { return parseFloat($('#hw').value); }

function updateAnchorInfo() {
  const c = clips[idx];
  const frac = pendingFrac !== null ? pendingFrac : (c.anchor ? c.anchor.foul_frac : null);
  const hw = hwValue();
  if (frac === null) {
    $('#anchor-info').innerHTML = 'No mark yet. Press <b>M</b> at the contact frame.';
    return;
  }
  const lo = Math.max(0, frac - hw), hi = Math.min(1, frac + hw);
  const tag = pendingFrac !== null ? ' <i>(unsaved)</i>' : '';
  $('#anchor-info').innerHTML =
    `foul_frac <b>${frac.toFixed(3)}</b> · half_width <b>${hw.toFixed(2)}</b> → window <b>${lo.toFixed(3)}–${hi.toFixed(3)}</b>${tag}`;
}

async function saveAndNext() {
  const c = clips[idx];
  const frac = pendingFrac !== null ? pendingFrac : (c.anchor ? c.anchor.foul_frac : null);
  if (frac === null) { skip(); return; }  // nothing to save -> treat as skip
  await fetch('/api/mark', {
    method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({key:c.key, foul_frac:frac, half_width:hwValue()})
  });
  c.status = 'done'; c.anchor = {foul_frac:frac, half_width:hwValue()};
  pendingFrac = null;
  recountAndAdvance(1);
}

async function skip() {
  const c = clips[idx];
  await fetch('/api/skip', {
    method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({key:c.key})
  });
  c.status = 'skipped'; c.anchor = null;
  pendingFrac = null;
  recountAndAdvance(1);
}

function recountAndAdvance(dir) {
  // refresh counts
  const d = {n_done: clips.filter(c=>c.status==='done').length,
             n_skipped: clips.filter(c=>c.status==='skipped').length};
  renderProgress(d);
  renderSidebar();
  // advance, skipping out-of-filter items when a filter is active
  let next = idx + dir;
  if (filter !== 'all') {
    while (next >= 0 && next < clips.length && clips[next].status !== filter) next += dir;
  }
  if (next < 0) next = 0;
  if (next >= clips.length) next = clips.length - 1;
  loadClip(next);
}

function seekBy(sec) {
  if (!video.duration) return;
  video.currentTime = Math.max(0, Math.min(video.duration, video.currentTime + sec));
}

function adjustHW(delta) {
  const el = $('#hw');
  el.value = Math.max(0.05, Math.min(0.40, parseFloat(el.value) + delta)).toFixed(2);
  $('#hw-val').textContent = parseFloat(el.value).toFixed(2);
  updateAnchorInfo();
}

document.addEventListener('keydown', e => {
  // ignore when typing in a range slider that's actually an input
  const tag = e.target.tagName;
  if (e.metaKey || e.ctrlKey || e.altKey) return;
  switch (e.key) {
    case ' ': e.preventDefault(); video.paused ? video.play() : video.pause(); break;
    case 'm': case 'M': markHere(); break;
    case 'Enter': e.preventDefault(); saveAndNext(); break;
    case 's': case 'S': skip(); break;
    case 'n': case 'N': recountAndAdvance(1); break;
    case 'p': case 'P': recountAndAdvance(-1); break;
    case ',': seekBy(-2); break;
    case '.': seekBy(2); break;
    case '[': adjustHW(-0.01); break;
    case ']': adjustHW(0.01); break;
    case 'f': case 'F': cycleFilter(); break;
  }
});

function cycleFilter() {
  const order = ['all','pending','done','skipped'];
  const cur = order.indexOf(filter);
  filter = order[(cur+1) % order.length];
  document.querySelectorAll('.filters button').forEach(b =>
    b.classList.toggle('active', b.dataset.filter === filter));
  renderSidebar();
}

document.querySelectorAll('.filters button').forEach(b =>
  b.onclick = () => {
    filter = b.dataset.filter;
    document.querySelectorAll('.filters button').forEach(x => x.classList.toggle('active', x===b));
    renderSidebar();
  });

$('#btn-mark').onclick = markHere;
$('#btn-save').onclick = saveAndNext;
$('#btn-skip').onclick = skip;
$('#btn-prev').onclick = () => recountAndAdvance(-1);
$('#btn-next').onclick = () => recountAndAdvance(1);
$('#hw').oninput = () => { $('#hw-val').textContent = parseFloat($('#hw').value).toFixed(2); updateAnchorInfo(); };

video.addEventListener('loadedmetadata', updateAnchorInfo);

loadList();
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(description="Browser anchor annotator for landing foul clips")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    args = p.parse_args()

    if not CLIP_LIST:
        raise SystemExit("No clips found on disk. Run landing_foul_video_dataset.py download first.")

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    url = f"http://{args.host}:{args.port}"
    print(f"\n  Landing foul anchor annotator running.\n  Open {url} in your browser.\n")
    print(f"  {len(CLIP_LIST)} clips to annotate. State saves to:\n    {ANCHORS_PATH}")
    print("  Press Ctrl+C to stop.\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping. Your anchors are already saved.")
        server.shutdown()


if __name__ == "__main__":
    main()
