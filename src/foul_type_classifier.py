"""Generate a self-contained HTML foul-type classification tool.

Usage:
    python src/foul_type_classifier.py --manifest data/processed/foul_type_manifest_james_harden.json

Output: output/foul_type_classifier_{player_slug}.html

Open the HTML file via a local server (not file://):
    python -m http.server 8080 --directory output
    # Then open http://localhost:8080/foul_type_classifier_james_harden.html

Taxonomy (v3):
  Mechanism  : DRV-FINISH, DRV-INIT, ARM-HOOK, CONTEST, LANDING, PUMP-JUMP,
               RIP-THRU, POST, PUTBACK, OFFBALL, TAKE, AMB
  Body Part  : HEAD, ARM, CHEST, SHOULDER, LOWER
  Timing     : BEFORE, DURING, AFTER  (drive mechanisms only)
  Severity   : STRONG, MEDIUM, MARGINAL  + optional one-sentence note
  Location   : RA, PAINT, MID, PERIM

Note: localStorage key uses prefix 'ftc_v3_' — data from the v2 classifier
(prefix 'ftc_') is not compatible and will not carry over.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Foul Type Classifier \u2014 __PLAYER_NAME__</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
html, body { height: 100%; overflow: hidden; }
body {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  background: #1a1a2e; color: #e0e0e0;
  display: flex; justify-content: center;
}
.page-wrap {
  display: flex; height: 100%; min-height: 0;
  width: var(--page-width-pct, 88%); max-width: 100%;
}
.app-shell {
  flex: 1; min-width: 0; min-height: 0;
  display: flex; flex-direction: column;
  padding: 12px 4px 12px 16px;
}
.split-col {
  flex: 1; min-height: 0;
  display: flex; flex-direction: column;
}
.top-pane {
  flex: var(--top-flex, 45) 1 0;
  min-height: 100px; min-width: 0;
  display: flex; flex-direction: column; overflow: hidden;
}
.classify-pane {
  flex: var(--bottom-flex, 55) 1 0;
  min-height: 100px; min-width: 0;
  overflow-y: auto;
}
.video-wrap {
  flex: 1 1 0; min-height: 48px;
  background: #000; border-radius: 6px;
  display: flex; align-items: center; justify-content: center;
  position: relative; overflow: hidden;
}
video { width: 100%; height: 100%; object-fit: contain; cursor: pointer; display: block; }
.video-toolbar {
  flex-shrink: 0; display: flex; align-items: center; gap: 6px;
  margin: 6px 0 8px; flex-wrap: wrap;
}
.vid-btn {
  padding: 5px 11px; background: #0f3460;
  border: 1px solid #1a4a8a; border-radius: 4px;
  color: #ddd; cursor: pointer; font-size: 12px; user-select: none;
}
.vid-btn:hover { background: #1a4a8a; color: #fff; }
.vid-btn.active { border-color: #e94560; background: #e94560; color: #fff; }
.resize-handle {
  flex-shrink: 0; background: transparent; user-select: none;
  transition: background 0.1s;
}
.resize-handle:hover, .resize-handle.dragging { background: rgba(233, 69, 96, 0.25); }
.resize-h {
  height: 8px; cursor: row-resize;
  border-top: 1px solid #0f3460; border-bottom: 1px solid #0f3460;
}
.resize-v {
  width: 10px; cursor: col-resize; flex-shrink: 0;
  border-left: 1px solid #0f3460;
}
.ctx-bar {
  flex-shrink: 0;
  font-size: 13px; color: #8899aa; margin-bottom: 8px; line-height: 1.4;
}
.ctx-bar strong { color: #fff; font-weight: 600; }
.ctx-bar .desc { color: #aab; }
.classify { background: #16213e; padding: 16px 18px; border-radius: 6px; }
.section { margin-bottom: 16px; }
.section:last-child { margin-bottom: 0; }
.section h3 {
  color: #e94560; font-size: 12px; text-transform: uppercase;
  letter-spacing: 1px; margin-bottom: 8px;
  display: flex; align-items: center; gap: 6px;
}
.section h3 .hint {
  color: #556; font-size: 11px; text-transform: none;
  letter-spacing: 0; font-weight: 400;
}
.btn-grid { display: flex; flex-wrap: wrap; gap: 8px; }
.tag-btn {
  padding: 12px 16px; background: #0f3460;
  border: 2px solid transparent; border-radius: 6px;
  cursor: pointer; font-size: 14px; color: #c0c0c0;
  transition: background 0.1s, border-color 0.1s, color 0.1s;
  white-space: nowrap; user-select: none;
}
.tag-btn:hover { background: #1a4a8a; color: #fff; }
.tag-btn.active { border-color: #e94560; background: #e94560; color: #fff; font-weight: 600; }
#timingSection.inactive { visibility: hidden; pointer-events: none; }
#timingSection.inactive .tag-btn { opacity: 0; }
.sev-note {
  width: 100%; margin-top: 8px; padding: 7px 10px;
  background: #0f3460; border: 1px solid #1a4a8a; border-radius: 4px;
  color: #e0e0e0; font-size: 12px; font-family: inherit;
}
.sev-note::placeholder { color: #4a5a7a; }
.sev-note:focus { outline: none; border-color: #e94560; }
.nav-row {
  display: flex; gap: 8px; margin-top: 16px;
  padding-top: 14px; border-top: 1px solid #0f3460; flex-wrap: wrap;
}
.nav-btn {
  padding: 8px 18px; background: #0f3460; color: #e0e0e0;
  border: 1px solid #1a4a8a; border-radius: 5px;
  cursor: pointer; font-size: 13px;
  transition: background 0.1s; user-select: none;
}
.nav-btn:hover { background: #1a4a8a; }
.nav-btn.primary {
  background: #e94560; border-color: #e94560;
  color: #fff; font-weight: 600;
}
.nav-btn.primary:hover { background: #c73652; }
.status-bar {
  flex-shrink: 0;
  display: flex; align-items: center; justify-content: space-between;
  margin-bottom: 8px; font-size: 14px;
}
.status-bar .prog-text { font-weight: 700; color: #fff; }
.progress-bar {
  flex: 1; height: 6px; background: #0f3460;
  border-radius: 3px; overflow: hidden; margin: 0 14px;
}
.progress-fill { height: 100%; background: #e94560; transition: width 0.3s; }
.tag {
  display: inline-block; padding: 2px 7px; border-radius: 3px;
  margin: 1px 2px 1px 0; font-size: 10px; font-weight: 700;
}
.tag-mech  { background: #1a4080; color: #9ce; }
.tag-body  { background: #5a3010; color: #fca; }
.tag-sev   { background: #3a1060; color: #dab; }
.tag-loc   { background: #104030; color: #8db; }
.flash { animation: flash-anim 0.5s ease-out; }
@keyframes flash-anim {
  0%   { background: #16213e; }
  35%  { background: #1c3d2a; }
  100% { background: #16213e; }
}
</style>
</head>
<body>

<div class="page-wrap" id="pageWrap">
<div class="app-shell" id="appShell">
<div class="split-col" id="splitCol">

<div class="top-pane" id="topPane">
  <div class="video-wrap">
    <video id="player" autoplay loop playsinline></video>
  </div>
  <div class="video-toolbar">
    <button class="vid-btn" id="btnPlayPause">\u23f8 Pause</button>
    <button class="vid-btn speed-btn" data-rate="0.25">0.25\u00d7</button>
    <button class="vid-btn speed-btn" data-rate="0.5">0.5\u00d7</button>
    <button class="vid-btn speed-btn active" data-rate="1">1\u00d7</button>
    <button class="vid-btn" id="btnReplay">\u21ba Replay</button>
  </div>
  <div class="ctx-bar" id="ctxBar">
    <strong id="ctxPlayer">__PLAYER_NAME__</strong>
    &middot; <span id="ctxGame">\u2014</span>
    &middot; <span id="ctxClock">\u2014</span>
    <div class="desc" id="ctxDesc">\u2014</div>
  </div>
  <div class="status-bar">
    <span class="prog-text" id="progText">0 / 0</span>
    <div class="progress-bar"><div class="progress-fill" id="progFill"></div></div>
    <span id="prevList" style="font-size:12px;color:#556"></span>
  </div>
</div>

<div class="resize-handle resize-h" id="handleH" title="Drag to resize video"></div>

<div class="classify-pane" id="classifyPane">
<div class="classify" id="classifyPanel">

    <div class="section">
      <h3>Mechanism</h3>
      <div class="btn-grid" id="mechRow"></div>
    </div>

    <div class="section">
      <h3>Body Part <span class="hint">where was contact made?</span></h3>
      <div class="btn-grid" id="bodyRow"></div>
    </div>

    <div class="section inactive" id="timingSection">
      <h3>Timing <span class="hint">when during the drive?</span></h3>
      <div class="btn-grid" id="timingRow"></div>
    </div>

    <div class="section">
      <h3>Severity</h3>
      <div class="btn-grid" id="sevRow"></div>
      <input
        type="text"
        class="sev-note"
        id="sevNote"
        placeholder="Optional: one sentence on tricky severity calls\u2026"
      />
    </div>

    <div class="section">
      <h3>Location</h3>
      <div class="btn-grid" id="locRow"></div>
    </div>

    <div class="nav-row">
      <button class="nav-btn" id="btnPrev">\u2190 Prev</button>
      <button class="nav-btn primary" id="btnNext">Next \u2192</button>
      <button class="nav-btn" id="btnExport">Export CSV</button>
      <button class="nav-btn" id="btnClear">Clear</button>
    </div>

</div>
</div>

</div>
</div>
<div class="resize-handle resize-v" id="handleV" title="Drag to resize width"></div>
</div>

<script>
const CLIPS = __MANIFEST_JSON__;

const MECH = [
  ['DRV-FINISH', 'Drive finish'],
  ['DRV-INIT',   'Drive initiate'],
  ['ARM-HOOK',   'Arm hook / lock'],
  ['CONTEST',    'Jumper contest'],
  ['LANDING',    'Landing space'],
  ['PUMP-JUMP',  'Pump-fake jump-in'],
  ['RIP-THRU',   'Rip-through'],
  ['POST',       'Post contact'],
  ['PUTBACK',    'Putback / rebound'],
  ['OFFBALL',    'Off-ball'],
  ['TAKE',       'Take foul'],
  ['AMB',        'Ambiguous']
];

const BODY = [
  ['HEAD',     'Head'],
  ['ARM',      'Arm / Hand'],
  ['CHEST',    'Chest / Body'],
  ['SHOULDER', 'Shoulder'],
  ['LOWER',    'Lower Body']
];

const TIMING = [
  ['BEFORE', 'Before shot (gather)'],
  ['DURING', 'During shot motion'],
  ['AFTER',  'Post-release']
];

const SEV = [
  ['STRONG',   'Strong'],
  ['MEDIUM',   'Medium'],
  ['MARGINAL', 'Marginal']
];

const LOC = [
  ['RA',    'Restricted Area'],
  ['PAINT', 'Paint'],
  ['MID',   'Mid-range'],
  ['PERIM', 'Perimeter']
];

const DRIVE_MECHS = new Set(['DRV-FINISH', 'DRV-INIT', 'ARM-HOOK']);

const LS_KEY = 'ftc_v3_' + CLIPS.player.replace(/\\s+/g, '_');

function loadState() {
  try { return JSON.parse(localStorage.getItem(LS_KEY)) || {}; }
  catch (e) { return {}; }
}

let state           = loadState();
let currentIdx      = state.idx || 0;
let classifications = state.cls || {};
let layout = {
  videoRatio:   state.videoRatio   != null ? state.videoRatio   : 0.45,
  pageWidthPct: state.pageWidthPct != null ? state.pageWidthPct : 88
};
let playbackRate = state.playbackRate != null ? state.playbackRate : 1;
const SPEEDS = [0.25, 0.5, 1];

function saveState() {
  localStorage.setItem(LS_KEY, JSON.stringify({
    idx: currentIdx,
    cls: classifications,
    videoRatio: layout.videoRatio,
    pageWidthPct: layout.pageWidthPct,
    playbackRate: playbackRate
  }));
}

function applyLayout() {
  var top = layout.videoRatio;
  var bot = 1 - top;
  document.documentElement.style.setProperty('--top-flex', String(top * 100));
  document.documentElement.style.setProperty('--bottom-flex', String(bot * 100));
  document.documentElement.style.setProperty('--page-width-pct', layout.pageWidthPct + '%');
}

function setupResize(handle, onDrag) {
  handle.addEventListener('mousedown', function(e) {
    e.preventDefault();
    handle.classList.add('dragging');
    function onMove(ev) { onDrag(ev); }
    function onUp() {
      handle.classList.remove('dragging');
      document.removeEventListener('mousemove', onMove);
      document.removeEventListener('mouseup', onUp);
      saveState();
    }
    document.addEventListener('mousemove', onMove);
    document.addEventListener('mouseup', onUp);
  });
}

setupResize(document.getElementById('handleH'), function(e) {
  var rect = document.getElementById('splitCol').getBoundingClientRect();
  layout.videoRatio = Math.min(0.85, Math.max(0.15, (e.clientY - rect.top) / rect.height));
  applyLayout();
});

setupResize(document.getElementById('handleV'), function(e) {
  layout.pageWidthPct = Math.min(100, Math.max(50, (e.clientX / window.innerWidth) * 100));
  applyLayout();
});

applyLayout();

const video = document.getElementById('player');

function syncPlayBtn() {
  document.getElementById('btnPlayPause').textContent = video.paused ? '\\u25b6 Play' : '\\u23f8 Pause';
}

function togglePlayPause() {
  if (video.paused) video.play().catch(function() {});
  else video.pause();
}

function setPlaybackRate(rate) {
  playbackRate = rate;
  video.playbackRate = rate;
  document.querySelectorAll('.speed-btn').forEach(function(btn) {
    btn.classList.toggle('active', parseFloat(btn.dataset.rate) === rate);
  });
  saveState();
}

function replay() {
  video.currentTime = 0;
  video.play().catch(function() {});
}

video.addEventListener('play', syncPlayBtn);
video.addEventListener('pause', syncPlayBtn);
video.addEventListener('click', togglePlayPause);

document.getElementById('btnPlayPause').addEventListener('click', function(e) {
  e.stopPropagation();
  togglePlayPause();
});
document.querySelectorAll('.speed-btn').forEach(function(btn) {
  btn.addEventListener('click', function(e) {
    e.stopPropagation();
    setPlaybackRate(parseFloat(btn.dataset.rate));
  });
});
document.getElementById('btnReplay').addEventListener('click', function(e) {
  e.stopPropagation();
  replay();
});

document.addEventListener('keydown', function(e) {
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
  if (e.code === 'Space') {
    e.preventDefault();
    togglePlayPause();
  } else if (e.key === 'r' || e.key === 'R') {
    replay();
  } else if (e.key === '[') {
    var i = SPEEDS.indexOf(playbackRate);
    if (i > 0) setPlaybackRate(SPEEDS[i - 1]);
  } else if (e.key === ']') {
    var j = SPEEDS.indexOf(playbackRate);
    if (j < SPEEDS.length - 1) setPlaybackRate(SPEEDS[j + 1]);
  }
});

setPlaybackRate(playbackRate);
syncPlayBtn();

function buildButtons(containerId, items, field) {
  const container = document.getElementById(containerId);
  container.innerHTML = '';
  items.forEach(function(item) {
    var code = item[0], label = item[1];
    var btn = document.createElement('button');
    btn.className     = 'tag-btn';
    btn.dataset.value = code;
    btn.dataset.field = field;
    btn.textContent   = label;
    btn.addEventListener('click', function() { classifyField(field, code); });
    container.appendChild(btn);
  });
}

buildButtons('mechRow',   MECH,   'mech');
buildButtons('bodyRow',   BODY,   'body');
buildButtons('timingRow', TIMING, 'timing');
buildButtons('sevRow',    SEV,    'sev');
buildButtons('locRow',    LOC,    'loc');

function clipId(idx) {
  var c = CLIPS.clips[idx];
  return c.game_id + '_' + c.event_id;
}

function isComplete(cls) {
  if (!cls || !cls.mech || !cls.body || !cls.sev || !cls.loc) return false;
  if (DRIVE_MECHS.has(cls.mech) && !cls.timing) return false;
  return true;
}

function highlightGroup(containerId, value) {
  document.getElementById(containerId).querySelectorAll('.tag-btn').forEach(function(btn) {
    btn.classList.toggle('active', btn.dataset.value === value);
  });
}

function syncUI() {
  var cid = clipId(currentIdx);
  var cls = classifications[cid] || {};
  highlightGroup('mechRow',   cls.mech);
  highlightGroup('bodyRow',   cls.body);
  highlightGroup('timingRow', cls.timing);
  highlightGroup('sevRow',    cls.sev);
  highlightGroup('locRow',    cls.loc);
  document.getElementById('timingSection').classList.toggle('inactive', !DRIVE_MECHS.has(cls.mech));
  document.getElementById('sevNote').value = cls.note || '';
}

function classifyField(field, value) {
  var cid = clipId(currentIdx);
  if (!classifications[cid]) classifications[cid] = {};
  classifications[cid][field] = value;
  classifications[cid].ts = new Date().toISOString();

  if (field === 'mech' && !DRIVE_MECHS.has(value)) {
    delete classifications[cid].timing;
  }

  saveState();
  syncUI();
  updateProgress();

  if (isComplete(classifications[cid])) {
    var panel = document.getElementById('classifyPanel');
    panel.classList.add('flash');
    setTimeout(function() {
      panel.classList.remove('flash');
      if (currentIdx < CLIPS.clips.length - 1) loadClip(currentIdx + 1);
    }, 500);
  }
}

document.getElementById('sevNote').addEventListener('input', function() {
  var cid = clipId(currentIdx);
  if (!classifications[cid]) classifications[cid] = {};
  classifications[cid].note = this.value;
  saveState();
});

function loadClip(idx) {
  if (idx < 0 || idx >= CLIPS.clips.length) return;
  currentIdx = idx;
  saveState();
  var c = CLIPS.clips[idx];
  video.src = c.video_url_960;
  video.playbackRate = playbackRate;
  video.play().catch(function() {});
  document.getElementById('ctxGame').textContent  = c.opponent || '\u2014';
  var p   = c.period || '?';
  var clk = (c.clock || '').replace('PT', '').replace('M', ':').replace('S', '');
  document.getElementById('ctxClock').textContent = 'Q' + p + '  ' + clk;
  document.getElementById('ctxDesc').textContent  = c.description || '\u2014';
  syncUI();
  updateProgress();
}

function updateProgress() {
  var total = CLIPS.clips.length;
  var done  = Object.values(classifications).filter(isComplete).length;
  document.getElementById('progFill').style.width = total ? (done / total * 100) + '%' : '0%';
  document.getElementById('progText').textContent = done + ' / ' + total;
  updatePrevList();
}

function updatePrevList() {
  var list    = document.getElementById('prevList');
  var entries = [];
  CLIPS.clips.forEach(function(c, i) {
    var cid = c.game_id + '_' + c.event_id;
    var cls = classifications[cid];
    if (cls && cls.mech) entries.push({ i: i, cls: cls });
  });
  var recent = entries.slice(-6).reverse();
  if (!recent.length) {
    list.textContent = '';
    return;
  }
  var latest = recent[0];
  var cls = latest.cls;
  list.innerHTML =
    '<span class="tag tag-mech">' + (cls.mech || '?') + '</span>' +
    (cls.body ? '<span class="tag tag-body">' + cls.body + '</span>' : '') +
    (cls.sev  ? '<span class="tag tag-sev">'  + cls.sev  + '</span>' : '') +
    (cls.loc  ? '<span class="tag tag-loc">'  + cls.loc  + '</span>' : '');
}

function clearCurrent() {
  var cid = clipId(currentIdx);
  delete classifications[cid];
  saveState();
  syncUI();
  updateProgress();
}

function exportCSV() {
  var header = ['game_id','event_id','period','clock','description','opponent',
                'mechanism','body_part','timing','severity','severity_note','location','timestamp'];
  var rows = [header.join(',')];
  CLIPS.clips.forEach(function(c) {
    var cid = c.game_id + '_' + c.event_id;
    var cls = classifications[cid];
    if (!cls || !cls.mech) return;
    rows.push([
      c.game_id, c.event_id, c.period, c.clock,
      '"' + (c.description || '').replace(/"/g, '""') + '"',
      c.opponent,
      cls.mech, cls.body || '', cls.timing || '', cls.sev || '',
      '"' + (cls.note || '').replace(/"/g, '""') + '"',
      cls.loc || '', cls.ts || ''
    ].join(','));
  });
  var csv  = rows.join('\\n');
  var blob = new Blob([csv], { type: 'text/csv' });
  var url  = URL.createObjectURL(blob);
  var a    = document.createElement('a');
  a.href = url; a.download = 'foul_type_classifications.csv';
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

document.getElementById('btnPrev').addEventListener('click',   function() { if (currentIdx > 0) loadClip(currentIdx - 1); });
document.getElementById('btnNext').addEventListener('click',   function() { if (currentIdx < CLIPS.clips.length - 1) loadClip(currentIdx + 1); });
document.getElementById('btnExport').addEventListener('click', exportCSV);
document.getElementById('btnClear').addEventListener('click',  clearCurrent);

loadClip(currentIdx);
</script>
</body>
</html>"""


def generate_classifier(manifest_path: Path) -> Path:
    """Read a manifest JSON and generate a self-contained HTML classifier."""
    with open(manifest_path) as f:
        manifest = json.load(f)

    player_name = manifest.get("player", "Unknown")
    num_clips = manifest.get("num_clips", 0)
    logger.info("Generating classifier for %s (%d clips)", player_name, num_clips)

    manifest_json = json.dumps(manifest, ensure_ascii=False)
    html = HTML_TEMPLATE.replace("__PLAYER_NAME__", player_name).replace(
        "__MANIFEST_JSON__", manifest_json
    )

    slug = config.player_slug(player_name)
    out_dir = config.OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"foul_type_classifier_{slug}.html"

    with open(out_path, "w") as f:
        f.write(html)

    logger.info("Wrote classifier to %s", out_path)
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Generate foul-type classifier HTML")
    parser.add_argument(
        "--manifest",
        required=True,
        help="Path to manifest JSON from foul_type_scraper.py",
    )
    args = parser.parse_args()

    out_path = generate_classifier(Path(args.manifest))
    print(f"\nTo view the classifier, run:")
    print(f"  python -m http.server 8080 --directory {out_path.parent}")
    print(f"  Then open http://localhost:8080/{out_path.name}")


if __name__ == "__main__":
    main()
