"""Generate a self-contained HTML landing-foul classification tool.

Usage:
    python src/landing_foul_classifier.py
    python src/landing_foul_classifier.py --manifest data/processed/landing_foul_manifest.json

Output: output/landing_foul_classifier.html

Open via local server (not file://):
    python -m http.server 8080 --directory output
    # http://localhost:8080/landing_foul_classifier.html

Classification:
    landing_foul: YES | NO | UNCLEAR
    note: optional free text
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
<title>Landing Foul Classifier</title>
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
  flex: var(--top-flex, 50) 1 0;
  min-height: 100px; min-width: 0;
  display: flex; flex-direction: column; overflow: hidden;
}
.classify-pane {
  flex: var(--bottom-flex, 50) 1 0;
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
  font-size: 13px; color: #8899aa; margin-bottom: 8px; line-height: 1.5;
}
.ctx-bar strong { color: #fff; font-weight: 600; }
.ctx-bar .desc { color: #aab; }
.rubric {
  background: #0f1a30; border: 1px solid #1a3050; border-radius: 6px;
  padding: 10px 12px; margin-bottom: 14px; font-size: 12px; line-height: 1.45; color: #9ab;
}
.rubric strong { color: #e94560; }
.classify { background: #16213e; padding: 16px 18px; border-radius: 6px; }
.section { margin-bottom: 16px; }
.section h3 {
  color: #e94560; font-size: 12px; text-transform: uppercase;
  letter-spacing: 1px; margin-bottom: 8px;
}
.btn-grid { display: flex; flex-wrap: wrap; gap: 10px; }
.tag-btn {
  padding: 14px 22px; background: #0f3460;
  border: 2px solid transparent; border-radius: 6px;
  cursor: pointer; font-size: 15px; color: #c0c0c0;
  transition: background 0.1s, border-color 0.1s, color 0.1s;
  white-space: nowrap; user-select: none; min-width: 100px;
}
.tag-btn:hover { background: #1a4a8a; color: #fff; }
.tag-btn.active { border-color: #e94560; background: #e94560; color: #fff; font-weight: 600; }
.tag-btn.yes.active { border-color: #2ecc71; background: #27ae60; }
.tag-btn.no.active { border-color: #e74c3c; background: #c0392b; }
.tag-btn.unclear.active { border-color: #f39c12; background: #d68910; }
.note-input {
  width: 100%; margin-top: 8px; padding: 8px 10px;
  background: #0f3460; border: 1px solid #1a4a8a; border-radius: 4px;
  color: #e0e0e0; font-size: 13px; font-family: inherit;
}
.note-input::placeholder { color: #4a5a7a; }
.note-input:focus { outline: none; border-color: #e94560; }
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
.counts { font-size: 12px; color: #8899aa; margin-left: 8px; }
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
    <strong id="ctxPlayer">\u2014</strong>
    &middot; <span id="ctxGame">\u2014</span>
    &middot; <span id="ctxClock">\u2014</span>
    &middot; <span id="ctxOfficial">\u2014</span>
    <div class="desc" id="ctxDesc">\u2014</div>
  </div>
  <div class="status-bar">
    <span class="prog-text" id="progText">0 / 0</span>
    <div class="progress-bar"><div class="progress-fill" id="progFill"></div></div>
    <span class="counts" id="countSummary"></span>
  </div>
</div>

<div class="resize-handle resize-h" id="handleH" title="Drag to resize video"></div>

<div class="classify-pane" id="classifyPane">
<div class="classify" id="classifyPanel">

    <div class="rubric">
      <strong>Landing foul?</strong> Defender's feet/body are under or moving into the
      shooter's landing zone while the shooter is airborne on a jump shot, and the foul
      is called because of that positioning. Standard arm/hand contest on the shot = <strong>NO</strong>.
    </div>

    <div class="section">
      <h3>Landing Foul</h3>
      <div class="btn-grid" id="landingRow">
        <button class="tag-btn yes" data-value="YES">Yes</button>
        <button class="tag-btn no" data-value="NO">No</button>
        <button class="tag-btn unclear" data-value="UNCLEAR">Unclear</button>
      </div>
      <input
        type="text"
        class="note-input"
        id="noteInput"
        placeholder="Optional note (edge case, why unclear, etc.)"
      />
    </div>

    <div class="nav-row">
      <button class="nav-btn" id="btnPrev">\u2190 Prev</button>
      <button class="nav-btn primary" id="btnNext">Next \u2192</button>
      <button class="nav-btn primary" id="btnNextUngraded">Next Ungraded \u2192</button>
      <button class="nav-btn" id="btnExport">Export CSV</button>
      <button class="nav-btn" id="btnImport">Import CSV</button>
      <button class="nav-btn" id="btnClear">Clear</button>
      <input type="file" id="importFile" accept=".csv,text/csv" style="display:none" />
    </div>

</div>
</div>

</div>
</div>
<div class="resize-handle resize-v" id="handleV" title="Drag to resize width"></div>
</div>

<script>
const CLIPS = __MANIFEST_JSON__;

const LS_KEY = 'lfc_v1_landing_foul';

function loadState() {
  try { return JSON.parse(localStorage.getItem(LS_KEY)) || {}; }
  catch (e) { return {}; }
}

let state           = loadState();
let currentIdx      = state.idx || 0;
let classifications = state.cls || {};
let layout = {
  videoRatio:   state.videoRatio   != null ? state.videoRatio   : 0.50,
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
  } else if (e.key === 'y' || e.key === 'Y') {
    classifyLanding('YES');
  } else if (e.key === 'n' || e.key === 'N') {
    classifyLanding('NO');
  } else if (e.key === 'u' || e.key === 'U') {
    classifyLanding('UNCLEAR');
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

function clipId(idx) {
  var c = CLIPS.clips[idx];
  return c.game_id + '_' + c.event_id;
}

function isComplete(cls) {
  return cls && cls.landing_foul;
}

document.getElementById('landingRow').querySelectorAll('.tag-btn').forEach(function(btn) {
  btn.addEventListener('click', function() { classifyLanding(btn.dataset.value); });
});

function syncUI() {
  var cid = clipId(currentIdx);
  var cls = classifications[cid] || {};
  document.getElementById('landingRow').querySelectorAll('.tag-btn').forEach(function(btn) {
    btn.classList.toggle('active', btn.dataset.value === cls.landing_foul);
  });
  document.getElementById('noteInput').value = cls.note || '';
}

function classifyLanding(value) {
  var cid = clipId(currentIdx);
  if (!classifications[cid]) classifications[cid] = {};
  classifications[cid].landing_foul = value;
  classifications[cid].ts = new Date().toISOString();
  saveState();
  syncUI();
  updateProgress();

  var panel = document.getElementById('classifyPanel');
  panel.classList.add('flash');
  setTimeout(function() {
    panel.classList.remove('flash');
    if (currentIdx < CLIPS.clips.length - 1) loadClip(currentIdx + 1);
  }, 400);
}

document.getElementById('noteInput').addEventListener('input', function() {
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

  var player = c.fouled_player_name || '\u2014';
  var matchup = (c.fouled_team_tricode || '?') + ' vs ' + (c.opponent || c.committing_team_tricode || '?');
  document.getElementById('ctxPlayer').textContent = player;
  document.getElementById('ctxGame').textContent = matchup + ' (' + c.game_id + ')';
  var p   = c.period || '?';
  var clk = (c.clock || '').replace('PT', '').replace('M', ':').replace('S', '');
  document.getElementById('ctxClock').textContent = 'Q' + p + '  ' + clk;
  document.getElementById('ctxOfficial').textContent = c.caller_official_name || '\u2014';
  document.getElementById('ctxDesc').textContent = c.description || '\u2014';
  syncUI();
  updateProgress();
}

function findNextUngradedIdx(fromIdx) {
  var n = CLIPS.clips.length;
  for (var offset = 1; offset <= n; offset++) {
    var idx = (fromIdx + offset) % n;
    if (!isComplete(classifications[clipId(idx)])) return idx;
  }
  return -1;
}

function updateProgress() {
  var total = CLIPS.clips.length;
  var done  = Object.values(classifications).filter(isComplete).length;
  var yes = 0, no = 0, unclear = 0;
  Object.values(classifications).forEach(function(cls) {
    if (!cls.landing_foul) return;
    if (cls.landing_foul === 'YES') yes++;
    else if (cls.landing_foul === 'NO') no++;
    else unclear++;
  });
  document.getElementById('progFill').style.width = total ? (done / total * 100) + '%' : '0%';
  document.getElementById('progText').textContent =
    'Clip ' + (currentIdx + 1) + '/' + total + ' \u00b7 ' + done + ' graded';
  document.getElementById('countSummary').textContent =
    'Y:' + yes + ' N:' + no + ' U:' + unclear +
    (done < total ? ' \u00b7 ' + (total - done) + ' left' : '');
}

function clearCurrent() {
  var cid = clipId(currentIdx);
  delete classifications[cid];
  saveState();
  syncUI();
  updateProgress();
}

function exportCSV() {
  var header = [
    'game_id','event_id','period','clock','description',
    'fouled_player_name','fouled_team_tricode','committing_player_name',
    'committing_team_tricode','caller_official_name',
    'landing_foul','note','timestamp'
  ];
  var rows = [header.join(',')];
  CLIPS.clips.forEach(function(c) {
    var cid = c.game_id + '_' + c.event_id;
    var cls = classifications[cid];
    if (!cls || !cls.landing_foul) return;
    rows.push([
      c.game_id, c.event_id, c.period, c.clock,
      '"' + (c.description || '').replace(/"/g, '""') + '"',
      '"' + (c.fouled_player_name || '').replace(/"/g, '""') + '"',
      c.fouled_team_tricode || '',
      '"' + (c.committing_player_name || '').replace(/"/g, '""') + '"',
      c.committing_team_tricode || c.opponent || '',
      c.caller_official_name || '',
      cls.landing_foul,
      '"' + (cls.note || '').replace(/"/g, '""') + '"',
      cls.ts || ''
    ].join(','));
  });
  var csv  = rows.join('\\n');
  var blob = new Blob([csv], { type: 'text/csv' });
  var url  = URL.createObjectURL(blob);
  var a    = document.createElement('a');
  a.href = url; a.download = 'landing_foul_classifications.csv';
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

function importCSV(text) {
  var lines = text.trim().split(/\\r?\\n/);
  if (lines.length < 2) return 0;
  var header = lines[0].split(',');
  var gi = header.indexOf('game_id');
  var ei = header.indexOf('event_id');
  var li = header.indexOf('landing_foul');
  var ni = header.indexOf('note');
  var ti = header.indexOf('timestamp');
  if (gi < 0 || ei < 0 || li < 0) {
    alert('CSV must include game_id, event_id, landing_foul columns');
    return 0;
  }
  var imported = 0;
  for (var r = 1; r < lines.length; r++) {
    var row = lines[r];
    if (!row.trim()) continue;
    var cols = [];
    var cur = '', inQ = false;
    for (var c = 0; c < row.length; c++) {
      var ch = row[c];
      if (inQ) {
        if (ch === '"' && row[c + 1] === '"') { cur += '"'; c++; }
        else if (ch === '"') inQ = false;
        else cur += ch;
      } else if (ch === '"') inQ = true;
      else if (ch === ',') { cols.push(cur); cur = ''; }
      else cur += ch;
    }
    cols.push(cur);
    var gid = cols[gi];
    var eid = cols[ei];
    var val = cols[li];
    if (!gid || !eid || !val) continue;
    var cid = gid + '_' + eid;
    if (!classifications[cid]) classifications[cid] = {};
    classifications[cid].landing_foul = val;
    classifications[cid].note = ni >= 0 ? (cols[ni] || '') : '';
    classifications[cid].ts = ti >= 0 ? (cols[ti] || new Date().toISOString()) : new Date().toISOString();
    imported++;
  }
  saveState();
  syncUI();
  updateProgress();
  return imported;
}

document.getElementById('btnPrev').addEventListener('click', function() {
  if (currentIdx > 0) loadClip(currentIdx - 1);
});
document.getElementById('btnNext').addEventListener('click', function() {
  if (currentIdx < CLIPS.clips.length - 1) loadClip(currentIdx + 1);
});
document.getElementById('btnNextUngraded').addEventListener('click', function() {
  var idx = findNextUngradedIdx(currentIdx);
  if (idx >= 0) loadClip(idx);
  else alert('All clips graded!');
});
document.getElementById('btnExport').addEventListener('click', exportCSV);
document.getElementById('btnImport').addEventListener('click', function() {
  document.getElementById('importFile').click();
});
document.getElementById('importFile').addEventListener('change', function(e) {
  var file = e.target.files && e.target.files[0];
  if (!file) return;
  var reader = new FileReader();
  reader.onload = function(ev) {
    var n = importCSV(ev.target.result);
    alert('Imported ' + n + ' classifications');
    var next = findNextUngradedIdx(currentIdx);
    if (next >= 0) loadClip(next);
  };
  reader.readAsText(file);
  e.target.value = '';
});
document.getElementById('btnClear').addEventListener('click', clearCurrent);

loadClip(currentIdx);
</script>
</body>
</html>"""


def generate_classifier(manifest_path: Path, output_path: Path | None = None) -> Path:
    with open(manifest_path) as f:
        manifest = json.load(f)

    num_clips = manifest.get("num_clips", len(manifest.get("clips", [])))
    logger.info("Generating landing foul classifier (%d clips)", num_clips)

    manifest_json = json.dumps(manifest, ensure_ascii=False)
    html = HTML_TEMPLATE.replace("__MANIFEST_JSON__", manifest_json)

    out_path = output_path or config.OUTPUT_DIR / "landing_foul_classifier.html"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write(html)

    logger.info("Wrote classifier to %s", out_path)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate landing foul classifier HTML")
    parser.add_argument(
        "--manifest",
        default=str(config.PROCESSED_DIR / "landing_foul_manifest.json"),
        help="Path to manifest JSON from landing_foul_manifest.py",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output HTML path (default: output/landing_foul_classifier.html)",
    )
    args = parser.parse_args()

    out_path = generate_classifier(
        Path(args.manifest),
        Path(args.output) if args.output else None,
    )
    print("\nTo view the classifier, run:")
    print(f"  python -m http.server 8080 --directory {out_path.parent}")
    print(f"  Then open http://localhost:8080/{out_path.name}")


if __name__ == "__main__":
    main()
