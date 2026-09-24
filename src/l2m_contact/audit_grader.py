"""Blind grader for tag_audit_sample.csv.

Shows each L2M comment without the tagger's label or the league decision. Grades are
appended to tag_audit_grades.csv (latest grade per row wins); the sample file is untouched.

    python3 src/l2m_contact/audit_grader.py        # then open http://localhost:8793
"""

from __future__ import annotations

import csv
import hashlib
import json
import random
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from paths import TRACKED

SAMPLE = TRACKED / "tag_audit_sample.csv"
GRADES = TRACKED / "tag_audit_grades.csv"
PORT = 8793
LEVELS = ["none", "marginal", "affecting", "legal", "not_contact_judgment", "unclear"]
FIELDS = ["row_id", "human_level", "note", "grader", "graded_at"]


def row_id(r: dict) -> str:
    return hashlib.sha1((r["clip_url"] + "|" + r["comment"]).encode()).hexdigest()[:12]


def load_rows() -> list[dict]:
    rows = list(csv.DictReader(SAMPLE.open()))
    random.Random(20260923).shuffle(rows)  # fixed order, mixes strata
    return [{"id": row_id(r), "call_type": r["call_type"], "comment": r["comment"],
             "clip_url": r["clip_url"]} for r in rows]


def load_grades() -> dict:
    out = {}
    if GRADES.exists():
        for g in csv.DictReader(GRADES.open()):
            out[g["row_id"]] = g
    return out


PAGE = """<!doctype html><meta charset=utf-8><title>L2M tag audit</title>
<style>
body{font:16px/1.5 -apple-system,system-ui,sans-serif;max-width:760px;margin:32px auto;padding:0 16px;color:#222}
.meta{color:#666;font-size:14px}.comment{font-size:20px;margin:18px 0;padding:16px;background:#f5f5f2;border-radius:8px}
button.l{display:block;width:100%;text-align:left;margin:6px 0;padding:10px 12px;font-size:15px;border:1px solid #ccc;border-radius:6px;background:#fff;cursor:pointer}
button.l.on{background:#1f5fbf;color:#fff;border-color:#1f5fbf}button.l b{display:inline-block;width:22px}
.nav{display:flex;gap:8px;margin-top:14px}.nav button{padding:8px 14px}input{width:100%;padding:8px;font-size:14px;margin-top:8px}
.bar{height:6px;background:#eee;border-radius:3px;margin:8px 0}.bar i{display:block;height:6px;background:#1f5fbf;border-radius:3px}
</style>
<div class=meta>Grader <input id=grader style="width:160px;display:inline" placeholder="your name"> <span id=prog></span></div>
<div class=bar><i id=barfill></i></div>
<div class=meta id=ct></div><div class=comment id=cm></div>
<div class=meta><a id=clip target=_blank>Watch clip</a> (optional; grade what the text says)</div>
<div id=btns></div>
<input id=note placeholder="note (optional)">
<div class=nav><button id=prev>&larr; Prev</button><button id=next>Next &rarr;</button><button id=skip>Next ungraded</button></div>
<script>
const L=[["none","No contact: league says no (illegal) contact was made"],
["marginal","Marginal: contact occurred and is described as marginal, incidental, brush, graze, brief"],
["affecting","Affecting: contact affected the player (RSBQ/FOM, dislodged, lost ball/balance) or 'more than marginal'"],
["legal","Legal: legal position, verticality, legal contest; contact level not stated"],
["not_contact_judgment","Not a contact judgment: take foul, 3 seconds, timeout, violation"],
["unclear","Unclear: the text does not say"]];
let rows=[],grades={},i=0;
const $=id=>document.getElementById(id);
$("grader").value=localStorage.getItem("grader")||"";
$("grader").onchange=e=>localStorage.setItem("grader",e.target.value);
L.forEach(([k,t],n)=>{const b=document.createElement("button");b.className="l";b.dataset.k=k;b.innerHTML=`<b>${n+1}</b>${t}`;b.onclick=()=>grade(k);$("btns").appendChild(b)});
async function load(){const d=await (await fetch("/api/rows")).json();rows=d.rows;grades=d.grades;i=Math.max(0,rows.findIndex(r=>!grades[r.id]));if(i<0)i=0;show()}
function show(){const r=rows[i],g=grades[r.id];$("ct").textContent=`${i+1} / ${rows.length} · ${r.call_type}`;$("cm").textContent=r.comment;
$("clip").href=r.clip_url;$("note").value=g?g.note:"";document.querySelectorAll("button.l").forEach(b=>b.classList.toggle("on",!!g&&g.human_level===b.dataset.k));
const n=Object.keys(grades).length;$("prog").textContent=`${n} graded`;$("barfill").style.width=(100*n/rows.length)+"%"}
async function grade(k){if(!$("grader").value.trim()){alert("Enter your name first");$("grader").focus();return}
const r=rows[i];const body={row_id:r.id,human_level:k,note:$("note").value,grader:$("grader").value.trim()};
const res=await fetch("/api/grade",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(body)});
if(!res.ok){alert("Save failed");return}grades[r.id]=body;if(i<rows.length-1)i++;show()}
$("prev").onclick=()=>{if(i>0){i--;show()}};$("next").onclick=()=>{if(i<rows.length-1){i++;show()}};
$("skip").onclick=()=>{const j=rows.findIndex((r,k)=>k>i&&!grades[r.id]);const j2=j<0?rows.findIndex(r=>!grades[r.id]):j;if(j2>=0){i=j2;show()}else alert("All rows graded")};
document.addEventListener("keydown",e=>{if(e.target.tagName==="INPUT")return;const n=parseInt(e.key);if(n>=1&&n<=6)grade(L[n-1][0]);
if(e.key==="ArrowLeft")$("prev").click();if(e.key==="ArrowRight")$("next").click()});
load();
</script>"""


class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/":
            self._send(200, PAGE.encode(), "text/html; charset=utf-8")
        elif self.path == "/api/rows":
            g = {k: {"human_level": v["human_level"], "note": v["note"]} for k, v in load_grades().items()}
            self._send(200, json.dumps({"rows": load_rows(), "grades": g}).encode(), "application/json")
        else:
            self._send(404, b"not found", "text/plain")

    def do_POST(self):
        if self.path != "/api/grade":
            return self._send(404, b"not found", "text/plain")
        d = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        valid = {r["id"] for r in load_rows()}
        if d.get("row_id") not in valid or d.get("human_level") not in LEVELS or not d.get("grader"):
            return self._send(400, b"bad grade", "text/plain")
        new = not GRADES.exists()
        with GRADES.open("a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            if new:
                w.writeheader()
            w.writerow({"row_id": d["row_id"], "human_level": d["human_level"], "note": d.get("note", ""),
                        "grader": d["grader"], "graded_at": time.strftime("%Y-%m-%dT%H:%M:%S")})
        self._send(200, b"ok", "text/plain")

    def log_message(self, *args):
        pass


if __name__ == "__main__":
    print(f"http://localhost:{PORT}")
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
