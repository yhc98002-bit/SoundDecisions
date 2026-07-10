"""Self-contained static HTML labeling bundles (June-13 manual section 13).

One generator serves every human pass from one build. Each bundle is a SINGLE
.html file with inline CSS+JS and base64-embedded media (no server, no external
deps, works in any browser incl. mobile). The annotator labels in-browser;
"Export JSONL" downloads the labels; in-progress work auto-saves to localStorage
so a session is resumable.

Two task types (the same widget set, configured per task):
  * "anchor"  — original FoleyBench clips (video+audio); widget = onset-tap only;
                exports {key, human_onset_s} (fills anchor_check_30.csv's column).
  * "validity"— GENERATED audio (audio only, no video — frozen interpretation:
                validity measures whether a human hears the same axis value the
                measurer computed on the MODEL'S OUTPUT, so the source video is
                deliberately withheld to avoid biasing toward the "intended"
                sound); widgets = presence toggle + event-class forced choice +
                abstain + onset-tap; exports {clip, presence, class, onset_s}.

The qwen labels for the same clips are embedded (read-only, shown after the
annotator commits each clip) so human and machine labels live on one pass; the
kappa join is done offline by scripts/compute_validity_kappa.py.

CPU-only, stdlib + soundfile/av (already in the venv). No web framework added.
"""

from __future__ import annotations

import base64
import html
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# 12 event-restricted coarse classes for the validity forced-choice (the coarse
# map's coarse_classes minus class_excluded_coarse). Loaded from the frozen map
# at build time; this fallback list documents the expected set.
_FALLBACK_EVENT_CLASSES = [
    "impact_friction", "footsteps_walk", "tools_hand", "machines_motors",
    "vehicles", "water_liquid", "guns_explosions", "doors_furniture",
    "electronics_ui", "animals", "food_cooking", "other",
]


def event_classes(coarse_map_path: Path) -> list[str]:
    """The event-restricted coarse classes (coarse_classes − class_excluded_coarse)."""
    d = json.loads(Path(coarse_map_path).read_text())
    excluded = set(d.get("class_excluded_coarse", []))
    classes = [c for c in d["coarse_classes"] if c not in excluded]
    return classes or list(_FALLBACK_EVENT_CLASSES)


@dataclass
class ClipItem:
    """One clip to label. media_b64 is a data-URI payload (audio or video)."""

    clip_id: str
    media_b64: str
    media_mime: str            # "audio/wav" | "video/mp4"
    caption: str = ""
    proposed_onset_s: Optional[float] = None
    qwen: dict = field(default_factory=dict)   # {axis_id: label}, shown read-only


def _b64_data_uri(path: Path, mime: str) -> str:
    return f"data:{mime};base64," + base64.b64encode(Path(path).read_bytes()).decode("ascii")


def audio_item(clip_id: str, wav_path: Path, caption: str = "",
               qwen: Optional[dict] = None) -> ClipItem:
    return ClipItem(clip_id=clip_id, media_b64=_b64_data_uri(wav_path, "audio/wav"),
                    media_mime="audio/wav", caption=caption, qwen=qwen or {})


def video_item(clip_id: str, mp4_path: Path, caption: str = "",
               proposed_onset_s: Optional[float] = None) -> ClipItem:
    return ClipItem(clip_id=clip_id, media_b64=_b64_data_uri(mp4_path, "video/mp4"),
                    media_mime="video/mp4", caption=caption,
                    proposed_onset_s=proposed_onset_s)


# --------------------------------------------------------------------------------------
# HTML rendering
# --------------------------------------------------------------------------------------
def render_bundle(task: str, items: list[ClipItem], classes: list[str],
                  title: str, prompt_version: str = "v1") -> str:
    """Return a complete self-contained HTML document string."""
    if task not in ("anchor", "validity"):
        raise ValueError(f"unknown task {task!r}")
    payload = {
        "task": task, "prompt_version": prompt_version, "classes": classes,
        "items": [{"clip_id": it.clip_id, "media": it.media_b64, "mime": it.media_mime,
                   "caption": it.caption, "proposed_onset_s": it.proposed_onset_s,
                   "qwen": it.qwen} for it in items],
    }
    data_json = json.dumps(payload)
    # embed as a script of type application/json to avoid </script> issues
    safe = data_json.replace("</", "<\\/")
    return _HTML_TEMPLATE.replace("__TITLE__", html.escape(title)) \
                         .replace("__TASK__", task) \
                         .replace("__DATA__", safe)


# A single inline template; CSS + JS are self-contained. {} are literal in JS, so
# the template uses __PLACEHOLDER__ markers (replaced above), not str.format.
_HTML_TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
  :root { color-scheme: light dark; }
  body { font-family: system-ui, sans-serif; max-width: 760px; margin: 0 auto;
         padding: 12px; line-height: 1.4; }
  h1 { font-size: 1.1rem; } .muted { opacity: .65; font-size: .85rem; }
  .card { border: 1px solid #8884; border-radius: 10px; padding: 14px; margin: 10px 0; }
  .row { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; }
  button { font: inherit; padding: 8px 12px; border-radius: 8px; border: 1px solid #8886;
           background: #8881; cursor: pointer; }
  button.sel { background: #3a7afe; color: #fff; border-color: #3a7afe; }
  button.tap { background: #1b9e54; color: #fff; border-color: #1b9e54; }
  .classgrid { display: grid; grid-template-columns: repeat(auto-fill, minmax(120px,1fr)); gap: 6px; }
  .nav { position: sticky; bottom: 0; background: var(--bg, #fff8); backdrop-filter: blur(6px);
         padding: 8px 0; display: flex; gap: 8px; align-items: center; }
  .done { color: #1b9e54; } .pending { color: #d08; }
  audio, video { width: 100%; margin: 8px 0; }
  .qwen { font-size: .8rem; opacity: .7; margin-top: 6px; }
  progress { width: 100%; height: 8px; }
</style></head>
<body>
<h1>__TITLE__</h1>
<p class="muted">Label each clip, then <b>Export JSONL</b> and send the file back.
Progress auto-saves in this browser. Task: <code>__TASK__</code>.</p>
<progress id="prog" value="0" max="1"></progress>
<div id="root"></div>
<div class="nav">
  <button id="prev">&larr; Prev</button>
  <span id="counter" class="muted"></span>
  <button id="next">Next &rarr;</button>
  <button id="export">Export JSONL</button>
  <span id="status" class="muted"></span>
</div>
<script type="application/json" id="data">__DATA__</script>
<script>
const D = JSON.parse(document.getElementById("data").textContent);
const TASK = D.task, KEYP = "foley_label_" + TASK + "_" + D.prompt_version;
const store = JSON.parse(localStorage.getItem(KEYP) || "{}");
let idx = 0;
const root = document.getElementById("root");

function save(){ localStorage.setItem(KEYP, JSON.stringify(store)); refreshProgress(); }
function rec(id){ return store[id] || (store[id] = {clip_id:id}); }
function refreshProgress(){
  const n = D.items.length;
  const done = D.items.filter(it => isDone(store[it.clip_id])).length;
  document.getElementById("prog").value = n ? done/n : 0;
  document.getElementById("counter").textContent = (idx+1)+" / "+n+"  ("+done+" done)";
}
function isDone(r){
  if(!r) return false;
  if(TASK==="anchor") return typeof r.human_onset_s === "number";
  return (r.presence!=null) && (r.class!=null);
}

function render(){
  const it = D.items[idx], r = rec(it.clip_id);
  let h = '<div class="card"><div class="row"><b>'+ (idx+1) +'.</b> <code>'+it.clip_id+'</code>';
  h += isDone(r) ? ' <span class="done">✓</span>' : ' <span class="pending">…</span></div>';
  if(it.caption) h += '<div class="muted">'+it.caption+'</div>';
  if(it.mime.startsWith("video")) h += '<video id="med" src="'+it.media+'" controls></video>';
  else h += '<audio id="med" src="'+it.media+'" controls></audio>';

  if(TASK==="validity"){
    h += '<div class="row"><b>presence:</b>'
       + '<button data-p="present">present</button>'
       + '<button data-p="absent">absent</button></div>';
    h += '<div><b>class:</b><div class="classgrid">'
       + D.classes.map(c=>'<button data-c="'+c+'">'+c+'</button>').join('')
       + '<button data-c="abstain">abstain</button></div></div>';
  }
  h += '<div class="row"><b>onset:</b><button class="tap" id="tap">TAP at playhead</button>'
     + '<span id="onset"></span>';
  if(it.proposed_onset_s!=null) h += ' <span class="muted">(proposed '+it.proposed_onset_s.toFixed(2)+'s)</span>';
  h += '</div>';
  if(it.qwen && Object.keys(it.qwen).length)
    h += '<div class="qwen">qwen: '+Object.entries(it.qwen).map(([k,v])=>k+'='+v).join('  ')+'</div>';
  h += '</div>';
  root.innerHTML = h;

  // restore + wire
  const med = document.getElementById("med");
  if(TASK==="validity"){
    root.querySelectorAll('[data-p]').forEach(b=>{
      if(r.presence===b.dataset.p) b.classList.add("sel");
      b.onclick=()=>{ r.presence=b.dataset.p; save(); render(); };
    });
    root.querySelectorAll('[data-c]').forEach(b=>{
      if(r.class===b.dataset.c) b.classList.add("sel");
      b.onclick=()=>{ r.class=b.dataset.c; save(); render(); };
    });
  }
  const onsetSpan = document.getElementById("onset");
  const cur = (TASK==="anchor") ? r.human_onset_s : r.onset_s;
  if(cur!=null) onsetSpan.textContent = cur.toFixed(2)+" s";
  document.getElementById("tap").onclick=()=>{
    const t = med ? med.currentTime : 0;
    if(TASK==="anchor") r.human_onset_s = t; else r.onset_s = t;
    save(); onsetSpan.textContent = t.toFixed(2)+" s";
  };
  refreshProgress();
}

document.getElementById("prev").onclick=()=>{ if(idx>0){idx--;render();} };
document.getElementById("next").onclick=()=>{ if(idx<D.items.length-1){idx++;render();} };
document.getElementById("export").onclick=()=>{
  const lines = D.items.map(it=>{
    const r = store[it.clip_id] || {clip_id: it.clip_id};
    const out = {clip_id: it.clip_id, task: TASK, prompt_version: D.prompt_version};
    if(TASK==="anchor"){ out.human_onset_s = (r.human_onset_s!=null)?r.human_onset_s:null; }
    else { out.presence=r.presence??null; out.class=r.class??null;
           out.onset_s=(r.onset_s!=null)?r.onset_s:null; }
    return JSON.stringify(out);
  });
  const blob = new Blob([lines.join("\n")+"\n"], {type:"application/jsonl"});
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "labels_"+TASK+"_"+D.prompt_version+".jsonl";
  a.click();
  document.getElementById("status").textContent = "exported "+lines.length+" rows";
};
render();
</script>
</body></html>
"""


def write_bundle(out_path: Path, task: str, items: list[ClipItem], classes: list[str],
                 title: str, prompt_version: str = "v1") -> dict:
    """Write the bundle and return a manifest (clip ids, sizes, byte total)."""
    doc = render_bundle(task, items, classes, title, prompt_version)
    Path(out_path).write_text(doc, encoding="utf-8")
    return {"task": task, "out": str(out_path), "n_clips": len(items),
            "bytes": len(doc.encode("utf-8")), "clip_ids": [it.clip_id for it in items],
            "prompt_version": prompt_version}
