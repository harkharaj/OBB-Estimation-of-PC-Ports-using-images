"""
socket_annotator.py
-------------------
Click 4 corners (TL -> TR -> BR -> BL) for each socket across ≥2 frames.
Triangulates all 4 corners -> derives real-world center + extent.
Rotation matrix is fixed (hardcoded panel rotation).

Controls:
  Click       -> add corner point (4 points total per socket)
  Esc         -> cancel current drawing
  Alt + drag  -> pan
  Scroll      -> zoom
"""

from flask import Flask, render_template_string, request, jsonify, send_file
import json, os, io, re
import numpy as np
from PIL import Image, ImageDraw

app = Flask(__name__)

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
DATA_DIR = r"C:\Users\harkh\OneDrive\Desktop\ROBOTIC_PERCEPTION_FINAL_PROJECT\Data"

ROTATION = np.array([
    [-0.004004375172752437,  0.9672545151126772, -0.25377680739897346],
    [ 0.01584254528462312,   0.25380835519540434, 0.9671247761234889],
    [ 0.9998664804554559,   -0.00014774012094266402, -0.016340117333610394]
])

K = np.array([
    [1477.00974684544,   0.0,                1298.2501500778505],
    [0.0,                1480.4424455584467, 686.8201623541711],
    [0.0,                0.0,                1.0]
])

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def load_poses():
    with open(os.path.join(DATA_DIR, "poses.json")) as f:
        return json.load(f)

def triangulate(observations, poses):
    A = []
    for fk, u, v in observations:
        c2w = np.array(poses[str(fk)])
        w2c = np.linalg.inv(c2w)
        R, t = w2c[:3, :3], w2c[:3, 3]
        P = K @ np.hstack([R, t.reshape(3, 1)])
        A.append(v * P[2] - P[1])
        A.append(u * P[2] - P[0])
    A = np.array(A)
    _, _, Vt = np.linalg.svd(A)
    X = Vt[-1]
    return X[:3] / X[3]

def get_corners_3d(center, extent, rotation):
    dx, dy, dz = extent
    pts = np.array([[sx*dx, sy*dy, sz*dz]
                    for sx in [-1, 1] for sy in [-1, 1] for sz in [-1, 1]]).T
    return (rotation @ pts + np.array(center).reshape(3, 1)).T

def project(world_pt, frame_key, poses):
    c2w = np.array(poses[str(frame_key)])
    w2c = np.linalg.inv(c2w)
    R, t = w2c[:3, :3], w2c[:3, 3]
    cam = R @ np.array(world_pt) + t
    if cam[2] <= 0:
        return None, None
    return float(K[0,0]*cam[0]/cam[2] + K[0,2]), float(K[1,1]*cam[1]/cam[2] + K[1,2])

def get_frames():
    try:
        return sorted([f for f in os.listdir(DATA_DIR)
                       if f.lower().endswith(".png") and "frame" in f.lower()])
    except:
        return []

def save_answers(entity, center, extent, rotation):
    path = os.path.join(DATA_DIR, "answers.json")
    answers = []
    if os.path.exists(path):
        with open(path) as f:
            answers = json.load(f)
    for e in answers:
        if e["entity"] == entity:
            e["obb"] = {"center": center, "extent": extent, "rotation": rotation}
            break
    else:
        answers.append({"entity": entity,
                        "obb": {"center": center, "extent": extent, "rotation": rotation}})
    with open(path, "w") as f:
        json.dump(answers, f, indent=2)

# ─────────────────────────────────────────────
# HTML + JS
# ─────────────────────────────────────────────
HTML = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Socket Annotator</title>
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Syne:wght@700;800&display=swap" rel="stylesheet">
<style>
  :root {
    --bg:    #0a0a0f; --surf: #12121a; --border: #1e1e2e;
    --acc:   #00ff9d; --acc2: #ff3c6e; --acc3: #3c8fff; --acc4: #ffb347;
    --text:  #e0e0f0; --muted: #555570;
    --mono:  'JetBrains Mono', monospace; --disp: 'Syne', sans-serif;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: var(--bg); color: var(--text); font-family: var(--mono); height: 100vh; display: grid; grid-template-rows: auto 1fr; overflow: hidden; }

  header { padding: 12px 26px; border-bottom: 1px solid var(--border); background: var(--surf); display: flex; align-items: center; gap: 14px; }
  header h1 { font-family: var(--disp); font-size: 1.1rem; font-weight: 800; color: var(--acc); }
  header span { color: var(--muted); font-size: 0.7rem; }

  .layout { display: grid; grid-template-columns: 230px 1fr 280px; overflow: hidden; height: 100%; }

  /* LEFT */
  .left { background: var(--surf); border-right: 1px solid var(--border); display: flex; flex-direction: column; overflow: hidden; }
  .sec-title { padding: 9px 13px 6px; font-size: 0.6rem; font-weight: 700; letter-spacing: 2px; text-transform: uppercase; color: var(--muted); border-bottom: 1px solid var(--border); }
  .frame-count { padding: 4px 13px; font-size: 0.63rem; color: var(--muted); border-bottom: 1px solid var(--border); }
  .frame-list { overflow-y: auto; flex: 1; }
  .frame-item { padding: 7px 13px; cursor: pointer; font-size: 0.71rem; color: var(--muted); border-left: 2px solid transparent; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; transition: all .13s; }
  .frame-item:hover { background: var(--border); color: var(--text); }
  .frame-item.active { background: rgba(0,255,157,.07); color: var(--acc); border-left-color: var(--acc); }
  .entity-bar { padding: 9px 11px; border-top: 1px solid var(--border); display: flex; flex-direction: column; gap: 6px; }
  .lbl { font-size: 0.6rem; color: var(--muted); letter-spacing: 1.5px; text-transform: uppercase; }
  select.ent-sel { background: var(--bg); color: var(--text); border: 1px solid var(--border); border-radius: 4px; padding: 6px 9px; font-family: var(--mono); font-size: 0.76rem; width: 100%; outline: none; cursor: pointer; }
  select.ent-sel:focus { border-color: var(--acc); }

  /* VIEWER */
  .viewer { position: relative; overflow: hidden; background: #05050a; cursor: crosshair; }
  #main-img { position: absolute; top: 0; left: 0; max-width: none; display: block; user-select: none; -webkit-user-drag: none; }
  #canvas { position: absolute; top: 0; left: 0; width: 100%; height: 100%; pointer-events: none; }
  .hud { position: absolute; top: 10px; left: 10px; background: rgba(10,10,15,.88); border: 1px solid var(--border); border-radius: 4px; padding: 4px 10px; font-size: 0.68rem; color: var(--muted); pointer-events: none; z-index: 10; }
  .hud span { color: var(--acc); }
  .ent-hud { position: absolute; top: 10px; right: 10px; border-radius: 4px; padding: 5px 12px; font-size: 0.72rem; font-weight: 700; letter-spacing: 1px; pointer-events: none; z-index: 10; }
  .ent-hud.eth { background: rgba(60,143,255,.15); border: 1px solid var(--acc3); color: var(--acc3); }
  .ent-hud.pwr { background: rgba(255,179,71,.15);  border: 1px solid var(--acc4); color: var(--acc4); }
  .ent-hud.vga { background: rgba(0,255,157,.15);   border: 1px solid var(--acc);  color: var(--acc);  }
  .mode-badge { position: absolute; bottom: 50px; left: 50%; transform: translateX(-50%); background: rgba(10,10,15,.9); border: 1px solid var(--border); border-radius: 4px; padding: 4px 12px; font-size: 0.64rem; color: var(--muted); pointer-events: none; z-index: 10; white-space: nowrap; }
  .no-img { position: absolute; inset: 0; display: flex; align-items: center; justify-content: center; color: var(--muted); font-size: .82rem; pointer-events: none; }
  .zoom-btns { position: absolute; bottom: 12px; right: 12px; display: flex; gap: 4px; z-index: 10; }
  .zbtn { background: var(--surf); border: 1px solid var(--border); color: var(--text); width: 30px; height: 30px; border-radius: 4px; cursor: pointer; font-size: .95rem; display: flex; align-items: center; justify-content: center; transition: all .13s; }
  .zbtn:hover { border-color: var(--acc); color: var(--acc); }

  /* RIGHT */
  .right { background: var(--surf); border-left: 1px solid var(--border); display: flex; flex-direction: column; overflow: hidden; }
  .obs-list { flex: 1; overflow-y: auto; padding: 7px; display: flex; flex-direction: column; gap: 3px; }
  .obs-group-hdr { font-size: 0.58rem; letter-spacing: 1.5px; text-transform: uppercase; padding: 7px 4px 3px; border-bottom: 1px solid var(--border); margin-bottom: 2px; }
  .obs-group-hdr.eth { color: var(--acc3); }
  .obs-group-hdr.pwr { color: var(--acc4); }
  .obs-group-hdr.vga { color: var(--acc);  }
  .obs-item { background: var(--bg); border: 1px solid var(--border); border-radius: 4px; padding: 5px 9px; font-size: 0.65rem; display: flex; justify-content: space-between; align-items: center; animation: fi .16s ease; }
  @keyframes fi { from { opacity:0; transform:translateY(-3px); } to { opacity:1; transform:none; } }
  .obs-frame { font-weight: 700; }
  .obs-frame.eth { color: var(--acc3); }
  .obs-frame.pwr { color: var(--acc4); }
  .obs-frame.vga { color: var(--acc);  }
  .obs-uv { color: var(--muted); font-size: 0.6rem; margin-left: 4px; }
  .del { background: none; border: none; color: var(--acc2); cursor: pointer; font-size: .75rem; opacity: .4; transition: opacity .13s; padding: 0 3px; }
  .del:hover { opacity: 1; }
  .result { margin: 0 9px 7px; background: var(--bg); border: 1px solid var(--border); border-radius: 4px; padding: 8px; font-size: 0.63rem; max-height: 120px; overflow-y: auto; white-space: pre-wrap; word-break: break-all; display: none; }
  .result.show { display: block; }
  .actions { padding: 9px 10px; border-top: 1px solid var(--border); display: flex; flex-direction: column; gap: 6px; }
  .btn { padding: 8px 11px; border-radius: 4px; border: none; cursor: pointer; font-family: var(--mono); font-size: 0.73rem; font-weight: 700; transition: all .13s; width: 100%; letter-spacing: .4px; }
  .btn-go  { background: var(--acc); color: var(--bg); }
  .btn-go:hover { filter: brightness(1.12); }
  .btn-val { background: transparent; border: 1px solid var(--acc3); color: var(--acc3); }
  .btn-val:hover { background: rgba(60,143,255,.1); }
  .btn-clr { background: transparent; border: 1px solid var(--border); color: var(--muted); }
  .btn-clr:hover { border-color: var(--acc2); color: var(--acc2); }
  .btn-sav { background: transparent; border: 1px solid var(--border); color: var(--muted); }
  .btn-sav:hover { border-color: var(--acc); color: var(--acc); }
  .val-row { display: flex; gap: 5px; }
  .val-row select { flex: 1; background: var(--bg); color: var(--text); border: 1px solid var(--border); border-radius: 4px; padding: 5px 7px; font-family: var(--mono); font-size: 0.7rem; outline: none; cursor: pointer; }
  .val-row select:focus { border-color: var(--acc3); }
  .overlay { display: none; position: fixed; inset: 0; background: rgba(0,0,0,.82); z-index: 100; align-items: center; justify-content: center; }
  .overlay.open { display: flex; }
  .modal { background: var(--surf); border: 1px solid var(--border); border-radius: 8px; padding: 16px; max-width: 94vw; max-height: 94vh; overflow: auto; display: flex; flex-direction: column; gap: 10px; }
  .modal h2 { font-family: var(--disp); font-size: .9rem; color: var(--acc); }
  .modal img { max-width: 100%; border-radius: 4px; border: 1px solid var(--border); }
  .modal-close { align-self: flex-end; background: var(--bg); border: 1px solid var(--border); color: var(--text); padding: 5px 13px; border-radius: 4px; cursor: pointer; font-family: var(--mono); font-size: .73rem; }
  .modal-close:hover { border-color: var(--acc2); color: var(--acc2); }
  ::-webkit-scrollbar { width: 4px; }
  ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 2px; }
</style>
</head>
<body>

<header>
  <h1>◈ Socket Annotator</h1>
  <span>Click 4 corners (TL → TR → BR → BL) · ≥2 frames · Alt+drag to pan · Esc to cancel drawing</span>
</header>

<div class="layout">

  <div class="left">
    <div class="sec-title">Frames</div>
    <div class="frame-count" id="fc">—</div>
    <div class="frame-list" id="frameList"></div>
    <div class="entity-bar">
      <div class="lbl">Active Entity</div>
      <select class="ent-sel" id="entSel" onchange="onEntChange()">
        <option value="ethernet_socket">ethernet_socket</option>
        <option value="power_socket">power_socket</option>
        <option value="vga_socket">vga_socket</option>
      </select>
    </div>
  </div>

  <div class="viewer" id="viewer">
    <div class="no-img" id="noImg">← Select a frame to begin</div>
    <img id="main-img" src="" alt="" style="display:none;">
    <canvas id="canvas"></canvas>
    <div class="hud">cursor: <span id="hx">—</span>, <span id="hy">—</span> &nbsp;|&nbsp; zoom: <span id="hz">1.0</span>x</div>
    <div class="ent-hud eth" id="entHud">ethernet_socket</div>
    <div class="mode-badge">click 4 corners in order (TL→TR→BR→BL)</div>
    <div class="zoom-btns">
      <button class="zbtn" onclick="doZoom(1.25)">+</button>
      <button class="zbtn" onclick="doZoom(0.8)">−</button>
      <button class="zbtn" onclick="resetZoom()">⊙</button>
    </div>
  </div>

  <div class="right">
    <div class="sec-title">Bounding Polygons</div>
    <div class="obs-list" id="obsList">
      <div style="padding:13px;color:var(--muted);font-size:.7rem;text-align:center;">Click 4 corners on ≥2 frames</div>
    </div>
    <div class="result" id="result"></div>
    <div class="actions">
      <button class="btn btn-go" onclick="compute()">▶ Triangulate + Compute OBB</button>
      <div class="val-row">
        <select id="valFrame"></select>
        <button class="btn btn-val" style="width:auto;padding:5px 11px;" onclick="validate()">⬡ Validate</button>
      </div>
      <button class="btn btn-clr" onclick="clearAll()">✕ Clear All</button>
      <button class="btn btn-sav" onclick="saveToFile()">↓ Save answers.json</button>
    </div>
  </div>

</div>

<div class="overlay" id="overlay">
  <div class="modal">
    <h2>OBB Validation</h2>
    <img id="valImg" src="" alt="">
    <button class="modal-close" onclick="closeModal()">Close</button>
  </div>
</div>

<script>
  let frames = {{ frames|tojson }};
  let obs = { ethernet_socket:[], power_socket:[], vga_socket:[] };
  let lastResult = {};
  let curFrame = null;
  let scale = 1, offset = {x:0, y:0};
  let natW = 2560, natH = 1440;

  // polygon draw state
  let currentPts = [];
  let cursorHover = null;
  // pan state
  let panning = false, panStart = {x:0,y:0};

  const viewer = document.getElementById('viewer');
  const img    = document.getElementById('main-img');
  const canvas = document.getElementById('canvas');
  const ctx    = canvas.getContext('2d');

  const ENT_COLOR = { ethernet_socket:'#3c8fff', power_socket:'#ffb347', vga_socket:'#00ff9d' };
  const ENT_CLS   = { ethernet_socket:'eth',     power_socket:'pwr',     vga_socket:'vga'     };
  const ENT_SHORT = { ethernet_socket:'ETH',     power_socket:'PWR',     vga_socket:'VGA'     };

  // ── Frame list ─────────────────────────────────────────────────────────────
  function buildFrames() {
    const list = document.getElementById('frameList');
    const fc   = document.getElementById('fc');
    const vf   = document.getElementById('valFrame');
    list.innerHTML = ''; vf.innerHTML = '';
    if (!frames.length) {
      fc.textContent = '⚠ No frames found'; fc.style.color = 'var(--acc2)'; return;
    }
    fc.textContent = frames.length + ' frames';
    frames.forEach(f => {
      const d = document.createElement('div');
      d.className = 'frame-item'; d.textContent = f; d.title = f; d.id = 'fi_' + f;
      d.onclick = () => loadFrame(f);
      list.appendChild(d);
      const o = document.createElement('option');
      o.value = f; o.textContent = f; vf.appendChild(o);
    });
  }

  function loadFrame(f) {
    curFrame = f;
    document.querySelectorAll('.frame-item').forEach(e => e.classList.remove('active'));
    const el = document.getElementById('fi_' + f);
    if (el) { el.classList.add('active'); el.scrollIntoView({block:'nearest'}); }
    img.src = '/frame/' + f;
    img.style.display = 'block';
    document.getElementById('noImg').style.display = 'none';
    img.onload = () => { natW = img.naturalWidth; natH = img.naturalHeight; resetZoom(); };
  }

  function onEntChange() {
    const ent = document.getElementById('entSel').value;
    const hud = document.getElementById('entHud');
    hud.textContent = ent;
    hud.className = 'ent-hud ' + ENT_CLS[ent];
    redraw();
  }

  // ── Zoom / pan ─────────────────────────────────────────────────────────────
  function applyT() {
    img.style.left = offset.x + 'px'; img.style.top = offset.y + 'px';
    img.style.width = natW * scale + 'px'; img.style.height = natH * scale + 'px';
    document.getElementById('hz').textContent = scale.toFixed(2);
    redraw();
  }
  function doZoom(f, cx, cy) {
    const r = viewer.getBoundingClientRect();
    cx = cx ?? r.width/2; cy = cy ?? r.height/2;
    const prev = scale;
    scale = Math.min(Math.max(scale * f, 0.05), 20);
    offset.x = cx - (cx - offset.x) * (scale/prev);
    offset.y = cy - (cy - offset.y) * (scale/prev);
    applyT();
  }
  function resetZoom() {
    const r = viewer.getBoundingClientRect();
    if (!r.width) { requestAnimationFrame(resetZoom); return; }
    scale = Math.min(r.width/natW, r.height/natH) * 0.97;
    offset.x = (r.width  - natW*scale) / 2;
    offset.y = (r.height - natH*scale) / 2;
    applyT();
  }
  viewer.addEventListener('wheel', e => {
    e.preventDefault();
    const r = viewer.getBoundingClientRect();
    doZoom(e.deltaY < 0 ? 1.12 : 0.89, e.clientX - r.left, e.clientY - r.top);
  }, {passive:false});

  // ── Mouse & Keys ───────────────────────────────────────────────────────────
  function imgCoords(e) {
    const r = viewer.getBoundingClientRect();
    return { u: (e.clientX - r.left - offset.x) / scale,
             v: (e.clientY - r.top  - offset.y) / scale };
  }

  viewer.addEventListener('mousedown', e => {
    if (e.button !== 0) return;
    if (e.altKey) {
      panning = true;
      panStart = {x: e.clientX - offset.x, y: e.clientY - offset.y};
      viewer.style.cursor = 'grabbing';
    } else if (curFrame) {
      const c = imgCoords(e);
      currentPts.push(c);
      
      if (currentPts.length === 4) {
        const ent  = document.getElementById('entSel').value;
        const match = curFrame.match(/(\d+)/);
        const fn   = match ? String(parseInt(match[1])) : curFrame;
        obs[ent].push({frame:fn, pts:[...currentPts], file:curFrame});
        currentPts = [];
        renderObs();
      }
      redraw();
    }
  });

  window.addEventListener('mousemove', e => {
    if (curFrame) {
      cursorHover = imgCoords(e);
      if (cursorHover.u >= 0 && cursorHover.u <= natW && cursorHover.v >= 0 && cursorHover.v <= natH) {
        document.getElementById('hx').textContent = Math.round(cursorHover.u);
        document.getElementById('hy').textContent = Math.round(cursorHover.v);
      }
    }
    if (panning) {
      offset.x = e.clientX - panStart.x;
      offset.y = e.clientY - panStart.y;
      applyT();
    } else if (currentPts.length > 0) {
      redraw();
    }
  });

  window.addEventListener('mouseup', e => {
    if (panning) {
      panning = false; viewer.style.cursor = 'crosshair';
    }
  });
  
  window.addEventListener('keydown', e => {
    if (e.key === 'Escape' && currentPts.length > 0) {
      currentPts = [];
      redraw();
    }
  });

  // ── Canvas ─────────────────────────────────────────────────────────────────
  function redraw() {
    const r = viewer.getBoundingClientRect();
    canvas.width = r.width; canvas.height = r.height;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (!curFrame) return;

    const sx = u => offset.x + u * scale;
    const sy = v => offset.y + v * scale;

    // Committed polygons for all entities
    Object.entries(obs).forEach(([ent, list]) => {
      const color = ENT_COLOR[ent];
      list.filter(o => o.file === curFrame).forEach(o => {
        // filled shape
        ctx.fillStyle = color + '18';
        ctx.strokeStyle = color; ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(sx(o.pts[0].u), sy(o.pts[0].v));
        ctx.lineTo(sx(o.pts[1].u), sy(o.pts[1].v));
        ctx.lineTo(sx(o.pts[2].u), sy(o.pts[2].v));
        ctx.lineTo(sx(o.pts[3].u), sy(o.pts[3].v));
        ctx.closePath();
        ctx.fill(); ctx.stroke();

        // corner markers with point number
        ctx.font = 'bold 10px JetBrains Mono, monospace';
        o.pts.forEach((p, i) => {
            const px = sx(p.u), py = sy(p.v);
            ctx.fillStyle = color;
            ctx.fillRect(px-3, py-3, 6, 6);
            ctx.fillStyle = 'white';
            ctx.fillText(i+1, px+5, py-5);
        });

        // center cross
        const cu = (o.pts[0].u + o.pts[1].u + o.pts[2].u + o.pts[3].u) / 4;
        const cv = (o.pts[0].v + o.pts[1].v + o.pts[2].v + o.pts[3].v) / 4;
        const cx = sx(cu), cy = sy(cv);
        ctx.strokeStyle = color; ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.moveTo(cx-8,cy); ctx.lineTo(cx+8,cy);
        ctx.moveTo(cx,cy-8); ctx.lineTo(cx,cy+8);
        ctx.stroke();

        // label
        ctx.fillStyle = color;
        ctx.font = 'bold 12px JetBrains Mono, monospace';
        ctx.fillText(ENT_SHORT[ent] + ' f' + o.frame, sx(o.pts[0].u)+4, sy(o.pts[0].v)-12);
      });
    });

    // In-progress polygon drawing (dashed)
    if (currentPts.length > 0) {
      const ent = document.getElementById('entSel').value;
      const color = ENT_COLOR[ent];
      
      ctx.strokeStyle = color; ctx.lineWidth = 1.5;
      ctx.setLineDash([6,3]);
      ctx.beginPath();
      ctx.moveTo(sx(currentPts[0].u), sy(currentPts[0].v));
      for(let i=1; i<currentPts.length; i++) {
        ctx.lineTo(sx(currentPts[i].u), sy(currentPts[i].v));
      }
      if (cursorHover) {
        ctx.lineTo(sx(cursorHover.u), sy(cursorHover.v));
      }
      ctx.stroke();
      ctx.setLineDash([]);
      
      ctx.font = 'bold 10px JetBrains Mono, monospace';
      currentPts.forEach((p, i) => {
        const px = sx(p.u), py = sy(p.v);
        ctx.fillStyle = color;
        ctx.fillRect(px-3, py-3, 6, 6);
        ctx.fillStyle = 'white';
        ctx.fillText(i+1, px+5, py-5);
      });
    }
  }

  // ── Obs list ───────────────────────────────────────────────────────────────
  function renderObs() {
    const list = document.getElementById('obsList');
    const total = Object.values(obs).reduce((s,a) => s+a.length, 0);
    if (!total) {
      list.innerHTML = '<div style="padding:13px;color:var(--muted);font-size:.7rem;text-align:center;">Click 4 corners on ≥2 frames</div>';
      return;
    }
    let html = '';
    [['ethernet_socket','eth'],['power_socket','pwr'],['vga_socket','vga']].forEach(([ent,cls]) => {
      if (!obs[ent].length) return;
      html += '<div class="obs-group-hdr ' + cls + '">' + ent + ' (' + obs[ent].length + 'x)</div>';
      obs[ent].forEach((o,i) => {
        html += '<div class="obs-item">' +
          '<div><span class="obs-frame ' + cls + '">f' + o.frame + '</span>' +
          '<span class="obs-uv">Polygon (4 points)</span></div>' +
          '<button class="del" onclick="delObs(\'' + ent + '\',' + i + ')">x</button>' +
          '</div>';
      });
    });
    list.innerHTML = html;
  }

  function delObs(ent, i) { obs[ent].splice(i,1); renderObs(); redraw(); }
  function clearAll() {
    obs = {ethernet_socket:[], power_socket:[], vga_socket:[]};
    lastResult = {}; currentPts = [];
    renderObs(); redraw();
    document.getElementById('result').classList.remove('show');
  }

  // ── Compute ────────────────────────────────────────────────────────────────
  async function compute() {
    const ent  = document.getElementById('entSel').value;
    const list = obs[ent];
    if (list.length < 2) {
      showResult('Need >=2 polygons for ' + ent + '. Have ' + list.length + '.', true); return;
    }
    showResult('Triangulating 4 corners...', false);
    try {
      const res = await fetch('/compute', {
        method:'POST', headers:{'Content-Type':'application/json'},
        body: JSON.stringify({ entity:ent, boxes: list.map(o => ({frame:o.frame, pts:o.pts})) })
      });
      const data = await res.json();
      if (data.error) { showResult('Error: ' + data.error, true); return; }
      lastResult[ent] = data;
      showResult(fmt(data), false);
    } catch(e) { showResult('Error: ' + e.message, true); }
  }

  function fmt(d) {
    return [
      'entity : ' + d.entity,
      'center : [' + d.center.map(x=>x.toFixed(5)).join(', ') + ']',
      'extent : [' + d.extent.map(x=>x.toFixed(5)).join(', ') + ']',
      '',
      'reproj errors (center):',
      ...d.reprojection.map(r => '  f' + r.frame + ': ' + r.error.toFixed(1) + 'px'),
      '',
      'saved to answers.json'
    ].join('\n');
  }

  function showResult(txt, err) {
    const el = document.getElementById('result');
    el.textContent = txt; el.style.color = err ? 'var(--acc2)' : 'var(--acc)';
    el.classList.add('show');
  }

  function validate() {
    const ent   = document.getElementById('entSel').value;
    const frame = document.getElementById('valFrame').value;
    if (!lastResult[ent]) { showResult('Compute OBB for ' + ent + ' first.', true); return; }
    document.getElementById('valImg').src = '/validate?entity=' + ent + '&frame=' + encodeURIComponent(frame) + '&t=' + Date.now();
    document.getElementById('overlay').classList.add('open');
  }
  function closeModal() { document.getElementById('overlay').classList.remove('open'); }

  async function saveToFile() {
    const r = await fetch('/save'); const d = await r.json();
    showResult(d.message, false);
  }

  buildFrames();
  window.addEventListener('resize', () => { if (curFrame) resetZoom(); });
</script>
</body>
</html>
"""

# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────
@app.route("/")
def index():
    return render_template_string(HTML, frames=get_frames())

@app.route("/frame/<path:filename>")
def serve_frame(filename):
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        return f"Not found: {path}", 404
    return send_file(path, mimetype="image/png")

@app.route("/compute", methods=["POST"])
def compute():
    try:
        data   = request.get_json()
        entity = data["entity"]
        boxes  = data["boxes"]  # [{frame, pts: [{u, v}, {u, v}, ...]}, ...]

        if len(boxes) < 2:
            return jsonify({"error": "Need >=2 bounding polygons"})

        poses = load_poses()

        # Collect observations per corner across all frames
        tl, tr, bl, br = [], [], [], []
        for b in boxes:
            fk = str(b["frame"])
            pts = b["pts"]
            
            # Assuming clockwise order: Top-Left(0), Top-Right(1), Bottom-Right(2), Bottom-Left(3)
            tl.append((fk, pts[0]["u"], pts[0]["v"]))
            tr.append((fk, pts[1]["u"], pts[1]["v"]))
            br.append((fk, pts[2]["u"], pts[2]["v"])) 
            bl.append((fk, pts[3]["u"], pts[3]["v"]))

        # Triangulate all 4 corners in 3D
        p_tl = triangulate(tl, poses)
        p_tr = triangulate(tr, poses)
        p_bl = triangulate(bl, poses)
        p_br = triangulate(br, poses)

        # Center = mean of 4 corners
        center = (p_tl + p_tr + p_bl + p_br) / 4.0

        # Extents from average edge lengths (half-extents)
        width  = (np.linalg.norm(p_tr - p_tl) + np.linalg.norm(p_br - p_bl)) / 2.0
        height = (np.linalg.norm(p_bl - p_tl) + np.linalg.norm(p_br - p_tr)) / 2.0
        # Depth: can't recover from 2D, estimate from diagonal
        depth  = np.linalg.norm(p_br - p_tl) * 0.15

        extent = [float(width/2), float(height/2), float(depth/2)]

        # Reprojection: project center back onto each frame, compare to polygon center
        reproj = []
        for b in boxes:
            fk = str(b["frame"])
            pts = b["pts"]
            
            u_cx = sum(p["u"] for p in pts) / 4.0
            v_cx = sum(p["v"] for p in pts) / 4.0
            
            up, vp = project(center, fk, poses)
            if up is not None:
                err = float(np.sqrt((up - u_cx)**2 + (vp - v_cx)**2))
                reproj.append({"frame": fk, "error": err})

        save_answers(entity, center.tolist(), extent, ROTATION.tolist())
        return jsonify({"entity": entity, "center": center.tolist(),
                        "extent": extent, "rotation": ROTATION.tolist(),
                        "reprojection": reproj})

    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "trace": traceback.format_exc()})

@app.route("/validate")
def validate():
    entity    = request.args.get("entity", "ethernet_socket")
    frame_raw = request.args.get("frame", "")

    answers_path = os.path.join(DATA_DIR, "answers.json")
    if not os.path.exists(answers_path):
        return "answers.json not found — compute first", 404

    with open(answers_path) as f:
        answers = json.load(f)
    obb = next((a["obb"] for a in answers if a["entity"] == entity), None)
    if not obb:
        return f"No OBB for {entity}", 404

    center   = np.array(obb["center"])
    extent   = obb["extent"]
    rotation = np.array(obb["rotation"])
    corners  = get_corners_3d(center, extent, rotation)

    m = re.search(r'\d+', frame_raw)
    frame_num = int(m.group()) if m else 0
    img_path  = os.path.join(DATA_DIR, f"frame_{frame_num:06d}.png")
    if not os.path.exists(img_path):
        img_path = os.path.join(DATA_DIR, f"frame_{frame_num}.png")

    poses = load_poses()
    pil   = Image.open(img_path)
    draw  = ImageDraw.Draw(pil)
    W, H  = pil.size

    c2w = np.array(poses[str(frame_num)])
    w2c = np.linalg.inv(c2w)
    R, t = w2c[:3,:3], w2c[:3,3]
    P = K @ np.hstack([R, t.reshape(3,1)])

    def proj2d(pt):
        x = P @ np.append(pt, 1)
        return (int(x[0]/x[2]), int(x[1]/x[2]))

    pts   = [proj2d(c) for c in corners]
    cmap  = {"ethernet_socket":(60,143,255), "power_socket":(255,179,71), "vga_socket":(0,255,157)}
    color = cmap.get(entity, (0,255,157))
    edges = [(0,1),(1,3),(3,2),(2,0),(4,5),(5,7),(7,6),(6,4),(0,4),(1,5),(2,6),(3,7)]
    for p1, p2 in edges:
        draw.line([pts[p1], pts[p2]], fill=color, width=3)

    uc, vc = project(center, str(frame_num), poses)
    if uc and 0 < uc < W and 0 < vc < H:
        draw.ellipse([uc-8, vc-8, uc+8, vc+8], fill=color)
        draw.text((int(uc)+11, int(vc)), entity, fill=color)

    out = pil.resize((1280, 720))
    buf = io.BytesIO()
    out.save(buf, format="JPEG", quality=90)
    buf.seek(0)
    return send_file(buf, mimetype="image/jpeg")

@app.route("/save")
def save_route():
    path = os.path.join(DATA_DIR, "answers.json")
    if os.path.exists(path):
        return jsonify({"message": f"answers.json up to date at {path}"})
    return jsonify({"message": "Nothing saved yet — compute first."})

# ─────────────────────────────────────────────
if __name__ == "__main__":
    frames = get_frames()
    print(f"\n  Socket Annotator — 4-Point Mode")
    print(f"  Data dir : {DATA_DIR}")
    print(f"  Frames   : {len(frames)} found")
    print(f"  Open     : http://localhost:5000\n")
    app.run(debug=True, port=5000)