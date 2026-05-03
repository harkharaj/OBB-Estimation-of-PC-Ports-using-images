from flask import Flask, render_template_string, request, jsonify, send_file
import json, os, io
import numpy as np
from PIL import Image, ImageDraw, ImageFont

app = Flask(__name__)

# ─────────────────────────────────────────────
# CONFIG — edit these
# ─────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE_DIR, "Data")
CAM_DIR    = os.path.join(BASE_DIR, "Camera_Properties")
ANSWER_DIR = os.path.join(BASE_DIR, "Answers")

ROTATION = np.array([
    [-0.004004375172752437,  0.9672545151126772, -0.25377680739897346],
    [ 0.01584254528462312,   0.25380835519540434, 0.9671247761234889],
    [ 0.9998664804554559,   -0.00014774012094266402, -0.016340117333610394]
])

EXTENTS = {
    # Motherboard I/O
    "ethernet_socket":  [0.008,   0.0065,  0.0055],
    "power_socket":     [0.014,   0.011,   0.0075],
    "usb2_port":        [0.007,   0.0065,  0.006 ],
    "usb3_port":        [0.007,   0.008,   0.006 ],
    "ps2_port":         [0.0075,  0.0075,  0.005 ],
    "audio_jack_3_5mm": [0.006,   0.006,   0.005 ],
    "hdmi_port":        [0.0105,  0.0055,  0.006 ],
    "displayport":      [0.0105,  0.0065,  0.006 ],
    "vga_socket":       [0.0155,  0.0075,  0.007 ],
    "dvi_port":         [0.0185,  0.008,   0.007 ],
    # GPU (PCIe card)
    "gpu_hdmi":         [0.0105,  0.0055,  0.006 ],
    "gpu_displayport":  [0.0105,  0.0065,  0.006 ],
    # Expansion / misc
    "pcie_slot_bracket":[0.0025,  0.055,   0.001 ],
    "rear_fan_120mm":   [0.060,   0.060,   0.0125],
    "rear_fan_80mm":    [0.040,   0.040,   0.0125],
}

ENTITY_COLORS = {
    "ethernet_socket":  (60,  143, 255),
    "power_socket":     (255, 179,  71),
    "usb2_port":        (0,   220, 130),
    "usb3_port":        (0,   180, 255),
    "ps2_port":         (200, 100, 255),
    "audio_jack_3_5mm": (255,  80, 150),
    "hdmi_port":        (255, 220,  50),
    "displayport":      (80,  255, 200),
    "vga_socket":         (255,  80,  80),
    "dvi_port":         (200, 160,  80),
    "gpu_hdmi":         (255, 160,  60),
    "gpu_displayport":  (60,  220, 255),
    "pcie_slot_bracket":(160, 160, 160),
    "rear_fan_120mm":   (100, 220, 100),
    "rear_fan_80mm":    ( 80, 180,  80),
}

K = np.array([
    [1477.00974684544,   0.0,              1298.2501500778505],
    [0.0,                1480.4424455584467, 686.8201623541711],
    [0.0,                0.0,              1.0]
])

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def load_poses():
    with open(os.path.join(CAM_DIR, "poses.json")) as f:
        return json.load(f)

def triangulate(observations, poses):
    A = []
    for frame_key, u, v in observations:
        c2w = np.array(poses[str(frame_key)])
        w2c = np.linalg.inv(c2w)
        R, t = w2c[:3,:3], w2c[:3,3]
        P = K @ np.hstack([R, t.reshape(3,1)])
        A.append(v * P[2] - P[1])
        A.append(u * P[2] - P[0])
    A = np.array(A)
    _, _, Vt = np.linalg.svd(A)
    X = Vt[-1]
    return X[:3] / X[3]

def get_corners(center, extent, rotation):
    dx, dy, dz = extent
    pts = np.array([[sx*dx, sy*dy, sz*dz]
                    for sx in [-1,1] for sy in [-1,1] for sz in [-1,1]]).T
    return (rotation @ pts + np.array(center).reshape(3,1)).T  # 8×3

def project(world_pt, frame_key, poses):
    c2w = np.array(poses[str(frame_key)])
    w2c = np.linalg.inv(c2w)
    R, t = w2c[:3,:3], w2c[:3,3]
    cam = R @ np.array(world_pt) + t
    if cam[2] <= 0: return None, None
    return float(K[0,0]*cam[0]/cam[2] + K[0,2]), float(K[1,1]*cam[1]/cam[2] + K[1,2])

def get_frames():
    try:
        return sorted([f for f in os.listdir(DATA_DIR)
                       if f.lower().endswith(".png") and "frame" in f.lower()])
    except:
        return []

def save_answers(entity, center, extent, rotation):
    path = os.path.join(ANSWER_DIR, "answers.json")
    answers = []
    if os.path.exists(path):
        with open(path) as f:
            answers = json.load(f)
    for e in answers:
        if e["entity"] == entity:
            e["obb"] = {"center": center, "extent": extent, "rotation": rotation}
            break
    else:
        answers.append({"entity": entity, "obb": {"center": center, "extent": extent, "rotation": rotation}})
    with open(path, "w") as f:
        json.dump(answers, f, indent=2)

# ─────────────────────────────────────────────
# HTML
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
    --bg:      #0a0a0f;
    --surf:    #12121a;
    --border:  #1e1e2e;
    --acc:     #00ff9d;
    --acc2:    #ff3c6e;
    --acc3:    #3c8fff;
    --text:    #e0e0f0;
    --muted:   #555570;
    --mono:    'JetBrains Mono', monospace;
    --disp:    'Syne', sans-serif;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: var(--bg); color: var(--text); font-family: var(--mono); height: 100vh; display: grid; grid-template-rows: auto 1fr; overflow: hidden; }

  header {
    padding: 13px 28px; border-bottom: 1px solid var(--border);
    background: var(--surf); display: flex; align-items: center; gap: 16px;
  }
  header h1 { font-family: var(--disp); font-size: 1.15rem; font-weight: 800; color: var(--acc); }
  header span { color: var(--muted); font-size: 0.72rem; }

  .layout { display: grid; grid-template-columns: 240px 1fr 290px; overflow: hidden; }

  /* ── LEFT ── */
  .left { background: var(--surf); border-right: 1px solid var(--border); display: flex; flex-direction: column; overflow: hidden; }
  .sec-title { padding: 10px 14px 7px; font-size: 0.62rem; font-weight: 700; letter-spacing: 2px; text-transform: uppercase; color: var(--muted); border-bottom: 1px solid var(--border); }
  .frame-count { padding: 5px 14px; font-size: 0.65rem; color: var(--muted); border-bottom: 1px solid var(--border); }
  .frame-list { overflow-y: auto; flex: 1; }
  .frame-item { padding: 8px 14px; cursor: pointer; font-size: 0.73rem; color: var(--muted); border-left: 2px solid transparent; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; transition: all .15s; }
  .frame-item:hover { background: var(--border); color: var(--text); }
  .frame-item.active { background: rgba(0,255,157,.07); color: var(--acc); border-left-color: var(--acc); }

  /* entity + mode */
  .entity-bar { padding: 10px 12px; border-top: 1px solid var(--border); display: flex; flex-direction: column; gap: 7px; }
  .lbl { font-size: 0.62rem; color: var(--muted); letter-spacing: 1.5px; text-transform: uppercase; }
  select.ent-sel {
    background: var(--bg); color: var(--text); border: 1px solid var(--border);
    border-radius: 4px; padding: 7px 10px; font-family: var(--mono); font-size: 0.78rem;
    width: 100%; outline: none; cursor: pointer;
  }
  select.ent-sel:focus { border-color: var(--acc); }

  /* ── VIEWER ── */
  .viewer { position: relative; overflow: hidden; background: #05050a; cursor: crosshair; }
  #main-img { position: absolute; top: 0; left: 0; max-width: none; display: block; user-select: none; -webkit-user-drag: none; }
  #canvas { position: absolute; top: 0; left: 0; width: 100%; height: 100%; pointer-events: none; }
  .hud { position: absolute; top: 11px; left: 11px; background: rgba(10,10,15,.88); border: 1px solid var(--border); border-radius: 4px; padding: 5px 11px; font-size: 0.7rem; color: var(--muted); pointer-events: none; z-index: 10; }
  .hud span { color: var(--acc); }
  .ent-hud { position: absolute; top: 11px; right: 11px; border-radius: 4px; padding: 6px 14px; font-size: 0.74rem; font-weight: 700; letter-spacing: 1px; pointer-events: none; z-index: 10; border: 1px solid transparent; }
  .no-img { position: absolute; inset: 0; display: flex; align-items: center; justify-content: center; color: var(--muted); font-size: .85rem; pointer-events: none; }
  .zoom-btns { position: absolute; bottom: 14px; right: 14px; display: flex; gap: 5px; z-index: 10; }
  .zbtn { background: var(--surf); border: 1px solid var(--border); color: var(--text); width: 32px; height: 32px; border-radius: 4px; cursor: pointer; font-size: 1rem; display: flex; align-items: center; justify-content: center; transition: all .15s; }
  .zbtn:hover { border-color: var(--acc); color: var(--acc); }

  /* ── RIGHT ── */
  .right { background: var(--surf); border-left: 1px solid var(--border); display: flex; flex-direction: column; overflow: hidden; }
  .obs-list { flex: 1; overflow-y: auto; padding: 8px; display: flex; flex-direction: column; gap: 3px; }
  .obs-group-hdr { font-size: 0.6rem; letter-spacing: 1.5px; text-transform: uppercase; padding: 8px 4px 3px; border-bottom: 1px solid var(--border); margin-bottom: 2px; }
  .obs-item { background: var(--bg); border: 1px solid var(--border); border-radius: 4px; padding: 6px 10px; font-size: 0.7rem; display: flex; justify-content: space-between; align-items: center; animation: fi .18s ease; }
  @keyframes fi { from { opacity:0; transform:translateY(-3px); } to { opacity:1; transform:none; } }
  .obs-frame { font-weight: 700; }
  .obs-uv { color: var(--muted); font-size: 0.65rem; margin-left: 6px; }
  .del { background: none; border: none; color: var(--acc2); cursor: pointer; font-size: .78rem; opacity: .4; transition: opacity .15s; padding: 0 3px; }
  .del:hover { opacity: 1; }

  /* result */
  .result { margin: 0 10px 8px; background: var(--bg); border: 1px solid var(--border); border-radius: 4px; padding: 9px; font-size: 0.66rem; max-height: 130px; overflow-y: auto; white-space: pre-wrap; word-break: break-all; display: none; }
  .result.show { display: block; }

  /* actions */
  .actions { padding: 10px 11px; border-top: 1px solid var(--border); display: flex; flex-direction: column; gap: 7px; }
  .btn { padding: 9px 12px; border-radius: 4px; border: none; cursor: pointer; font-family: var(--mono); font-size: 0.76rem; font-weight: 700; transition: all .15s; width: 100%; letter-spacing: .4px; }
  .btn-go   { background: var(--acc); color: var(--bg); }
  .btn-go:hover { filter: brightness(1.12); }
  .btn-val  { background: transparent; border: 1px solid var(--acc3); color: var(--acc3); }
  .btn-val:hover { background: rgba(60,143,255,.1); }
  .btn-clr  { background: transparent; border: 1px solid var(--border); color: var(--muted); }
  .btn-clr:hover { border-color: var(--acc2); color: var(--acc2); }
  .btn-save { background: transparent; border: 1px solid var(--border); color: var(--muted); }
  .btn-save:hover { border-color: var(--acc); color: var(--acc); }

  /* validate frame picker */
  .val-row { display: flex; gap: 6px; }
  .val-row select { flex: 1; background: var(--bg); color: var(--text); border: 1px solid var(--border); border-radius: 4px; padding: 6px 8px; font-family: var(--mono); font-size: 0.72rem; outline: none; cursor: pointer; }
  .val-row select:focus { border-color: var(--acc3); }

  /* modal */
  .overlay { display: none; position: fixed; inset: 0; background: rgba(0,0,0,.8); z-index: 100; align-items: center; justify-content: center; }
  .overlay.open { display: flex; }
  .modal { background: var(--surf); border: 1px solid var(--border); border-radius: 8px; padding: 18px; max-width: 96vw; max-height: 96vh; display: flex; flex-direction: column; gap: 10px; }
  .modal h2 { font-family: var(--disp); font-size: .95rem; color: var(--acc); }
  .modal-hint { font-size: 0.65rem; color: var(--muted); }
  .val-viewport { overflow: hidden; position: relative; width: 80vw; height: 70vh; background: #05050a; border: 1px solid var(--border); border-radius: 4px; cursor: crosshair; }
  .val-viewport img { position: absolute; max-width: none; user-select: none; -webkit-user-drag: none; display: block; }
  .val-zoom-btns { position: absolute; bottom: 10px; right: 10px; display: flex; gap: 5px; z-index: 10; }
  .val-zoom-btns .zbtn { background: var(--surf); border: 1px solid var(--border); color: var(--text); width: 30px; height: 30px; border-radius: 4px; cursor: pointer; font-size: .95rem; display: flex; align-items: center; justify-content: center; }
  .modal-close { align-self: flex-end; background: var(--bg); border: 1px solid var(--border); color: var(--text); padding: 5px 14px; border-radius: 4px; cursor: pointer; font-family: var(--mono); font-size: .75rem; }
  .modal-close:hover { border-color: var(--acc2); color: var(--acc2); }

  ::-webkit-scrollbar { width: 4px; }
  ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 2px; }
</style>
</head>
<body>

<header>
  <h1>◈ Socket Annotator</h1>
  <span>Click socket center on ≥2 frames · Compute · Validate</span>
</header>

<div class="layout">

  <!-- LEFT -->
  <div class="left">
    <div class="sec-title">Frames</div>
    <div class="frame-count" id="fc">—</div>
    <div class="frame-list" id="frameList"></div>
    <div class="entity-bar">
      <div class="lbl">Active Entity</div>
      <select class="ent-sel" id="entSel" onchange="onEntityChange()">
        <!-- Filled dynamically -->
      </select>
    </div>
  </div>

  <!-- VIEWER -->
  <div class="viewer" id="viewer">
    <div class="no-img" id="noImg">← Select a frame</div>
    <img id="main-img" src="" alt="" style="display:none;">
    <canvas id="canvas"></canvas>
    <div class="hud">cursor: <span id="hx">—</span>, <span id="hy">—</span> &nbsp;|&nbsp; zoom: <span id="hz">1.0</span>x</div>
    <div class="ent-hud" id="entHud"></div>
    <div class="zoom-btns">
      <button class="zbtn" onclick="zoom(1.25)">+</button>
      <button class="zbtn" onclick="zoom(0.8)">−</button>
      <button class="zbtn" onclick="resetZoom()" title="fit">⊙</button>
    </div>
  </div>

  <!-- RIGHT -->
  <div class="right">
    <div class="sec-title">Observations</div>
    <div class="obs-list" id="obsList"><div style="padding:14px;color:var(--muted);font-size:.73rem;text-align:center;">Click the socket center on ≥2 frames</div></div>
    <div class="result" id="result"></div>
    <div class="actions">
      <button class="btn btn-go" onclick="compute()">▶ Triangulate + Compute OBB</button>
      <div class="val-row">
        <select id="valFrame"></select>
        <button class="btn btn-val" style="width:auto;padding:6px 12px;" onclick="validate()">⬡ Validate</button>
      </div>
      <button class="btn btn-clr" onclick="clearObs()">✕ Clear Observations</button>
      <button class="btn btn-save" onclick="saveToFile()">↓ Save to answers.json</button>
    </div>
  </div>

</div>

<!-- VALIDATE MODAL -->
<div class="overlay" id="overlay">
  <div class="modal">
    <h2>OBB Validation</h2>
    <div class="modal-hint">scroll = zoom &nbsp;|&nbsp; click+drag = pan &nbsp;|&nbsp; ⊙ = reset</div>
    <div class="val-viewport" id="valVP">
      <img id="valImg" src="" alt="">
      <div class="val-zoom-btns">
        <button class="zbtn" onclick="valZoom(1.25)">+</button>
        <button class="zbtn" onclick="valZoom(0.8)">−</button>
        <button class="zbtn" onclick="valResetZoom()">⊙</button>
      </div>
    </div>
    <button class="modal-close" onclick="closeModal()">Close</button>
  </div>
</div>

<script>
  let frames = {{ frames|tojson }};
  let entities = {{ entities|tojson }};
  let ENT_COLOR = {{ colors|tojson }};

  // obs[entity] = [{frame, u, v, file}, ...]
  let obs = {};
  entities.forEach(e => obs[e] = []);

  let lastResult = {};   // entity -> computed data
  let curFrame = null;
  let scale = 1, offset = {x:0, y:0};
  let dragging = false, dragStart = {x:0,y:0};
  let natW = 2560, natH = 1440;

  const viewer = document.getElementById('viewer');
  const img    = document.getElementById('main-img');
  const canvas = document.getElementById('canvas');
  const ctx    = canvas.getContext('2d');

  // ── Frame & Entity list ───────────────────────────────────────────────────
  function buildFrames() {
    const list = document.getElementById('frameList');
    const fc   = document.getElementById('fc');
    const vf   = document.getElementById('valFrame');
    list.innerHTML = ''; vf.innerHTML = '';
    if (!frames.length) {
      fc.textContent = '⚠ No frames found'; fc.style.color = 'var(--acc2)'; return;
    }
    fc.textContent = `${frames.length} frames`;
    frames.forEach(f => {
      const d = document.createElement('div');
      d.className = 'frame-item'; d.textContent = f; d.title = f;
      d.id = 'fi_' + f;
      d.onclick = () => loadFrame(f);
      list.appendChild(d);

      const o = document.createElement('option');
      o.value = f; o.textContent = f;
      vf.appendChild(o);
    });

    const entSel = document.getElementById('entSel');
    entities.forEach(e => {
        const opt = document.createElement('option');
        opt.value = e; opt.textContent = e;
        entSel.appendChild(opt);
    });
    onEntityChange(); // setup initial styling
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

  // ── Entity HUD ────────────────────────────────────────────────────────────
  function onEntityChange() {
    const ent = document.getElementById('entSel').value;
    const hud = document.getElementById('entHud');
    const col = ENT_COLOR[ent];
    hud.textContent = ent;
    // 26 in hex is ~15% opacity
    hud.style.backgroundColor = col + '26';
    hud.style.borderColor = col;
    hud.style.color = col;
    redraw();
  }

  // ── Zoom / pan ────────────────────────────────────────────────────────────
  function applyT() {
    img.style.left = offset.x + 'px'; img.style.top = offset.y + 'px';
    img.style.width = natW * scale + 'px'; img.style.height = natH * scale + 'px';
    document.getElementById('hz').textContent = scale.toFixed(2);
    redraw();
  }
  function zoom(f, cx, cy) {
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
    zoom(e.deltaY < 0 ? 1.12 : 0.89, e.clientX - r.left, e.clientY - r.top);
  }, {passive:false});
  viewer.addEventListener('mousedown', e => {
    if (e.button === 1 || e.altKey) {
      dragging = true; dragStart = {x: e.clientX - offset.x, y: e.clientY - offset.y};
      viewer.style.cursor = 'grabbing';
    }
  });
  window.addEventListener('mousemove', e => {
    if (dragging) { offset.x = e.clientX - dragStart.x; offset.y = e.clientY - dragStart.y; applyT(); }
    if (curFrame) {
      const r = viewer.getBoundingClientRect();
      const mx = (e.clientX - r.left - offset.x) / scale;
      const my = (e.clientY - r.top  - offset.y) / scale;
      if (mx >= 0 && mx <= natW && my >= 0 && my <= natH) {
        document.getElementById('hx').textContent = Math.round(mx);
        document.getElementById('hy').textContent = Math.round(my);
      }
    }
  });
  window.addEventListener('mouseup', () => { dragging = false; viewer.style.cursor = 'crosshair'; });

  // ── Click to pick ─────────────────────────────────────────────────────────
  viewer.addEventListener('click', e => {
    if (!curFrame || dragging || e.altKey || e.button !== 0) return;
    const r = viewer.getBoundingClientRect();
    const u = Math.round((e.clientX - r.left - offset.x) / scale);
    const v = Math.round((e.clientY - r.top  - offset.y) / scale);
    if (u < 0 || u > natW || v < 0 || v > natH) return;

    const ent = document.getElementById('entSel').value;
    const match = curFrame.match(/(\d+)/);
    const fn = match ? String(parseInt(match[1])) : curFrame;
    obs[ent].push({frame: fn, u, v, file: curFrame});
    renderObs(); redraw();
  });

  // ── Canvas draw ───────────────────────────────────────────────────────────
  function redraw() {
    const r = viewer.getBoundingClientRect();
    canvas.width = r.width; canvas.height = r.height;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (!curFrame) return;

    const cx = u => offset.x + u * scale;
    const cy = v => offset.y + v * scale;

    // Crosshair sizing:
    // - Zoomed out: fixed screen-px size so markers stay visible
    // - Zoomed in:  shrinks to ~1 image-pixel so you see the exact pixel clicked
    // Take the smaller of (fixed screen size) vs (N image-pixels * scale)
    const R   = Math.min(10,  0.5 * scale);   // circle radius in screen px
    const ARM = Math.min(22,  3.5 * scale);   // arm length in screen px
    const LW  = Math.max(1,   Math.min(1.5, 1.5 / scale));  // line width

    Object.entries(obs).forEach(([ent, list]) => {
      const color = ENT_COLOR[ent];
      const short = ent.substring(0,3).toUpperCase();
      list.filter(o => o.file === curFrame).forEach(o => {
        const x = cx(o.u), y = cy(o.v);
        ctx.strokeStyle = color; ctx.fillStyle = color;
        ctx.lineWidth = LW;
        // Arms
        ctx.beginPath();
        ctx.moveTo(x - ARM, y); ctx.lineTo(x - R, y);
        ctx.moveTo(x + R,   y); ctx.lineTo(x + ARM, y);
        ctx.moveTo(x, y - ARM); ctx.lineTo(x, y - R);
        ctx.moveTo(x, y + R);   ctx.lineTo(x, y + ARM);
        ctx.stroke();
        // Circle
        ctx.beginPath(); ctx.arc(x, y, Math.max(R, 1), 0, Math.PI*2); ctx.stroke();
        // Solid centre dot when zoomed in enough to matter
        if (scale > 4) {
          ctx.beginPath(); ctx.arc(x, y, 1.5, 0, Math.PI*2); ctx.fill();
        }
        // Label at fixed screen size, offset stays constant
        ctx.font = 'bold 11px JetBrains Mono, monospace';
        ctx.fillText(`${short} (${o.u},${o.v})`, x + Math.max(R, 7) + 4, y - 4);
      });
    });
  }

  // ── Obs list ──────────────────────────────────────────────────────────────
  function renderObs() {
    const list = document.getElementById('obsList');
    let total = 0;
    Object.values(obs).forEach(arr => total += arr.length);

    if (!total) {
      list.innerHTML = '<div style="padding:14px;color:var(--muted);font-size:.73rem;text-align:center;">Click the socket center on ≥2 frames</div>';
      return;
    }
    let html = '';
    entities.forEach(ent => {
      if (!obs[ent] || !obs[ent].length) return;
      const col = ENT_COLOR[ent];
      html += `<div class="obs-group-hdr" style="color:${col};">${ent} (${obs[ent].length}×)</div>`;
      obs[ent].forEach((o,i) => {
        html += `<div class="obs-item">
          <div><span class="obs-frame" style="color:${col};">f${o.frame}</span><span class="obs-uv">(${o.u}, ${o.v})</span></div>
          <button class="del" onclick="delObs('${ent}',${i})">✕</button>
        </div>`;
      });
    });
    list.innerHTML = html;
  }

  function delObs(ent, i) { obs[ent].splice(i,1); renderObs(); redraw(); }
  function clearObs() {
      entities.forEach(e => obs[e] = []);
      lastResult={}; renderObs(); redraw(); hideResult();
  }

  // ── Compute ───────────────────────────────────────────────────────────────
  async function compute() {
    const ent = document.getElementById('entSel').value;
    const list = obs[ent];
    if (list.length < 2) { showResult(`⚠ Need ≥2 observations for ${ent}. Have ${list.length}.`, true); return; }

    showResult('Triangulating…', false);
    try {
      const res  = await fetch('/compute', {
        method: 'POST',
        headers: {'Content-Type':'application/json'},
        body: JSON.stringify({ entity: ent, observations: list.map(o => [o.frame, o.u, o.v]) })
      });
      const data = await res.json();
      if (data.error) { showResult('Error: ' + data.error, true); return; }
      lastResult[ent] = data;
      showResult(fmt(data), false);
    } catch(e) { showResult('Error: ' + e.message, true); }
  }

  function fmt(d) {
    return [
      `entity : ${d.entity}`,
      `center : [${d.center.map(x=>x.toFixed(5)).join(', ')}]`,
      `extent : [${d.extent.map(x=>x.toFixed(5)).join(', ')}]`,
      '',
      'reprojection errors:',
      ...d.reprojection.map(r => `  f${r.frame}: ${r.error.toFixed(1)}px`),
      '',
      '✓ auto-saved to answers.json'
    ].join('\n');
  }

  function showResult(txt, err) {
    const el = document.getElementById('result');
    el.textContent = txt;
    el.style.color = err ? 'var(--acc2)' : 'var(--acc)';
    el.classList.add('show');
  }
  function hideResult() { document.getElementById('result').classList.remove('show'); }

  // ── Validate ──────────────────────────────────────────────────────────────
  function validate() {
    const ent   = document.getElementById('entSel').value;
    const frame = document.getElementById('valFrame').value;
    if (!lastResult[ent]) { showResult(`⚠ Compute OBB for ${ent} first.`, true); return; }
    document.getElementById('valImg').src = `/validate?entity=${ent}&frame=${encodeURIComponent(frame)}&t=${Date.now()}`;
    document.getElementById('overlay').classList.add('open');
  }
  // ── Validate modal zoom / pan ─────────────────────────────────────────────
  let vScale = 1, vOff = {x:0,y:0}, vDrag = false, vDragS = {x:0,y:0};
  let vNatW = 0, vNatH = 0;
  const vVP  = document.getElementById('valVP');
  const vImg = document.getElementById('valImg');

  function valApplyT() {
    vImg.style.left   = vOff.x + 'px';
    vImg.style.top    = vOff.y + 'px';
    vImg.style.width  = vNatW * vScale + 'px';
    vImg.style.height = vNatH * vScale + 'px';
  }
  function valZoom(f, cx, cy) {
    const r = vVP.getBoundingClientRect();
    cx = cx ?? r.width/2; cy = cy ?? r.height/2;
    const prev = vScale;
    vScale = Math.min(Math.max(vScale * f, 0.05), 20);
    vOff.x = cx - (cx - vOff.x) * (vScale/prev);
    vOff.y = cy - (cy - vOff.y) * (vScale/prev);
    valApplyT();
  }
  function valResetZoom() {
    // Fit image to viewport (same logic as annotation resetZoom)
    const r = vVP.getBoundingClientRect();
    if (!r.width || !vNatW) { requestAnimationFrame(valResetZoom); return; }
    vScale = Math.min(r.width / vNatW, r.height / vNatH) * 0.97;
    vOff.x = (r.width  - vNatW * vScale) / 2;
    vOff.y = (r.height - vNatH * vScale) / 2;
    valApplyT();
  }
  // When a new validate image loads, capture its natural size and fit it
  vImg.addEventListener('load', () => {
    vNatW = vImg.naturalWidth;
    vNatH = vImg.naturalHeight;
    valResetZoom();
  });
  vVP.addEventListener('wheel', e => {
    e.preventDefault();
    const r = vVP.getBoundingClientRect();
    valZoom(e.deltaY < 0 ? 1.15 : 0.87, e.clientX - r.left, e.clientY - r.top);
  }, {passive: false});
  vVP.addEventListener('mousedown', e => {
    vDrag = true; vDragS = {x: e.clientX - vOff.x, y: e.clientY - vOff.y};
    vVP.style.cursor = 'grabbing'; e.preventDefault();
  });
  window.addEventListener('mousemove', e => {
    if (vDrag) { vOff.x = e.clientX - vDragS.x; vOff.y = e.clientY - vDragS.y; valApplyT(); }
  });
  window.addEventListener('mouseup', () => { vDrag = false; vVP.style.cursor = 'crosshair'; });

  function closeModal() {
    document.getElementById('overlay').classList.remove('open');
    vScale = 1; vOff = {x:0,y:0}; vNatW = 0; vNatH = 0;
    vImg.style.left = '0px'; vImg.style.top = '0px';
    vImg.style.width = ''; vImg.style.height = '';
  }

  // ── Save ──────────────────────────────────────────────────────────────────
  async function saveToFile() {
    const res = await fetch('/save'); const d = await res.json();
    showResult(d.message, false);
  }

  // ── Init ──────────────────────────────────────────────────────────────────
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
    # Convert RGB tuples into Hex for the frontend CSS
    hex_colors = {k: f"#{r:02x}{g:02x}{b:02x}" for k, (r, g, b) in ENTITY_COLORS.items()}
    return render_template_string(HTML, frames=get_frames(), entities=list(EXTENTS.keys()), colors=hex_colors)

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
        raw    = data["observations"]  # [[frame, u, v], ...]
        if len(raw) < 2:
            return jsonify({"error": "Need ≥2 observations"})

        obs = [(str(r[0]), float(r[1]), float(r[2])) for r in raw]
        poses  = load_poses()
        center = triangulate(obs, poses)
        extent = EXTENTS.get(entity, [0.008, 0.0065, 0.0055])

        # reprojection errors
        reproj = []
        for fk, u_obs, v_obs in obs:
            up, vp = project(center, fk, poses)
            if up is not None:
                reproj.append({"frame": fk, "error": float(np.sqrt((up-u_obs)**2 + (vp-v_obs)**2))})

        save_answers(entity, center.tolist(), extent, ROTATION.tolist())
        return jsonify({"entity": entity, "center": center.tolist(), "extent": extent,
                        "rotation": ROTATION.tolist(), "reprojection": reproj})
    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "trace": traceback.format_exc()})

@app.route("/validate")
def validate():
    entity    = request.args.get("entity", "ethernet_socket")
    frame_raw = request.args.get("frame", "")

    answers_path = os.path.join(ANSWER_DIR, "answers.json")
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
    corners  = get_corners(center, extent, rotation)

    # parse frame number from filename or plain number
    m = __import__('re').search(r'\d+', frame_raw)
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
    color = ENTITY_COLORS.get(entity, (255, 255, 255))
    edges = [(0,1),(1,3),(3,2),(2,0),(4,5),(5,7),(7,6),(6,4),(0,4),(1,5),(2,6),(3,7)]
    for p1, p2 in edges:
        draw.line([pts[p1], pts[p2]], fill=color, width=3)

    # center dot
    uc, vc = project(center, str(frame_num), poses)
    if uc and 0 < uc < W and 0 < vc < H:
        draw.ellipse([uc-9, vc-9, uc+9, vc+9], fill=color)
        draw.text((int(uc)+12, int(vc)), entity, fill=color)

    out = pil  # serve full resolution so zoom is useful
    buf = io.BytesIO()
    out.save(buf, format="JPEG", quality=90)
    buf.seek(0)
    return send_file(buf, mimetype="image/jpeg")

@app.route("/save")
def save_route():
    path = os.path.join(ANSWER_DIR, "answers.json")
    if os.path.exists(path):
        return jsonify({"message": f"✓ answers.json up to date at {path}"})
    return jsonify({"message": "Nothing saved yet — compute first."})

# ─────────────────────────────────────────────
if __name__ == "__main__":
    frames = get_frames()
    print(f"\n  Socket Annotator")
    print(f"  Data dir : {DATA_DIR}")
    print(f"  Frames   : {len(frames)} found")
    print(f"  Entities : {len(EXTENTS)} configured")
    print(f"  Open     : http://localhost:5000\n")
    app.run(debug=True, port=5000)
