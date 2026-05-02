"""
calibration_app.py
------------------
Flask web app for picking pixel coordinates on frames and computing OBBs.

Workflow:
  - Click Point A (VGA center), Point B (Ethernet center), Point C (Power center)
    across multiple frames.
  - Triangulates all three points.
  - Derives panel rotation from A→B (horizontal) and A→C (vertical) vectors.
  - Computes per-entity OBB centers relative to the panel.

Usage:
    pip install flask numpy pillow
    python calibration_app.py
    Open http://localhost:5000
"""

from flask import Flask, render_template_string, request, jsonify, send_file
import json
import numpy as np
import os
import io
from PIL import Image, ImageDraw

app = Flask(__name__)

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
DATA_DIR = r"C:\Users\harkh\OneDrive\Desktop\ROBOTIC_PERCEPTION_FINAL_PROJECT\Data"

VGA_CENTER = np.array([0.2704921202927293, 0.2261220732082181, 0.8349008829378597])
VGA_ROTATION_GT = np.array([
    [-0.004004375172752437,  0.9672545151126772, -0.25377680739897346],
    [ 0.01584254528462312,   0.25380835519540434, 0.9671247761234889],
    [ 0.9998664804554559,   -0.00014774012094266402, -0.016340117333610394]
])

KNOWN_EXTENTS = {
    "vga_socket":      [0.03537766175069747, 0.011822199241650923, 0.0061316691090621735],
    "ethernet_socket": [0.008,  0.0065, 0.0055],
    "power_socket":    [0.014,  0.011,  0.0075],
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
    with open(os.path.join(DATA_DIR, "poses.json")) as f:
        return json.load(f)

def triangulate(observations, poses):
    A = []
    for frame_key, u, v in observations:
        c2w = np.array(poses[str(frame_key)])
        w2c = np.linalg.inv(c2w)
        R = w2c[:3, :3]; t = w2c[:3, 3]
        P = K @ np.hstack([R, t.reshape(3,1)])
        A.append(v * P[2] - P[1])
        A.append(u * P[2] - P[0])
    A = np.array(A)
    _, _, Vt = np.linalg.svd(A)
    X = Vt[-1]
    return X[:3] / X[3]

def derive_rotation_from_points(pt_A, pt_B, pt_C):
    """
    Derive panel rotation from three triangulated 3-D points:
      pt_A = VGA center        (origin)
      pt_B = Ethernet center   (to the right of VGA  → defines horizontal axis)
      pt_C = Power center      (below the others     → defines down direction)

    Returns a 3×3 rotation matrix whose columns are [right, down, normal].
    """
    horiz = np.array(pt_B) - np.array(pt_A)
    horiz /= np.linalg.norm(horiz)

    down_raw = np.array(pt_C) - np.array(pt_A)
    normal = np.cross(horiz, down_raw)
    normal /= np.linalg.norm(normal)

    # Make sure normal points roughly toward the camera cluster (positive Z-ish)
    # Flip if needed based on dot with world-up or a known camera direction
    vert = np.cross(normal, horiz)
    vert /= np.linalg.norm(vert)

    # Columns: [horizontal-right, vertical-down, panel-normal]
    return np.column_stack([horiz, vert, normal])

def get_obb_corners(center, extent, rotation):
    dx, dy, dz = extent
    corners = np.array([
        [sx*dx, sy*dy, sz*dz]
        for sx in [-1,1] for sy in [-1,1] for sz in [-1,1]
    ]).T
    return (rotation @ corners + np.array(center).reshape(3,1)).T

def project_point(world_pt, frame_key, poses):
    c2w = np.array(poses[str(frame_key)])
    w2c = np.linalg.inv(c2w)
    R = w2c[:3,:3]; t = w2c[:3,3]
    cam = R @ np.array(world_pt) + t
    if cam[2] <= 0: return None, None
    u = K[0,0]*cam[0]/cam[2] + K[0,2]
    v = K[1,1]*cam[1]/cam[2] + K[1,2]
    return float(u), float(v)

def get_frame_list():
    """Return sorted list of frame PNGs regardless of zero-padding style."""
    try:
        files = sorted([
            f for f in os.listdir(DATA_DIR)
            if f.lower().endswith(".png") and "frame" in f.lower()
        ])
        return files
    except Exception:
        return []

# ─────────────────────────────────────────────
# HTML TEMPLATE
# ─────────────────────────────────────────────
HTML = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>OBB Calibration Tool</title>
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Syne:wght@400;700;800&display=swap" rel="stylesheet">
<style>
  :root {
    --bg:       #0a0a0f;
    --surface:  #12121a;
    --border:   #1e1e2e;
    --accent:   #00ff9d;
    --accent2:  #ff3c6e;
    --accent3:  #3c8fff;
    --accent4:  #ffb347;
    --text:     #e0e0f0;
    --muted:    #555570;
    --font-mono: 'JetBrains Mono', monospace;
    --font-display: 'Syne', sans-serif;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    background: var(--bg);
    color: var(--text);
    font-family: var(--font-mono);
    min-height: 100vh;
    display: grid;
    grid-template-rows: auto 1fr;
  }
  header {
    padding: 14px 32px;
    border-bottom: 1px solid var(--border);
    display: flex;
    align-items: center;
    gap: 20px;
    background: var(--surface);
  }
  header h1 {
    font-family: var(--font-display);
    font-size: 1.2rem;
    font-weight: 800;
    color: var(--accent);
  }
  header span { color: var(--muted); font-size: 0.72rem; }

  .layout {
    display: grid;
    grid-template-columns: 260px 1fr 300px;
    height: calc(100vh - 53px);
    overflow: hidden;
  }

  /* LEFT PANEL */
  .left-panel {
    background: var(--surface);
    border-right: 1px solid var(--border);
    display: flex;
    flex-direction: column;
    overflow: hidden;
  }
  .panel-title {
    padding: 12px 14px 8px;
    font-family: var(--font-display);
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: var(--muted);
    border-bottom: 1px solid var(--border);
  }
  .frame-list { overflow-y: auto; flex: 1; padding: 6px 0; }
  .frame-item {
    padding: 8px 14px;
    cursor: pointer;
    font-size: 0.74rem;
    color: var(--muted);
    transition: all 0.15s;
    border-left: 2px solid transparent;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  .frame-item:hover { background: var(--border); color: var(--text); }
  .frame-item.active {
    background: rgba(0,255,157,0.07);
    color: var(--accent);
    border-left-color: var(--accent);
  }
  .frame-count {
    padding: 6px 14px;
    font-size: 0.65rem;
    color: var(--muted);
    border-bottom: 1px solid var(--border);
  }

  /* POINT MODE SELECTOR */
  .point-bar {
    padding: 10px 12px;
    border-top: 1px solid var(--border);
    display: flex;
    flex-direction: column;
    gap: 6px;
  }
  .point-bar-label {
    font-size: 0.65rem;
    color: var(--muted);
    letter-spacing: 1.5px;
    text-transform: uppercase;
    margin-bottom: 2px;
  }
  .point-btns { display: flex; gap: 5px; }
  .point-btn {
    flex: 1;
    padding: 8px 4px;
    border-radius: 4px;
    border: 1px solid var(--border);
    background: var(--bg);
    color: var(--muted);
    font-family: var(--font-mono);
    font-size: 0.72rem;
    font-weight: 700;
    cursor: pointer;
    transition: all 0.15s;
    text-align: center;
  }
  .point-btn:hover { border-color: var(--text); color: var(--text); }
  .point-btn.active-A { border-color: var(--accent); color: var(--accent); background: rgba(0,255,157,0.08); }
  .point-btn.active-B { border-color: var(--accent3); color: var(--accent3); background: rgba(60,143,255,0.08); }
  .point-btn.active-C { border-color: var(--accent4); color: var(--accent4); background: rgba(255,179,71,0.08); }

  .point-legend {
    font-size: 0.65rem;
    color: var(--muted);
    line-height: 1.6;
    padding: 4px 0 2px;
  }
  .point-legend .pa { color: var(--accent); }
  .point-legend .pb { color: var(--accent3); }
  .point-legend .pc { color: var(--accent4); }

  /* CENTER: IMAGE VIEWER */
  .image-viewer {
    position: relative;
    overflow: hidden;
    background: #05050a;
    cursor: crosshair;
  }
  #canvas {
    position: absolute;
    top: 0; left: 0;
    width: 100%; height: 100%;
    pointer-events: none;
  }
  #main-img {
    position: absolute;
    top: 0; left: 0;
    max-width: none;
    transform-origin: 0 0;
    display: block;
    user-select: none;
    -webkit-user-drag: none;
  }
  .zoom-controls {
    position: absolute;
    bottom: 16px; right: 16px;
    display: flex; gap: 6px; z-index: 10;
  }
  .zoom-btn {
    background: var(--surface);
    border: 1px solid var(--border);
    color: var(--text);
    width: 34px; height: 34px;
    border-radius: 4px;
    cursor: pointer; font-size: 1.1rem;
    display: flex; align-items: center; justify-content: center;
    transition: all 0.15s;
  }
  .zoom-btn:hover { border-color: var(--accent); color: var(--accent); }
  .coords-hud {
    position: absolute;
    top: 12px; left: 12px;
    background: rgba(10,10,15,0.85);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 6px 12px;
    font-size: 0.72rem;
    color: var(--muted);
    pointer-events: none;
    z-index: 10;
  }
  .coords-hud span { color: var(--accent); }
  /* Active point mode indicator */
  .mode-hud {
    position: absolute;
    top: 12px; right: 12px;
    border-radius: 4px;
    padding: 6px 14px;
    font-size: 0.75rem;
    font-weight: 700;
    pointer-events: none;
    z-index: 10;
    letter-spacing: 1px;
  }
  .mode-hud.A { background: rgba(0,255,157,0.15); border: 1px solid var(--accent); color: var(--accent); }
  .mode-hud.B { background: rgba(60,143,255,0.15); border: 1px solid var(--accent3); color: var(--accent3); }
  .mode-hud.C { background: rgba(255,179,71,0.15); border: 1px solid var(--accent4); color: var(--accent4); }
  .no-image { color: var(--muted); font-size: 0.85rem; text-align: center; pointer-events: none; }

  /* RIGHT PANEL */
  .right-panel {
    background: var(--surface);
    border-left: 1px solid var(--border);
    display: flex;
    flex-direction: column;
    overflow: hidden;
  }
  .obs-list {
    flex: 1;
    overflow-y: auto;
    padding: 8px;
    display: flex;
    flex-direction: column;
    gap: 4px;
  }
  .obs-group-title {
    font-size: 0.62rem;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    padding: 6px 4px 2px;
    border-bottom: 1px solid var(--border);
    margin-bottom: 2px;
  }
  .obs-group-title.A { color: var(--accent); }
  .obs-group-title.B { color: var(--accent3); }
  .obs-group-title.C { color: var(--accent4); }
  .obs-item {
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 7px 10px;
    font-size: 0.7rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
    animation: fadeIn 0.2s ease;
  }
  @keyframes fadeIn { from { opacity:0; transform: translateY(-4px); } to { opacity:1; transform: none; } }
  .obs-frame { font-weight: 700; }
  .obs-frame.A { color: var(--accent); }
  .obs-frame.B { color: var(--accent3); }
  .obs-frame.C { color: var(--accent4); }
  .obs-coords { color: var(--muted); font-size: 0.65rem; }
  .del-btn {
    background: none; border: none; color: var(--accent2);
    cursor: pointer; font-size: 0.8rem; padding: 0 4px;
    opacity: 0.4; transition: opacity 0.15s;
  }
  .del-btn:hover { opacity: 1; }

  /* Progress indicator */
  .progress-bar {
    padding: 8px 12px;
    border-bottom: 1px solid var(--border);
    display: flex;
    gap: 6px;
    align-items: center;
    font-size: 0.68rem;
  }
  .prog-dot {
    width: 10px; height: 10px;
    border-radius: 50%;
    border: 1.5px solid var(--border);
    flex-shrink: 0;
  }
  .prog-dot.filled-A { background: var(--accent); border-color: var(--accent); }
  .prog-dot.filled-B { background: var(--accent3); border-color: var(--accent3); }
  .prog-dot.filled-C { background: var(--accent4); border-color: var(--accent4); }
  .prog-text { color: var(--muted); }

  .action-area {
    padding: 12px;
    border-top: 1px solid var(--border);
    display: flex;
    flex-direction: column;
    gap: 8px;
  }
  .btn {
    padding: 10px 14px; border-radius: 4px; border: none;
    cursor: pointer; font-family: var(--font-mono); font-size: 0.78rem;
    font-weight: 700; letter-spacing: 0.5px; transition: all 0.15s; width: 100%;
  }
  .btn-primary { background: var(--accent); color: var(--bg); }
  .btn-primary:hover { filter: brightness(1.15); }
  .btn-secondary { background: transparent; border: 1px solid var(--border); color: var(--muted); }
  .btn-secondary:hover { border-color: var(--accent2); color: var(--accent2); }
  .btn-validate { background: transparent; border: 1px solid var(--accent3); color: var(--accent3); }
  .btn-validate:hover { background: rgba(60,143,255,0.1); }

  .result-box {
    margin: 0 12px 8px;
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 10px;
    font-size: 0.67rem;
    color: var(--muted);
    max-height: 140px;
    overflow-y: auto;
    white-space: pre-wrap;
    word-break: break-all;
    display: none;
  }
  .result-box.visible { display: block; }

  /* ENTITY selector for OBB compute (right panel bottom) */
  .compute-entity-row {
    display: flex;
    gap: 6px;
    align-items: center;
    margin-bottom: 2px;
  }
  .entity-select {
    flex: 1;
    background: var(--bg);
    color: var(--text);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 6px 8px;
    font-family: var(--font-mono);
    font-size: 0.74rem;
    outline: none;
    cursor: pointer;
  }
  .entity-select:focus { border-color: var(--accent); }

  /* VALIDATE MODAL */
  .modal-overlay {
    display: none; position: fixed; inset: 0;
    background: rgba(0,0,0,0.8); z-index: 100;
    align-items: center; justify-content: center;
  }
  .modal-overlay.open { display: flex; }
  .modal {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 8px; padding: 20px;
    max-width: 92vw; max-height: 92vh;
    overflow: auto; display: flex; flex-direction: column; gap: 12px;
  }
  .modal h2 { font-family: var(--font-display); font-size: 1rem; color: var(--accent); }
  .modal img { max-width: 100%; border-radius: 4px; border: 1px solid var(--border); }
  .modal-close {
    align-self: flex-end; background: var(--bg); border: 1px solid var(--border);
    color: var(--text); padding: 6px 14px; border-radius: 4px;
    cursor: pointer; font-family: var(--font-mono); font-size: 0.78rem;
  }
  .modal-close:hover { border-color: var(--accent2); color: var(--accent2); }

  ::-webkit-scrollbar { width: 4px; }
  ::-webkit-scrollbar-track { background: transparent; }
  ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 2px; }
</style>
</head>
<body>

<header>
  <h1>⬡ OBB Calibration Tool</h1>
  <span>Click A/B/C points across frames → Compute OBBs · Scroll to zoom · Alt+drag to pan</span>
</header>

<div class="layout">

  <!-- LEFT: frame list + point selector -->
  <div class="left-panel">
    <div class="panel-title">Frames</div>
    <div class="frame-count" id="frameCount">Loading…</div>
    <div class="frame-list" id="frameList"></div>

    <div class="point-bar">
      <div class="point-bar-label">Active Point Mode</div>
      <div class="point-btns">
        <button class="point-btn active-A" id="btnA" onclick="setMode('A')">A · VGA</button>
        <button class="point-btn" id="btnB" onclick="setMode('B')">B · ETH</button>
        <button class="point-btn" id="btnC" onclick="setMode('C')">C · PWR</button>
      </div>
      <div class="point-legend">
        <span class="pa">A</span> — VGA port center<br>
        <span class="pb">B</span> — Ethernet port center (right of VGA)<br>
        <span class="pc">C</span> — Power port center (below)
      </div>
    </div>
  </div>

  <!-- CENTER: image viewer -->
  <div class="image-viewer" id="viewer">
    <div class="no-image" id="noImage">← Select a frame to begin</div>
    <img id="main-img" src="" alt="" style="display:none;">
    <canvas id="canvas"></canvas>
    <div class="coords-hud" id="coordsHud">
      cursor: <span id="hudX">—</span>, <span id="hudY">—</span>
      &nbsp;|&nbsp; zoom: <span id="hudZoom">1.0</span>x
    </div>
    <div class="mode-hud A" id="modeHud">MODE · A (VGA)</div>
    <div class="zoom-controls">
      <button class="zoom-btn" onclick="adjustZoom(1.25)">+</button>
      <button class="zoom-btn" onclick="adjustZoom(0.8)">−</button>
      <button class="zoom-btn" onclick="resetZoom()" title="Reset">⊙</button>
    </div>
  </div>

  <!-- RIGHT: observations + actions -->
  <div class="right-panel">
    <div class="panel-title">3-Point Observations</div>

    <!-- progress dots -->
    <div class="progress-bar" id="progressBar">
      <div class="prog-dot" id="dotA"></div>
      <span class="prog-text" style="color:var(--accent); font-size:0.65rem;">A</span>
      <div class="prog-dot" id="dotB"></div>
      <span class="prog-text" style="color:var(--accent3); font-size:0.65rem;">B</span>
      <div class="prog-dot" id="dotC"></div>
      <span class="prog-text" style="color:var(--accent4); font-size:0.65rem;">C</span>
      <span class="prog-text" id="progMsg" style="margin-left:6px;"></span>
    </div>

    <div class="obs-list" id="obsList">
      <div style="padding:16px;color:var(--muted);font-size:0.74rem;text-align:center;">
        Select a frame and click A, B, C points
      </div>
    </div>

    <div class="result-box" id="resultBox"></div>

    <div class="action-area">
      <div class="compute-entity-row">
        <select class="entity-select" id="entitySelect">
          <option value="vga_socket">vga_socket</option>
          <option value="ethernet_socket">ethernet_socket</option>
          <option value="power_socket">power_socket</option>
        </select>
      </div>
      <button class="btn btn-primary" onclick="computeOBB()">▶ Compute OBB</button>
      <button class="btn btn-validate" onclick="openValidateModal()">⬡ Validate on Frame</button>
      <button class="btn btn-secondary" onclick="clearAll()">✕ Clear All</button>
      <button class="btn btn-secondary" onclick="saveAnswers()">↓ Save to answers.json</button>
    </div>
  </div>

</div>

<!-- VALIDATE MODAL -->
<div class="modal-overlay" id="modalOverlay">
  <div class="modal">
    <h2>OBB Projection Validation</h2>
    <img id="validateImg" src="" alt="validation">
    <button class="modal-close" onclick="closeModal()">Close</button>
  </div>
</div>

<script>
  // ── State ──────────────────────────────────────────────────────────────────
  let frames = {{ frames|tojson }};
  let currentFrame = null;
  // observations per point type: { A: [{frame, u, v, file}], B: [...], C: [...] }
  let obs = { A: [], B: [], C: [] };
  let activeMode = 'A';
  let scale = 1.0;
  let offset = {x: 0, y: 0};
  let dragging = false;
  let dragStart = {x: 0, y: 0};
  let naturalW = 2560, naturalH = 1440;
  let lastOBBs = {};   // entity -> obb data

  const viewer  = document.getElementById('viewer');
  const img     = document.getElementById('main-img');
  const canvas  = document.getElementById('canvas');
  const ctx     = canvas.getContext('2d');

  const COLORS = { A: '#00ff9d', B: '#3c8fff', C: '#ffb347' };
  const LABELS = { A: 'VGA', B: 'ETH', C: 'PWR' };

  // ── Frame list ─────────────────────────────────────────────────────────────
  function buildFrameList() {
    const list = document.getElementById('frameList');
    const countEl = document.getElementById('frameCount');
    list.innerHTML = '';
    if (frames.length === 0) {
      countEl.textContent = '⚠ No frames found in DATA_DIR';
      countEl.style.color = 'var(--accent2)';
      list.innerHTML = '<div style="padding:12px 14px;color:var(--accent2);font-size:0.72rem;">Check DATA_DIR path in script</div>';
      return;
    }
    countEl.textContent = `${frames.length} frame${frames.length !== 1 ? 's' : ''} found`;
    frames.forEach(f => {
      const div = document.createElement('div');
      div.className = 'frame-item';
      div.textContent = f;
      div.title = f;
      div.onclick = () => loadFrame(f);
      div.id = 'fi_' + f;
      list.appendChild(div);
    });
  }

  function loadFrame(fname) {
    currentFrame = fname;
    document.querySelectorAll('.frame-item').forEach(el => el.classList.remove('active'));
    const el = document.getElementById('fi_' + fname);
    if (el) { el.classList.add('active'); el.scrollIntoView({block:'nearest'}); }
    img.src = '/frame/' + fname;
    img.style.display = 'block';
    document.getElementById('noImage').style.display = 'none';
    img.onload = () => {
      naturalW = img.naturalWidth;
      naturalH = img.naturalHeight;
      resetZoom();
      redrawCanvas();
    };
  }

  // ── Mode selector ──────────────────────────────────────────────────────────
  function setMode(m) {
    activeMode = m;
    ['A','B','C'].forEach(x => {
      const btn = document.getElementById('btn' + x);
      btn.className = 'point-btn' + (x === m ? ` active-${x}` : '');
    });
    const modeNames = { A: 'A (VGA)', B: 'B (ETH)', C: 'C (PWR)' };
    const hud = document.getElementById('modeHud');
    hud.textContent = `MODE · ${modeNames[m]}`;
    hud.className = `mode-hud ${m}`;
  }

  // ── Zoom & pan ─────────────────────────────────────────────────────────────
  function applyTransform() {
    img.style.left   = offset.x + 'px';
    img.style.top    = offset.y + 'px';
    img.style.width  = (naturalW * scale) + 'px';
    img.style.height = (naturalH * scale) + 'px';
    // Canvas covers the full viewer, not the image — we map coords in draw
    const rect = viewer.getBoundingClientRect();
    canvas.style.left = '0px';
    canvas.style.top  = '0px';
    canvas.width  = rect.width  || canvas.width;
    canvas.height = rect.height || canvas.height;
    document.getElementById('hudZoom').textContent = scale.toFixed(2);
  }
  function adjustZoom(factor, cx, cy) {
    const rect = viewer.getBoundingClientRect();
    cx = cx ?? rect.width / 2;  cy = cy ?? rect.height / 2;
    const prev = scale;
    scale = Math.min(Math.max(scale * factor, 0.1), 20);
    offset.x = cx - (cx - offset.x) * (scale / prev);
    offset.y = cy - (cy - offset.y) * (scale / prev);
    applyTransform();
  }
  function resetZoom() {
    const rect = viewer.getBoundingClientRect();
    if (rect.width === 0 || rect.height === 0) {
      requestAnimationFrame(resetZoom);
      return;
    }
    const fit = Math.min(rect.width / naturalW, rect.height / naturalH) * 0.95;
    scale = fit;
    offset.x = (rect.width  - naturalW * scale) / 2;
    offset.y = (rect.height - naturalH * scale) / 2;
    applyTransform(); redrawCanvas();
  }
  viewer.addEventListener('wheel', e => {
    e.preventDefault();
    const rect = viewer.getBoundingClientRect();
    adjustZoom(e.deltaY < 0 ? 1.12 : 0.89, e.clientX - rect.left, e.clientY - rect.top);
    redrawCanvas();
  }, {passive: false});
  viewer.addEventListener('mousedown', e => {
    if (e.button === 1 || e.altKey) {
      dragging = true;
      dragStart = {x: e.clientX - offset.x, y: e.clientY - offset.y};
      viewer.style.cursor = 'grabbing';
    }
  });
  window.addEventListener('mousemove', e => {
    if (dragging) {
      offset.x = e.clientX - dragStart.x;
      offset.y = e.clientY - dragStart.y;
      applyTransform(); redrawCanvas();
    }
    if (currentFrame) {
      const rect = viewer.getBoundingClientRect();
      const mx = (e.clientX - rect.left - offset.x) / scale;
      const my = (e.clientY - rect.top  - offset.y) / scale;
      if (mx >= 0 && mx <= naturalW && my >= 0 && my <= naturalH) {
        document.getElementById('hudX').textContent = Math.round(mx);
        document.getElementById('hudY').textContent = Math.round(my);
      }
    }
  });
  window.addEventListener('mouseup', () => { dragging = false; viewer.style.cursor = 'crosshair'; });

  // ── Click to pick pixel ────────────────────────────────────────────────────
  viewer.addEventListener('click', e => {
    if (!currentFrame || dragging || e.altKey || e.button !== 0) return;
    const rect = viewer.getBoundingClientRect();
    const u = Math.round((e.clientX - rect.left - offset.x) / scale);
    const v = Math.round((e.clientY - rect.top  - offset.y) / scale);
    if (u < 0 || u > naturalW || v < 0 || v > naturalH) return;

    const match = currentFrame.match(/(\d+)/);
    const frameNum = match ? String(parseInt(match[1])) : currentFrame;

    obs[activeMode].push({frame: frameNum, u, v, file: currentFrame});
    renderObsList();
    updateProgress();
    redrawCanvas();
  });

  // ── Canvas overlay ─────────────────────────────────────────────────────────
  function redrawCanvas() {
    if (!currentFrame) return;
    const rect = viewer.getBoundingClientRect();
    canvas.width  = rect.width;
    canvas.height = rect.height;
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Helper: map natural image pixel -> canvas display pixel
    const cx = u => offset.x + u * scale;
    const cy = v => offset.y + v * scale;
    const R = 10;  // crosshair radius in display px

    ['A','B','C'].forEach(pt => {
      const color = COLORS[pt];
      obs[pt].filter(o => o.file === currentFrame).forEach(o => {
        const x = cx(o.u), y = cy(o.v);
        ctx.strokeStyle = color;
        ctx.fillStyle   = color;
        ctx.lineWidth = 2;

        // crosshair
        ctx.beginPath();
        ctx.moveTo(x - R*2, y); ctx.lineTo(x + R*2, y);
        ctx.moveTo(x, y - R*2); ctx.lineTo(x, y + R*2);
        ctx.stroke();

        // circle
        ctx.beginPath();
        ctx.arc(x, y, R, 0, Math.PI*2); ctx.stroke();

        // label
        ctx.fillStyle = color;
        ctx.font = 'bold 13px JetBrains Mono, monospace';
        ctx.fillText(`${pt}·${LABELS[pt]} (${o.u},${o.v})`, x + 14, y - 6);
      });
    });
  }

  // ── Observations list ──────────────────────────────────────────────────────
  function renderObsList() {
    const list = document.getElementById('obsList');
    const total = obs.A.length + obs.B.length + obs.C.length;
    if (total === 0) {
      list.innerHTML = '<div style="padding:16px;color:var(--muted);font-size:0.74rem;text-align:center;">Select a frame and click A, B, C points</div>';
      return;
    }
    let html = '';
    ['A','B','C'].forEach(pt => {
      if (obs[pt].length === 0) return;
      const names = { A:'Point A · VGA', B:'Point B · Ethernet', C:'Point C · Power' };
      html += `<div class="obs-group-title ${pt}">${names[pt]} (${obs[pt].length}×)</div>`;
      obs[pt].forEach((o, i) => {
        html += `
          <div class="obs-item">
            <div>
              <span class="obs-frame ${pt}">f${o.frame}</span>
              <span class="obs-coords" style="margin-left:6px;">(${o.u}, ${o.v})</span>
            </div>
            <button class="del-btn" onclick="deleteObs('${pt}',${i})">✕</button>
          </div>`;
      });
    });
    list.innerHTML = html;
  }

  function deleteObs(pt, i) {
    obs[pt].splice(i, 1);
    renderObsList(); updateProgress(); redrawCanvas();
  }

  function clearAll() {
    obs = { A: [], B: [], C: [] };
    lastOBBs = {};
    renderObsList(); updateProgress(); redrawCanvas();
    const rb = document.getElementById('resultBox');
    rb.classList.remove('visible');
  }

  // ── Progress indicator ─────────────────────────────────────────────────────
  function updateProgress() {
    const hasA = obs.A.length > 0, hasB = obs.B.length > 0, hasC = obs.C.length > 0;
    document.getElementById('dotA').className = 'prog-dot' + (hasA ? ' filled-A' : '');
    document.getElementById('dotB').className = 'prog-dot' + (hasB ? ' filled-B' : '');
    document.getElementById('dotC').className = 'prog-dot' + (hasC ? ' filled-C' : '');

    let msg = '';
    if (!hasA) msg = 'Click Point A (VGA) on ≥2 frames';
    else if (!hasB) msg = 'Now click Point B (Ethernet)';
    else if (!hasC) msg = 'Now click Point C (Power)';
    else msg = `✓ All 3 points set — compute OBB`;
    document.getElementById('progMsg').textContent = msg;
  }

  // ── Compute OBB ───────────────────────────────────────────────────────────
  async function computeOBB() {
    const entity = document.getElementById('entitySelect').value;

    if (obs.A.length < 2) {
      showResult(`⚠ Need ≥2 observations for Point A (VGA).\nHave: ${obs.A.length}`, true);
      return;
    }
    if (obs.B.length < 2) {
      showResult(`⚠ Need ≥2 observations for Point B (Ethernet).\nHave: ${obs.B.length}`, true);
      return;
    }
    if (obs.C.length < 2) {
      showResult(`⚠ Need ≥2 observations for Point C (Power).\nHave: ${obs.C.length}`, true);
      return;
    }

    const payload = {
      entity,
      obs_A: obs.A.map(o => [o.frame, o.u, o.v]),
      obs_B: obs.B.map(o => [o.frame, o.u, o.v]),
      obs_C: obs.C.map(o => [o.frame, o.u, o.v]),
    };

    showResult('Triangulating A, B, C…', false);
    try {
      const res = await fetch('/compute_obb', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(payload)
      });
      const data = await res.json();
      if (data.error) { showResult('Error: ' + data.error, true); return; }
      lastOBBs[entity] = data;
      showResult(formatResult(data), false);
    } catch(e) {
      showResult('Error: ' + e.message, true);
    }
  }

  function formatResult(d) {
    const lines = [
      `entity  : ${d.entity}`,
      `center  : [${d.center.map(x=>x.toFixed(5)).join(', ')}]`,
      `extent  : [${d.extent.map(x=>x.toFixed(5)).join(', ')}]`,
      ``,
      `triangulated points:`,
      `  A(VGA): [${d.pt_A.map(x=>x.toFixed(4)).join(', ')}]`,
      `  B(ETH): [${d.pt_B.map(x=>x.toFixed(4)).join(', ')}]`,
      `  C(PWR): [${d.pt_C.map(x=>x.toFixed(4)).join(', ')}]`,
      ``,
      `reproj errors (center):`,
      ...d.reprojection.map(r => `  f${r.frame}: ${r.error.toFixed(1)}px`),
    ];
    return lines.join('\n');
  }

  function showResult(text, isError) {
    const rb = document.getElementById('resultBox');
    rb.textContent = text;
    rb.style.color = isError ? 'var(--accent2)' : 'var(--accent)';
    rb.classList.add('visible');
  }

  // ── Validate ───────────────────────────────────────────────────────────────
  function openValidateModal() {
    const entity = document.getElementById('entitySelect').value;
    if (!lastOBBs[entity]) { showResult(`⚠ Compute OBB for ${entity} first.`, true); return; }
    if (!currentFrame) { showResult('⚠ Select a frame first.', true); return; }
    const match = currentFrame.match(/(\d+)/);
    const frameNum = match ? String(parseInt(match[1])) : '0';
    const url = `/validate?entity=${entity}&frame=${frameNum}&t=${Date.now()}`;
    document.getElementById('validateImg').src = url;
    document.getElementById('modalOverlay').classList.add('open');
  }
  function closeModal() { document.getElementById('modalOverlay').classList.remove('open'); }

  // ── Save ───────────────────────────────────────────────────────────────────
  async function saveAnswers() {
    const res  = await fetch('/save_answers');
    const data = await res.json();
    showResult(data.message || 'Saved!', false);
  }

  // ── Init ───────────────────────────────────────────────────────────────────
  buildFrameList();
  updateProgress();
  window.addEventListener('resize', () => { if (currentFrame) resetZoom(); });
</script>
</body>
</html>
"""

# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────
@app.route("/")
def index():
    frames = get_frame_list()
    return render_template_string(HTML, frames=frames)


@app.route("/frame/<path:filename>")
def serve_frame(filename):
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        return f"Frame not found: {path}", 404
    return send_file(path, mimetype="image/png")


@app.route("/compute_obb", methods=["POST"])
def compute_obb_route():
    try:
        data = request.get_json()
        entity  = data["entity"]
        raw_A   = data["obs_A"]   # [[frame, u, v], ...]
        raw_B   = data["obs_B"]
        raw_C   = data["obs_C"]

        for label, raw in [("A", raw_A), ("B", raw_B), ("C", raw_C)]:
            if len(raw) < 2:
                return jsonify({"error": f"Point {label} needs ≥2 observations (have {len(raw)})"})

        def to_obs(raw):
            return [(str(r[0]), float(r[1]), float(r[2])) for r in raw]

        poses = load_poses()
        pt_A = triangulate(to_obs(raw_A), poses)
        pt_B = triangulate(to_obs(raw_B), poses)
        pt_C = triangulate(to_obs(raw_C), poses)

        # Rotation derived from geometry of the three panel points
        rotation = derive_rotation_from_points(pt_A, pt_B, pt_C)

        # Entity center = triangulated point for that entity
        entity_center_map = {
            "vga_socket":      pt_A,
            "ethernet_socket": pt_B,
            "power_socket":    pt_C,
        }
        center = entity_center_map.get(entity, pt_A)
        extent = KNOWN_EXTENTS.get(entity, [0.008, 0.0065, 0.0055])

        # Reprojection errors on the entity's center
        reproj = []
        obs_for_entity = {"vga_socket": raw_A, "ethernet_socket": raw_B, "power_socket": raw_C}
        for frame_key, u_obs, v_obs in to_obs(obs_for_entity.get(entity, raw_A)):
            u_p, v_p = project_point(center, frame_key, poses)
            if u_p is not None:
                err = float(np.sqrt((u_p - u_obs)**2 + (v_p - v_obs)**2))
                reproj.append({"frame": frame_key, "error": err})

        obb = {
            "entity":       entity,
            "center":       center.tolist(),
            "extent":       extent,
            "rotation":     rotation.tolist(),
            "pt_A":         pt_A.tolist(),
            "pt_B":         pt_B.tolist(),
            "pt_C":         pt_C.tolist(),
            "reprojection": reproj,
        }

        _save_to_answers(entity, center.tolist(), extent, rotation.tolist())
        return jsonify(obb)

    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "trace": traceback.format_exc()})


@app.route("/validate")
def validate():
    entity    = request.args.get("entity", "vga_socket")
    frame_key = request.args.get("frame", "461")

    answers_path = os.path.join(DATA_DIR, "answers.json")
    if not os.path.exists(answers_path):
        return "answers.json not found", 404

    with open(answers_path) as f:
        answers = json.load(f)

    obb_data = next((a["obb"] for a in answers if a["entity"] == entity), None)
    if not obb_data:
        return f"No OBB found for {entity}", 404

    center   = np.array(obb_data["center"])
    extent   = obb_data["extent"]
    rotation = np.array(obb_data["rotation"])
    corners  = get_obb_corners(center, extent, rotation)

    poses    = load_poses()
    img_path = os.path.join(DATA_DIR, f"frame_{int(frame_key):06d}.png")
    if not os.path.exists(img_path):
        # try without zero-padding
        img_path = os.path.join(DATA_DIR, f"frame_{int(frame_key)}.png")
    pil_img  = Image.open(img_path)
    draw     = ImageDraw.Draw(pil_img)
    W, H     = pil_img.size

    c2w = np.array(poses[str(frame_key)])
    w2c = np.linalg.inv(c2w)
    R   = w2c[:3,:3]; t = w2c[:3,3]
    P   = K @ np.hstack([R, t.reshape(3,1)])

    def proj(pt):
        x = P @ np.append(pt, 1)
        return (int(x[0]/x[2]), int(x[1]/x[2]))

    pts   = [proj(c) for c in corners]
    edges = [(0,1),(1,3),(3,2),(2,0),(4,5),(5,7),(7,6),(6,4),(0,4),(1,5),(2,6),(3,7)]
    for p1, p2 in edges:
        draw.line([pts[p1], pts[p2]], fill=(0,255,100), width=3)

    # VGA reference circle
    u_vga, v_vga = project_point(VGA_CENTER, frame_key, poses)
    if u_vga and 0 < u_vga < W and 0 < v_vga < H:
        draw.ellipse([u_vga-12, v_vga-12, u_vga+12, v_vga+12], outline=(255,60,60), width=3)
        draw.text((int(u_vga)+15, int(v_vga)), "VGA(GT)", fill=(255,60,60))

    # Entity center dot
    u_c, v_c = project_point(center, frame_key, poses)
    if u_c and 0 < u_c < W and 0 < v_c < H:
        draw.ellipse([u_c-10, v_c-10, u_c+10, v_c+10], fill=(0,255,100))
        draw.text((int(u_c)+14, int(v_c)), entity, fill=(0,255,100))

    out = pil_img.resize((1280, 720))
    buf = io.BytesIO()
    out.save(buf, format="JPEG", quality=88)
    buf.seek(0)
    return send_file(buf, mimetype="image/jpeg")


@app.route("/save_answers")
def save_answers_route():
    answers_path = os.path.join(DATA_DIR, "answers.json")
    if os.path.exists(answers_path):
        return jsonify({"message": f"✓ answers.json up to date at {answers_path}"})
    return jsonify({"message": "Nothing to save yet — compute an OBB first."})


def _save_to_answers(entity, center, extent, rotation):
    answers_path = os.path.join(DATA_DIR, "answers.json")
    if os.path.exists(answers_path):
        with open(answers_path) as f:
            answers = json.load(f)
    else:
        answers = [
            {
                "entity": "vga_socket",
                "obb": {
                    "center":   VGA_CENTER.tolist(),
                    "extent":   KNOWN_EXTENTS["vga_socket"],
                    "rotation": VGA_ROTATION_GT.tolist()
                }
            }
        ]

    updated = False
    for entry in answers:
        if entry["entity"] == entity:
            entry["obb"] = {"center": center, "extent": extent, "rotation": rotation}
            updated = True
            break
    if not updated:
        answers.append({"entity": entity, "obb": {"center": center, "extent": extent, "rotation": rotation}})

    with open(answers_path, "w") as f:
        json.dump(answers, f, indent=2)


# ─────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    frames = get_frame_list()
    print(f"\n OBB Calibration Tool — 3-Point Panel Workflow")
    print(f" Data dir : {DATA_DIR}")
    print(f" Frames   : {len(frames)} found")
    print(f" Open     : http://localhost:5000\n")
    app.run(debug=True, port=5000)