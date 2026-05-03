"""
obb_compare.py
--------------
Click centers for VGA, ETH, PWR across frames.
Draws OBBs with NEW rotation (from rotation.json) vs OLD (hardcoded sample answer)
on the selected validation frame. Saves comparison image to OUTPUT_DIR.

Controls:
  Click          → place center for active entity
  MMB / Alt+drag → pan
  Scroll         → zoom
"""

from flask import Flask, render_template_string, request, jsonify, send_file
import json, os, io
import numpy as np
from PIL import Image, ImageDraw, ImageFont

app = Flask(__name__)

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
DATA_DIR   = r"C:\Users\harkh\OneDrive\Desktop\ROBOTIC_PERCEPTION_FINAL_PROJECT\Camera_Properties"
IMAGE_DIR = r"C:\Users\harkh\OneDrive\Desktop\ROBOTIC_PERCEPTION_FINAL_PROJECT\Data"
OUTPUT_DIR = r"C:\Users\harkh\OneDrive\Desktop\ROBOTIC_PERCEPTION_FINAL_PROJECT"

ROTATION_OLD = np.array([
    [-0.004004375172752437,  0.9672545151126772, -0.25377680739897346],
    [ 0.01584254528462312,   0.25380835519540434, 0.9671247761234889],
    [ 0.9998664804554559,   -0.00014774012094266402, -0.016340117333610394]
])

EXTENTS = {
    "vga_socket":      [0.0155,  0.0075,  0.007 ],
    "ethernet_socket": [0.008,   0.0065,  0.0055],
    "power_socket":    [0.014,   0.011,   0.0075],
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

def load_new_rotation():
    path = os.path.join(OUTPUT_DIR, "rotation.json")
    if not os.path.exists(path):
        return None, f"rotation.json not found at {path}"
    with open(path) as f:
        return np.array(json.load(f)), None

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

def get_corners(center, extent, rotation):
    dx, dy, dz = extent
    pts = np.array([[sx*dx, sy*dy, sz*dz]
                    for sx in [-1,1] for sy in [-1,1] for sz in [-1,1]]).T
    return (rotation @ pts + np.array(center).reshape(3,1)).T

def project(world_pt, frame_key, poses):
    c2w = np.array(poses[str(frame_key)])
    w2c = np.linalg.inv(c2w)
    R, t = w2c[:3,:3], w2c[:3,3]
    cam = R @ np.array(world_pt) + t
    if cam[2] <= 0: return None, None
    return float(K[0,0]*cam[0]/cam[2]+K[0,2]), float(K[1,1]*cam[1]/cam[2]+K[1,2])

def draw_obb(draw, corners, poses, frame_key, color, width=3):
    edges = [(0,1),(1,3),(3,2),(2,0),(4,5),(5,7),(7,6),(6,4),(0,4),(1,5),(2,6),(3,7)]
    c2w = np.array(poses[str(frame_key)])
    w2c = np.linalg.inv(c2w)
    R, t = w2c[:3,:3], w2c[:3,3]
    P = K @ np.hstack([R, t.reshape(3,1)])
    def proj(pt):
        x = P @ np.append(pt, 1)
        return (int(x[0]/x[2]), int(x[1]/x[2]))
    pts = [proj(c) for c in corners]
    for p1, p2 in edges:
        draw.line([pts[p1], pts[p2]], fill=color, width=width)

def get_frames():
    try:
        return sorted([f for f in os.listdir(IMAGE_DIR)
                       if f.lower().endswith(".png") and "frame" in f.lower()])
    except:
        return []

# ─────────────────────────────────────────────
# HTML
# ─────────────────────────────────────────────
HTML = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>OBB Compare</title>
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Syne:wght@700;800&display=swap" rel="stylesheet">
<style>
  :root {
    --bg:#0a0a0f; --surf:#12121a; --border:#1e1e2e;
    --acc:#00ff9d; --acc2:#ff3c6e; --acc3:#3c8fff; --acc4:#ffb347;
    --text:#e0e0f0; --muted:#555570;
    --mono:'JetBrains Mono',monospace; --disp:'Syne',sans-serif;
  }
  * { box-sizing:border-box; margin:0; padding:0; }
  body { background:var(--bg); color:var(--text); font-family:var(--mono); height:100vh; display:grid; grid-template-rows:auto 1fr; overflow:hidden; }
  header { padding:12px 26px; border-bottom:1px solid var(--border); background:var(--surf); display:flex; align-items:center; gap:14px; }
  header h1 { font-family:var(--disp); font-size:1.1rem; font-weight:800; color:var(--acc); }
  header span { color:var(--muted); font-size:0.7rem; }
  .layout { display:grid; grid-template-columns:220px 1fr 270px; overflow:hidden; height:100%; }

  .left { background:var(--surf); border-right:1px solid var(--border); display:flex; flex-direction:column; overflow:hidden; }
  .sec-title { padding:9px 13px 6px; font-size:0.6rem; font-weight:700; letter-spacing:2px; text-transform:uppercase; color:var(--muted); border-bottom:1px solid var(--border); }
  .frame-count { padding:4px 13px; font-size:0.63rem; color:var(--muted); border-bottom:1px solid var(--border); }
  .frame-list { overflow-y:auto; flex:1; }
  .frame-item { padding:7px 13px; cursor:pointer; font-size:0.71rem; color:var(--muted); border-left:2px solid transparent; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; transition:all .13s; }
  .frame-item:hover { background:var(--border); color:var(--text); }
  .frame-item.active { background:rgba(0,255,157,.07); color:var(--acc); border-left-color:var(--acc); }

  .ent-bar { padding:10px 11px; border-top:1px solid var(--border); display:flex; flex-direction:column; gap:5px; }
  .lbl { font-size:0.6rem; color:var(--muted); letter-spacing:1.5px; text-transform:uppercase; margin-bottom:2px; }
  .ent-btn { padding:7px 10px; border-radius:4px; border:1px solid var(--border); background:var(--bg); color:var(--muted); font-family:var(--mono); font-size:0.71rem; font-weight:700; cursor:pointer; text-align:left; transition:all .13s; display:flex; justify-content:space-between; }
  .ent-btn:hover { color:var(--text); border-color:var(--text); }
  .ent-btn.active-vga { border-color:var(--acc);  color:var(--acc);  background:rgba(0,255,157,.07); }
  .ent-btn.active-eth { border-color:var(--acc3); color:var(--acc3); background:rgba(60,143,255,.07); }
  .ent-btn.active-pwr { border-color:var(--acc4); color:var(--acc4); background:rgba(255,179,71,.07); }
  .ent-cnt { font-size:0.6rem; opacity:.7; }

  .viewer { position:relative; overflow:hidden; background:#05050a; cursor:crosshair; }
  #main-img { position:absolute; top:0; left:0; max-width:none; display:block; user-select:none; -webkit-user-drag:none; }
  #canvas { position:absolute; top:0; left:0; width:100%; height:100%; pointer-events:none; }
  .hud { position:absolute; top:10px; left:10px; background:rgba(10,10,15,.88); border:1px solid var(--border); border-radius:4px; padding:4px 10px; font-size:0.68rem; color:var(--muted); pointer-events:none; z-index:10; }
  .hud span { color:var(--acc); }
  .ent-hud { position:absolute; top:10px; right:10px; border-radius:4px; padding:5px 13px; font-size:0.72rem; font-weight:700; letter-spacing:1px; pointer-events:none; z-index:10; }
  .ent-hud.vga { background:rgba(0,255,157,.15); border:1px solid var(--acc);  color:var(--acc);  }
  .ent-hud.eth { background:rgba(60,143,255,.15);border:1px solid var(--acc3); color:var(--acc3); }
  .ent-hud.pwr { background:rgba(255,179,71,.15); border:1px solid var(--acc4); color:var(--acc4); }
  .no-img { position:absolute; inset:0; display:flex; align-items:center; justify-content:center; color:var(--muted); font-size:.82rem; pointer-events:none; }
  .zoom-btns { position:absolute; bottom:12px; right:12px; display:flex; gap:4px; z-index:10; }
  .zbtn { background:var(--surf); border:1px solid var(--border); color:var(--text); width:30px; height:30px; border-radius:4px; cursor:pointer; font-size:.95rem; display:flex; align-items:center; justify-content:center; transition:all .13s; }
  .zbtn:hover { border-color:var(--acc); color:var(--acc); }

  .right { background:var(--surf); border-left:1px solid var(--border); display:flex; flex-direction:column; overflow:hidden; }
  .progress { padding:10px 12px; border-bottom:1px solid var(--border); display:flex; flex-direction:column; gap:5px; }
  .prog-row { display:flex; align-items:center; gap:7px; font-size:0.67rem; }
  .prog-dot { width:8px; height:8px; border-radius:50%; border:1.5px solid var(--border); flex-shrink:0; transition:all .2s; }
  .prog-dot.vga { background:var(--acc);  border-color:var(--acc);  }
  .prog-dot.eth { background:var(--acc3); border-color:var(--acc3); }
  .prog-dot.pwr { background:var(--acc4); border-color:var(--acc4); }
  .prog-lbl { color:var(--muted); }
  .prog-lbl.vga { color:var(--acc);  }
  .prog-lbl.eth { color:var(--acc3); }
  .prog-lbl.pwr { color:var(--acc4); }
  .prog-n { margin-left:auto; color:var(--muted); font-size:0.6rem; }

  .obs-list { flex:1; overflow-y:auto; padding:7px; display:flex; flex-direction:column; gap:3px; }
  .obs-group-hdr { font-size:0.58rem; letter-spacing:1.5px; text-transform:uppercase; padding:7px 4px 3px; border-bottom:1px solid var(--border); margin-bottom:2px; }
  .obs-group-hdr.vga { color:var(--acc);  }
  .obs-group-hdr.eth { color:var(--acc3); }
  .obs-group-hdr.pwr { color:var(--acc4); }
  .obs-item { background:var(--bg); border:1px solid var(--border); border-radius:4px; padding:5px 9px; font-size:0.67rem; display:flex; justify-content:space-between; align-items:center; }
  .obs-frame { font-weight:700; }
  .obs-frame.vga { color:var(--acc);  }
  .obs-frame.eth { color:var(--acc3); }
  .obs-frame.pwr { color:var(--acc4); }
  .obs-uv { color:var(--muted); font-size:0.62rem; margin-left:5px; }
  .del { background:none; border:none; color:var(--acc2); cursor:pointer; font-size:.75rem; opacity:.4; transition:opacity .13s; padding:0 3px; }
  .del:hover { opacity:1; }

  /* legend */
  .legend { padding:8px 12px; border-bottom:1px solid var(--border); display:flex; gap:14px; font-size:0.63rem; }
  .leg-item { display:flex; align-items:center; gap:5px; }
  .leg-dot { width:10px; height:4px; border-radius:2px; }

  .result { margin:0 9px 7px; background:var(--bg); border:1px solid var(--border); border-radius:4px; padding:8px; font-size:0.63rem; max-height:100px; overflow-y:auto; white-space:pre-wrap; display:none; }
  .result.show { display:block; }
  .actions { padding:9px 10px; border-top:1px solid var(--border); display:flex; flex-direction:column; gap:6px; }
  .btn { padding:8px 11px; border-radius:4px; border:none; cursor:pointer; font-family:var(--mono); font-size:0.73rem; font-weight:700; transition:all .13s; width:100%; }
  .btn-go { background:var(--acc); color:var(--bg); }
  .btn-go:hover { filter:brightness(1.12); }
  .btn-go:disabled { opacity:.4; cursor:not-allowed; filter:none; }
  .btn-clr { background:transparent; border:1px solid var(--border); color:var(--muted); }
  .btn-clr:hover { border-color:var(--acc2); color:var(--acc2); }

  /* modal */
  .overlay { display:none; position:fixed; inset:0; background:rgba(0,0,0,.85); z-index:100; align-items:center; justify-content:center; }
  .overlay.open { display:flex; }
  .modal { background:var(--surf); border:1px solid var(--border); border-radius:8px; padding:16px; max-width:96vw; max-height:96vh; overflow:auto; display:flex; flex-direction:column; gap:10px; }
  .modal h2 { font-family:var(--disp); font-size:.9rem; color:var(--acc); }
  .modal img { max-width:100%; border-radius:4px; border:1px solid var(--border); }
  .modal-close { align-self:flex-end; background:var(--bg); border:1px solid var(--border); color:var(--text); padding:5px 13px; border-radius:4px; cursor:pointer; font-family:var(--mono); font-size:.73rem; }
  .modal-close:hover { border-color:var(--acc2); color:var(--acc2); }
  ::-webkit-scrollbar { width:4px; }
  ::-webkit-scrollbar-thumb { background:var(--border); border-radius:2px; }
</style>
</head>
<body>

<header>
  <h1>◈ OBB Compare — New vs Old Rotation</h1>
  <span>Click socket centers · ≥2 obs each · then pick a frame to compare · MMB/Alt+drag to pan</span>
</header>

<div class="layout">

  <div class="left">
    <div class="sec-title">Frames</div>
    <div class="frame-count" id="fc">—</div>
    <div class="frame-list" id="frameList"></div>
    <div class="ent-bar">
      <div class="lbl">Active Entity</div>
      <button class="ent-btn active-vga" id="btn-vga" onclick="setEnt('vga_socket')">
        <span>VGA socket</span><span class="ent-cnt" id="cnt-vga">0</span>
      </button>
      <button class="ent-btn" id="btn-eth" onclick="setEnt('ethernet_socket')">
        <span>Ethernet</span><span class="ent-cnt" id="cnt-eth">0</span>
      </button>
      <button class="ent-btn" id="btn-pwr" onclick="setEnt('power_socket')">
        <span>Power</span><span class="ent-cnt" id="cnt-pwr">0</span>
      </button>
    </div>
  </div>

  <div class="viewer" id="viewer">
    <div class="no-img" id="noImg">← Select a frame to begin</div>
    <img id="main-img" src="" alt="" style="display:none;">
    <canvas id="canvas"></canvas>
    <div class="hud">cursor: <span id="hx">—</span>, <span id="hy">—</span> &nbsp;|&nbsp; zoom: <span id="hz">1.0</span>x</div>
    <div class="ent-hud vga" id="entHud">VGA socket</div>
    <div class="zoom-btns">
      <button class="zbtn" onclick="doZoom(1.25)">+</button>
      <button class="zbtn" onclick="doZoom(0.8)">−</button>
      <button class="zbtn" onclick="resetZoom()">⊙</button>
    </div>
  </div>

  <div class="right">
    <div class="sec-title">Observations</div>
    <div class="progress">
      <div class="prog-row"><div class="prog-dot" id="dot-vga"></div><span class="prog-lbl" id="pl-vga">vga_socket</span><span class="prog-n" id="pn-vga">0/≥2</span></div>
      <div class="prog-row"><div class="prog-dot" id="dot-eth"></div><span class="prog-lbl" id="pl-eth">ethernet_socket</span><span class="prog-n" id="pn-eth">0/≥2</span></div>
      <div class="prog-row"><div class="prog-dot" id="dot-pwr"></div><span class="prog-lbl" id="pl-pwr">power_socket</span><span class="prog-n" id="pn-pwr">0/≥2</span></div>
    </div>

    <!-- legend -->
    <div class="legend">
      <div class="leg-item"><div class="leg-dot" style="background:#00ff9d"></div><span style="color:#00ff9d;font-size:.65rem;">NEW rotation</span></div>
      <div class="leg-item"><div class="leg-dot" style="background:#ff3c6e"></div><span style="color:#ff3c6e;font-size:.65rem;">OLD rotation</span></div>
    </div>

    <div class="obs-list" id="obsList">
      <div style="padding:13px;color:var(--muted);font-size:.7rem;text-align:center;">Click socket centers on ≥2 frames each</div>
    </div>
    <div class="result" id="result"></div>
    <div class="actions">
      <select id="valFrame" style="background:var(--bg);color:var(--text);border:1px solid var(--border);border-radius:4px;padding:6px 8px;font-family:var(--mono);font-size:.72rem;width:100%;outline:none;margin-bottom:2px;"></select>
      <button class="btn btn-go" id="compareBtn" onclick="compare()" disabled>▶ Compare OBBs on Frame</button>
      <button class="btn btn-clr" onclick="clearAll()">✕ Clear All</button>
    </div>
  </div>

</div>

<div class="overlay" id="overlay">
  <div class="modal">
    <h2>OBB Comparison — <span style="color:#00ff9d">NEW</span> vs <span style="color:#ff3c6e">OLD</span></h2>
    <img id="valImg" src="" alt="">
    <button class="modal-close" onclick="closeModal()">Close</button>
  </div>
</div>

<script>
  let frames = {{ frames|tojson }};
  let obs = { vga_socket:[], ethernet_socket:[], power_socket:[] };
  let activeEnt = 'vga_socket';
  let curFrame = null;
  let scale = 1, offset = {x:0,y:0};
  let natW = 2560, natH = 1440;
  let panning = false, panStart = {x:0,y:0};

  const viewer = document.getElementById('viewer');
  const img    = document.getElementById('main-img');
  const canvas = document.getElementById('canvas');
  const ctx    = canvas.getContext('2d');

  const ENT_COLOR = { vga_socket:'#00ff9d', ethernet_socket:'#3c8fff', power_socket:'#ffb347' };
  const ENT_CLS   = { vga_socket:'vga',     ethernet_socket:'eth',     power_socket:'pwr'     };
  const ENT_SHORT = { vga_socket:'VGA',     ethernet_socket:'ETH',     power_socket:'PWR'     };
  const ENT_HUD   = { vga_socket:'VGA socket', ethernet_socket:'Ethernet', power_socket:'Power' };

  function buildFrames() {
    const list = document.getElementById('frameList');
    const vf   = document.getElementById('valFrame');
    const fc   = document.getElementById('fc');
    list.innerHTML = ''; vf.innerHTML = '';
    if (!frames.length) { fc.textContent = '⚠ No frames found'; return; }
    fc.textContent = frames.length + ' frames';
    frames.forEach(f => {
      const d = document.createElement('div');
      d.className = 'frame-item'; d.textContent = f; d.id = 'fi_'+f;
      d.onclick = () => loadFrame(f); list.appendChild(d);
      const o = document.createElement('option');
      o.value = f; o.textContent = f; vf.appendChild(o);
    });
  }

  function loadFrame(f) {
    curFrame = f;
    document.querySelectorAll('.frame-item').forEach(e => e.classList.remove('active'));
    const el = document.getElementById('fi_'+f);
    if (el) { el.classList.add('active'); el.scrollIntoView({block:'nearest'}); }
    img.src = '/frame/'+f; img.style.display = 'block';
    document.getElementById('noImg').style.display = 'none';
    img.onload = () => { natW = img.naturalWidth; natH = img.naturalHeight; resetZoom(); };
  }

  function setEnt(ent) {
    activeEnt = ent;
    Object.keys(ENT_CLS).forEach(e => {
      document.getElementById('btn-'+ENT_CLS[e]).className =
        'ent-btn' + (e===ent ? ' active-'+ENT_CLS[e] : '');
    });
    const hud = document.getElementById('entHud');
    hud.textContent = ENT_HUD[ent]; hud.className = 'ent-hud '+ENT_CLS[ent];
    redraw();
  }

  function applyT() {
    img.style.left = offset.x+'px'; img.style.top = offset.y+'px';
    img.style.width = natW*scale+'px'; img.style.height = natH*scale+'px';
    document.getElementById('hz').textContent = scale.toFixed(2);
    redraw();
  }
  function doZoom(f, cx, cy) {
    const r = viewer.getBoundingClientRect();
    cx = cx??r.width/2; cy = cy??r.height/2;
    const prev = scale;
    scale = Math.min(Math.max(scale*f, 0.05), 20);
    offset.x = cx-(cx-offset.x)*(scale/prev);
    offset.y = cy-(cy-offset.y)*(scale/prev);
    applyT();
  }
  function resetZoom() {
    const r = viewer.getBoundingClientRect();
    if (!r.width) { requestAnimationFrame(resetZoom); return; }
    scale = Math.min(r.width/natW, r.height/natH)*0.97;
    offset.x = (r.width -natW*scale)/2;
    offset.y = (r.height-natH*scale)/2;
    applyT();
  }
  viewer.addEventListener('wheel', e => {
    e.preventDefault();
    const r = viewer.getBoundingClientRect();
    doZoom(e.deltaY<0?1.12:0.89, e.clientX-r.left, e.clientY-r.top);
  }, {passive:false});
  viewer.addEventListener('auxclick', e => { if(e.button===1) e.preventDefault(); });
  viewer.addEventListener('mousedown', e => {
    if (e.button===1 || (e.button===0 && e.altKey)) {
      e.preventDefault(); panning = true;
      panStart = {x:e.clientX-offset.x, y:e.clientY-offset.y};
      viewer.style.cursor = 'grabbing';
    }
  });
  window.addEventListener('mousemove', e => {
    if (curFrame) {
      const r = viewer.getBoundingClientRect();
      const u=(e.clientX-r.left-offset.x)/scale, v=(e.clientY-r.top-offset.y)/scale;
      if (u>=0&&u<=natW&&v>=0&&v<=natH) {
        document.getElementById('hx').textContent = Math.round(u);
        document.getElementById('hy').textContent = Math.round(v);
      }
    }
    if (panning) { offset.x=e.clientX-panStart.x; offset.y=e.clientY-panStart.y; applyT(); }
  });
  window.addEventListener('mouseup', () => { if(panning){panning=false;viewer.style.cursor='crosshair';} });

  viewer.addEventListener('click', e => {
    if (!curFrame||e.altKey||e.button!==0||panning) return;
    const r = viewer.getBoundingClientRect();
    const u=Math.round((e.clientX-r.left-offset.x)/scale);
    const v=Math.round((e.clientY-r.top -offset.y)/scale);
    if (u<0||u>natW||v<0||v>natH) return;
    const m = curFrame.match(/(\d+)/);
    const fn = m ? String(parseInt(m[1])) : curFrame;
    obs[activeEnt].push({frame:fn, u, v, file:curFrame});
    updateProgress(); renderObs(); redraw();
  });

  function redraw() {
    const r = viewer.getBoundingClientRect();
    canvas.width = r.width; canvas.height = r.height;
    ctx.clearRect(0,0,canvas.width,canvas.height);
    if (!curFrame) return;
    const sx = u => offset.x+u*scale;
    const sy = v => offset.y+v*scale;
    const R=6, ARM=14;
    Object.entries(obs).forEach(([ent,list]) => {
      const color = ENT_COLOR[ent];
      list.filter(o=>o.file===curFrame).forEach(o => {
        const x=sx(o.u), y=sy(o.v);
        ctx.strokeStyle=color; ctx.fillStyle=color; ctx.lineWidth=1.5;
        ctx.beginPath();
        ctx.moveTo(x-ARM,y); ctx.lineTo(x-R,y);
        ctx.moveTo(x+R,  y); ctx.lineTo(x+ARM,y);
        ctx.moveTo(x,y-ARM); ctx.lineTo(x,y-R);
        ctx.moveTo(x,y+R);   ctx.lineTo(x,y+ARM);
        ctx.stroke();
        ctx.beginPath(); ctx.arc(x,y,R,0,Math.PI*2); ctx.stroke();
        ctx.font='bold 12px JetBrains Mono,monospace';
        ctx.fillText(ENT_SHORT[ent]+' f'+o.frame, x+R+5, y-5);
      });
    });
  }

  function renderObs() {
    const list = document.getElementById('obsList');
    const total = Object.values(obs).reduce((s,a)=>s+a.length,0);
    if (!total) { list.innerHTML='<div style="padding:13px;color:var(--muted);font-size:.7rem;text-align:center;">Click socket centers on ≥2 frames each</div>'; return; }
    let html='';
    [['vga_socket','vga'],['ethernet_socket','eth'],['power_socket','pwr']].forEach(([ent,cls])=>{
      if (!obs[ent].length) return;
      html+='<div class="obs-group-hdr '+cls+'">'+ent+' ('+obs[ent].length+'x)</div>';
      obs[ent].forEach((o,i)=>{
        html+='<div class="obs-item"><div><span class="obs-frame '+cls+'">f'+o.frame+'</span><span class="obs-uv">('+o.u+','+o.v+')</span></div><button class="del" onclick="delObs(\''+ent+'\','+i+')">✕</button></div>';
      });
    });
    list.innerHTML=html;
  }

  function delObs(ent,i) { obs[ent].splice(i,1); updateProgress(); renderObs(); redraw(); }
  function clearAll() {
    obs={vga_socket:[],ethernet_socket:[],power_socket:[]};
    updateProgress(); renderObs(); redraw();
    document.getElementById('result').classList.remove('show');
  }

  function updateProgress() {
    [['vga_socket','vga'],['ethernet_socket','eth'],['power_socket','pwr']].forEach(([ent,cls])=>{
      const n=obs[ent].length, ready=n>=2;
      document.getElementById('dot-'+cls).className='prog-dot'+(ready?' '+cls:'');
      document.getElementById('pl-' +cls).className='prog-lbl'+(ready?' '+cls:'');
      document.getElementById('pn-' +cls).textContent=n+'/≥2';
      document.getElementById('cnt-'+cls).textContent=n;
    });
    const allReady=['vga_socket','ethernet_socket','power_socket'].every(e=>obs[e].length>=2);
    document.getElementById('compareBtn').disabled=!allReady;
  }

  async function compare() {
    const frame = document.getElementById('valFrame').value;
    showResult('Triangulating + rendering…', false);
    try {
      const res = await fetch('/compare', {
        method:'POST', headers:{'Content-Type':'application/json'},
        body: JSON.stringify({
          frame,
          vga: obs.vga_socket.map(o=>[o.frame,o.u,o.v]),
          eth: obs.ethernet_socket.map(o=>[o.frame,o.u,o.v]),
          pwr: obs.power_socket.map(o=>[o.frame,o.u,o.v]),
        })
      });
      const data = await res.json();
      if (data.error) { showResult('Error: '+data.error, true); return; }
      document.getElementById('valImg').src = '/result_image?t='+Date.now();
      document.getElementById('overlay').classList.add('open');
      showResult('✓ Saved comparison image to OUTPUT_DIR', false);
    } catch(e) { showResult('Error: '+e.message, true); }
  }

  function showResult(txt,err) {
    const el=document.getElementById('result');
    el.textContent=txt; el.style.color=err?'var(--acc2)':'var(--acc)';
    el.classList.add('show');
  }
  function closeModal() { document.getElementById('overlay').classList.remove('open'); }

  buildFrames();
  updateProgress();
  window.addEventListener('resize',()=>{ if(curFrame) resetZoom(); });
</script>
</body>
</html>
"""

# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────
last_image_buf = None

@app.route("/")
def index():
    return render_template_string(HTML, frames=get_frames())

@app.route("/frame/<path:filename>")
def serve_frame(filename):
    path = os.path.join(IMAGE_DIR, filename)
    if not os.path.exists(path): return f"Not found: {path}", 404
    return send_file(path, mimetype="image/png")

@app.route("/compare", methods=["POST"])
def compare():
    global last_image_buf
    try:
        data     = request.get_json()
        frame_raw= data["frame"]
        raw_vga  = data["vga"]
        raw_eth  = data["eth"]
        raw_pwr  = data["pwr"]

        for label, raw in [("VGA",raw_vga),("ETH",raw_eth),("PWR",raw_pwr)]:
            if len(raw) < 2:
                return jsonify({"error": f"{label} needs ≥2 obs"})

        rot_new, err = load_new_rotation()
        if err: return jsonify({"error": err})

        poses = load_poses()
        to_obs = lambda r: [(str(x[0]),float(x[1]),float(x[2])) for x in r]

        p_vga = triangulate(to_obs(raw_vga), poses)
        p_eth = triangulate(to_obs(raw_eth), poses)
        p_pwr = triangulate(to_obs(raw_pwr), poses)

        center_map = {"vga_socket":p_vga, "ethernet_socket":p_eth, "power_socket":p_pwr}

        # parse frame number
        import re
        m = re.search(r'\d+', frame_raw)
        fn = int(m.group()) if m else 0
        img_path = os.path.join(IMAGE_DIR, f"frame_{fn:06d}.png")
        if not os.path.exists(img_path):
            img_path = os.path.join(IMAGE_DIR, f"frame_{fn}.png")

        pil  = Image.open(img_path).copy()
        draw = ImageDraw.Draw(pil)

        # draw both rotations per entity
        for ent, extent in EXTENTS.items():
            center = center_map[ent]
            # NEW — green
            corners_new = get_corners(center, extent, rot_new)
            draw_obb(draw, corners_new, poses, str(fn), color=(0,255,157), width=3)
            # OLD — red
            corners_old = get_corners(center, extent, ROTATION_OLD)
            draw_obb(draw, corners_old, poses, str(fn), color=(255,60,110), width=3)
            # center dot
            uc, vc = project(center, str(fn), poses)
            if uc and 0 < uc < pil.width and 0 < vc < pil.height:
                draw.ellipse([uc-6,vc-6,uc+6,vc+6], fill=(255,255,255))
                draw.text((int(uc)+9,int(vc)-6), ent.split('_')[0], fill=(255,255,255))

        # legend
        draw.rectangle([20,20,220,80], fill=(10,10,15,200))
        draw.rectangle([30,30,60,46], fill=(0,255,157))
        draw.text((68,32), "NEW rotation", fill=(0,255,157))
        draw.rectangle([30,54,60,70], fill=(255,60,110))
        draw.text((68,56), "OLD rotation", fill=(255,60,110))

        # save to OUTPUT_DIR
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        out_path = os.path.join(OUTPUT_DIR, f"obb_compare_frame{fn}.jpg")
        pil.save(out_path, quality=92)

        # also keep in memory for modal display
        out_small = pil.resize((1280, 720))
        buf = io.BytesIO()
        out_small.save(buf, format="JPEG", quality=90)
        buf.seek(0)
        last_image_buf = buf.getvalue()

        return jsonify({"ok": True})

    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "trace": traceback.format_exc()})

@app.route("/result_image")
def result_image():
    global last_image_buf
    if last_image_buf is None:
        return "No image yet", 404
    return send_file(io.BytesIO(last_image_buf), mimetype="image/jpeg")

# ─────────────────────────────────────────────
if __name__ == "__main__":
    frames = get_frames()
    rot_new, err = load_new_rotation()
    print(f"\n  OBB Compare Tool")
    print(f"  Data dir   : {DATA_DIR}")
    print(f"  Output dir : {OUTPUT_DIR}")
    print(f"  Frames     : {len(frames)} found")
    print(f"  rotation.json : {'OK loaded' if rot_new is not None else 'NOT FOUND - ' + str(err)}")
    print(f"  Open       : http://localhost:5000\n")
    app.run(debug=True, port=5000)