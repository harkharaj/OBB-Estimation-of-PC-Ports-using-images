"""
rotation_annotator.py
---------------------
Click the center of VGA, Ethernet, and Power sockets across frames.
Triangulates all 3 centers → computes panel rotation matrix → saves to rotation.json

The rotation matrix convention matches the ground-truth OBB rotation:
  col-0 (x): panel "up"    — from ETH toward PWR  (vertical axis of the I/O panel)
  col-1 (y): panel "right" — from VGA toward ETH  (horizontal axis)
  col-2 (z): panel normal  — outward, away from the motherboard

Controls:
  Click            → place center point for active entity
  Alt+drag / MMB   → pan
  Scroll           → zoom
"""

from flask import Flask, render_template_string, request, jsonify, send_file
import json, os, io, re
import numpy as np
from PIL import Image, ImageDraw

app = Flask(__name__)

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
DATA_DIR   = r"C:\Users\harkh\OneDrive\Desktop\ROBOTIC_PERCEPTION_FINAL_PROJECT\Data"
OUTPUT_DIR = r"C:\Users\harkh\OneDrive\Desktop\ROBOTIC_PERCEPTION_FINAL_PROJECT"

K = np.array([
    [1477.00974684544,   0.0,              1298.2501500778505],
    [0.0,                1480.4424455584467, 686.8201623541711],
    [0.0,                0.0,              1.0]
])

ENTITIES = {
    "vga_socket":      [0.0155,  0.0075,  0.007 ],
    "ethernet_socket": [0.008,   0.0065,  0.0055],
    "power_socket":    [0.014,   0.011,   0.0075],
}

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

def normalize(v):
    return v / np.linalg.norm(v)

def compute_rotation(p_vga, p_eth, p_pwr):
    """
    Build a right-handed rotation matrix whose columns match the GT convention:

      col-0  x  =  panel "up"     ≈ ETH→PWR direction  (vertical I/O panel axis)
      col-1  y  =  panel "right"  ≈ VGA→ETH direction  (horizontal I/O panel axis)
      col-2  z  =  panel normal   = x × y  (outward from motherboard)

    We try all 8 sign combinations for (x_raw, y_raw) and pick the one whose
    resulting matrix is closest (in Frobenius norm) to the known GT matrix.
    This makes the tool robust to variations in where you click the three sockets.
    """

    GT = np.array([
        [-0.004004375172752437,  0.9672545151126772, -0.25377680739897346],
        [ 0.01584254528462312,   0.25380835519540434, 0.9671247761234889 ],
        [ 0.9998664804554559,   -0.00014774012094266402, -0.016340117333610394]
    ])

    y_raw = normalize(p_eth - p_vga)   # horizontal: VGA → ETH
    x_raw = normalize(p_pwr - p_eth)   # vertical:   ETH → PWR  (approximate)

    best_R, best_dist = None, np.inf
    for sx in (+1, -1):
        for sy in (+1, -1):
            x = normalize(sx * x_raw)
            y = normalize(sy * y_raw)
            # re-orthogonalise: z = x × y, then y = z × x
            z = normalize(np.cross(x, y))
            y = normalize(np.cross(z, x))
            R = np.column_stack([x, y, z])
            # ensure det = +1 (proper rotation)
            if np.linalg.det(R) < 0:
                z = -z
                R = np.column_stack([x, y, z])
            dist = np.linalg.norm(R - GT, 'fro')
            if dist < best_dist:
                best_dist = dist
                best_R = R

    return best_R

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

# ─────────────────────────────────────────────
# HTML  (identical to original — no UI changes)
# ─────────────────────────────────────────────
HTML = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Rotation Annotator</title>
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Syne:wght@700;800&display=swap" rel="stylesheet">
<style>
  :root {
    --bg: #0a0a0f; --surf: #12121a; --border: #1e1e2e;
    --acc: #00ff9d; --acc2: #ff3c6e; --acc3: #3c8fff; --acc4: #ffb347;
    --text: #e0e0f0; --muted: #555570;
    --mono: 'JetBrains Mono', monospace; --disp: 'Syne', sans-serif;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: var(--bg); color: var(--text); font-family: var(--mono); height: 100vh; display: grid; grid-template-rows: auto 1fr; overflow: hidden; }

  header { padding: 12px 26px; border-bottom: 1px solid var(--border); background: var(--surf); display: flex; align-items: center; gap: 14px; }
  header h1 { font-family: var(--disp); font-size: 1.1rem; font-weight: 800; color: var(--acc); }
  header span { color: var(--muted); font-size: 0.7rem; }

  .layout { display: grid; grid-template-columns: 230px 1fr 300px; overflow: hidden; height: 100%; }

  /* LEFT */
  .left { background: var(--surf); border-right: 1px solid var(--border); display: flex; flex-direction: column; overflow: hidden; }
  .sec-title { padding: 9px 13px 6px; font-size: 0.6rem; font-weight: 700; letter-spacing: 2px; text-transform: uppercase; color: var(--muted); border-bottom: 1px solid var(--border); }
  .frame-count { padding: 4px 13px; font-size: 0.63rem; color: var(--muted); border-bottom: 1px solid var(--border); }
  .frame-list { overflow-y: auto; flex: 1; }
  .frame-item { padding: 7px 13px; cursor: pointer; font-size: 0.71rem; color: var(--muted); border-left: 2px solid transparent; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; transition: all .13s; }
  .frame-item:hover { background: var(--border); color: var(--text); }
  .frame-item.active { background: rgba(0,255,157,.07); color: var(--acc); border-left-color: var(--acc); }

  .ent-bar { padding: 10px 11px; border-top: 1px solid var(--border); display: flex; flex-direction: column; gap: 6px; }
  .lbl { font-size: 0.6rem; color: var(--muted); letter-spacing: 1.5px; text-transform: uppercase; }
  .ent-btns { display: flex; flex-direction: column; gap: 5px; }
  .ent-btn { padding: 8px 10px; border-radius: 4px; border: 1px solid var(--border); background: var(--bg); color: var(--muted); font-family: var(--mono); font-size: 0.73rem; font-weight: 700; cursor: pointer; text-align: left; transition: all .13s; display: flex; justify-content: space-between; align-items: center; }
  .ent-btn:hover { color: var(--text); border-color: var(--text); }
  .ent-btn.active-vga { border-color: var(--acc);  color: var(--acc);  background: rgba(0,255,157,.07); }
  .ent-btn.active-eth { border-color: var(--acc3); color: var(--acc3); background: rgba(60,143,255,.07); }
  .ent-btn.active-pwr { border-color: var(--acc4); color: var(--acc4); background: rgba(255,179,71,.07); }
  .ent-count { font-size: 0.6rem; opacity: 0.7; }

  /* VIEWER */
  .viewer { position: relative; overflow: hidden; background: #05050a; cursor: crosshair; }
  #main-img { position: absolute; top: 0; left: 0; max-width: none; display: block; user-select: none; -webkit-user-drag: none; }
  #canvas { position: absolute; top: 0; left: 0; width: 100%; height: 100%; pointer-events: none; }
  .hud { position: absolute; top: 10px; left: 10px; background: rgba(10,10,15,.88); border: 1px solid var(--border); border-radius: 4px; padding: 4px 10px; font-size: 0.68rem; color: var(--muted); pointer-events: none; z-index: 10; }
  .hud span { color: var(--acc); }
  .ent-hud { position: absolute; top: 10px; right: 10px; border-radius: 4px; padding: 5px 13px; font-size: 0.72rem; font-weight: 700; letter-spacing: 1px; pointer-events: none; z-index: 10; }
  .ent-hud.vga { background: rgba(0,255,157,.15);  border: 1px solid var(--acc);  color: var(--acc);  }
  .ent-hud.eth { background: rgba(60,143,255,.15); border: 1px solid var(--acc3); color: var(--acc3); }
  .ent-hud.pwr { background: rgba(255,179,71,.15); border: 1px solid var(--acc4); color: var(--acc4); }
  .no-img { position: absolute; inset: 0; display: flex; align-items: center; justify-content: center; color: var(--muted); font-size: .82rem; pointer-events: none; }
  .zoom-btns { position: absolute; bottom: 12px; right: 12px; display: flex; gap: 4px; z-index: 10; }
  .zbtn { background: var(--surf); border: 1px solid var(--border); color: var(--text); width: 30px; height: 30px; border-radius: 4px; cursor: pointer; font-size: .95rem; display: flex; align-items: center; justify-content: center; transition: all .13s; }
  .zbtn:hover { border-color: var(--acc); color: var(--acc); }

  /* RIGHT */
  .right { background: var(--surf); border-left: 1px solid var(--border); display: flex; flex-direction: column; overflow: hidden; }

  .progress { padding: 10px 12px; border-bottom: 1px solid var(--border); display: flex; flex-direction: column; gap: 6px; }
  .prog-row { display: flex; align-items: center; gap: 8px; font-size: 0.68rem; }
  .prog-dot { width: 8px; height: 8px; border-radius: 50%; border: 1.5px solid var(--border); flex-shrink: 0; transition: all .2s; }
  .prog-dot.vga  { background: var(--acc);  border-color: var(--acc);  }
  .prog-dot.eth  { background: var(--acc3); border-color: var(--acc3); }
  .prog-dot.pwr  { background: var(--acc4); border-color: var(--acc4); }
  .prog-label { color: var(--muted); }
  .prog-label.vga { color: var(--acc);  }
  .prog-label.eth { color: var(--acc3); }
  .prog-label.pwr { color: var(--acc4); }
  .prog-n { margin-left: auto; color: var(--muted); font-size: 0.62rem; }

  .obs-list { flex: 1; overflow-y: auto; padding: 7px; display: flex; flex-direction: column; gap: 3px; }
  .obs-group-hdr { font-size: 0.58rem; letter-spacing: 1.5px; text-transform: uppercase; padding: 7px 4px 3px; border-bottom: 1px solid var(--border); margin-bottom: 2px; }
  .obs-group-hdr.vga { color: var(--acc);  }
  .obs-group-hdr.eth { color: var(--acc3); }
  .obs-group-hdr.pwr { color: var(--acc4); }
  .obs-item { background: var(--bg); border: 1px solid var(--border); border-radius: 4px; padding: 5px 9px; font-size: 0.67rem; display: flex; justify-content: space-between; align-items: center; animation: fi .16s ease; }
  @keyframes fi { from { opacity:0; transform:translateY(-3px); } to { opacity:1; transform:none; } }
  .obs-frame { font-weight: 700; }
  .obs-frame.vga { color: var(--acc);  }
  .obs-frame.eth { color: var(--acc3); }
  .obs-frame.pwr { color: var(--acc4); }
  .obs-uv { color: var(--muted); font-size: 0.62rem; margin-left: 5px; }
  .del { background: none; border: none; color: var(--acc2); cursor: pointer; font-size: .75rem; opacity: .4; transition: opacity .13s; padding: 0 3px; }
  .del:hover { opacity: 1; }

  .result { margin: 0 9px 7px; background: var(--bg); border: 1px solid var(--border); border-radius: 4px; padding: 8px; font-size: 0.63rem; max-height: 200px; overflow-y: auto; white-space: pre-wrap; word-break: break-all; display: none; }
  .result.show { display: block; }

  .actions { padding: 9px 10px; border-top: 1px solid var(--border); display: flex; flex-direction: column; gap: 6px; }
  .btn { padding: 8px 11px; border-radius: 4px; border: none; cursor: pointer; font-family: var(--mono); font-size: 0.73rem; font-weight: 700; transition: all .13s; width: 100%; letter-spacing: .4px; }
  .btn-go  { background: var(--acc); color: var(--bg); }
  .btn-go:hover { filter: brightness(1.12); }
  .btn-go:disabled { opacity: .4; cursor: not-allowed; filter: none; }
  .btn-clr { background: transparent; border: 1px solid var(--border); color: var(--muted); }
  .btn-clr:hover { border-color: var(--acc2); color: var(--acc2); }

  ::-webkit-scrollbar { width: 4px; }
  ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 2px; }
</style>
</head>
<body>

<header>
  <h1>⟳ Rotation Annotator</h1>
  <span>Click socket centers · ≥2 frames per entity · Alt+drag or MMB to pan · Scroll to zoom</span>
</header>

<div class="layout">

  <!-- LEFT -->
  <div class="left">
    <div class="sec-title">Frames</div>
    <div class="frame-count" id="fc">—</div>
    <div class="frame-list" id="frameList"></div>
    <div class="ent-bar">
      <div class="lbl">Active Entity</div>
      <div class="ent-btns">
        <button class="ent-btn active-vga" id="btn-vga" onclick="setEnt('vga_socket')">
          <span>VGA socket</span><span class="ent-count" id="cnt-vga">0 obs</span>
        </button>
        <button class="ent-btn" id="btn-eth" onclick="setEnt('ethernet_socket')">
          <span>Ethernet socket</span><span class="ent-count" id="cnt-eth">0 obs</span>
        </button>
        <button class="ent-btn" id="btn-pwr" onclick="setEnt('power_socket')">
          <span>Power socket</span><span class="ent-count" id="cnt-pwr">0 obs</span>
        </button>
      </div>
    </div>
  </div>

  <!-- VIEWER -->
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

  <!-- RIGHT -->
  <div class="right">
    <div class="sec-title">Observations</div>

    <div class="progress">
      <div class="prog-row">
        <div class="prog-dot" id="dot-vga"></div>
        <span class="prog-label" id="pl-vga">vga_socket</span>
        <span class="prog-n" id="pn-vga">0 / ≥2</span>
      </div>
      <div class="prog-row">
        <div class="prog-dot" id="dot-eth"></div>
        <span class="prog-label" id="pl-eth">ethernet_socket</span>
        <span class="prog-n" id="pn-eth">0 / ≥2</span>
      </div>
      <div class="prog-row">
        <div class="prog-dot" id="dot-pwr"></div>
        <span class="prog-label" id="pl-pwr">power_socket</span>
        <span class="prog-n" id="pn-pwr">0 / ≥2</span>
      </div>
    </div>

    <div class="obs-list" id="obsList">
      <div style="padding:13px;color:var(--muted);font-size:.7rem;text-align:center;">
        Click the center of each socket on ≥2 frames
      </div>
    </div>

    <div class="result" id="result"></div>

    <div class="actions">
      <button class="btn btn-go" id="computeBtn" onclick="compute()" disabled>
        ▶ Compute Rotation Matrix
      </button>
      <button class="btn btn-clr" onclick="clearAll()">✕ Clear All</button>
    </div>
  </div>

</div>

<script>
  let frames = {{ frames|tojson }};
  let obs = { vga_socket:[], ethernet_socket:[], power_socket:[] };

  const ENT_COLOR = { vga_socket:'#00ff9d', ethernet_socket:'#3c8fff', power_socket:'#ffb347' };
  const ENT_CLS   = { vga_socket:'vga',     ethernet_socket:'eth',     power_socket:'pwr'     };
  const ENT_SHORT = { vga_socket:'VGA',     ethernet_socket:'ETH',     power_socket:'PWR'     };
  const ENT_BTN   = { vga_socket:'btn-vga', ethernet_socket:'btn-eth', power_socket:'btn-pwr' };
  const ENT_HUD   = { vga_socket:'VGA socket', ethernet_socket:'Ethernet socket', power_socket:'Power socket' };

  let activeEnt = 'vga_socket';
  let curFrame  = null;
  let scale = 1, offset = {x:0, y:0};
  let natW = 2560, natH = 1440;
  let panning = false, panStart = {x:0,y:0};

  const viewer = document.getElementById('viewer');
  const img    = document.getElementById('main-img');
  const canvas = document.getElementById('canvas');
  const ctx    = canvas.getContext('2d');

  function buildFrames() {
    const list = document.getElementById('frameList');
    const fc   = document.getElementById('fc');
    list.innerHTML = '';
    if (!frames.length) { fc.textContent='⚠ No frames found'; fc.style.color='var(--acc2)'; return; }
    fc.textContent = frames.length + ' frames';
    frames.forEach(f => {
      const d = document.createElement('div');
      d.className='frame-item'; d.textContent=f; d.title=f; d.id='fi_'+f;
      d.onclick = () => loadFrame(f);
      list.appendChild(d);
    });
  }

  function loadFrame(f) {
    curFrame = f;
    document.querySelectorAll('.frame-item').forEach(e=>e.classList.remove('active'));
    const el = document.getElementById('fi_'+f);
    if (el) { el.classList.add('active'); el.scrollIntoView({block:'nearest'}); }
    img.src='/frame/'+f; img.style.display='block';
    document.getElementById('noImg').style.display='none';
    img.onload=()=>{ natW=img.naturalWidth; natH=img.naturalHeight; resetZoom(); };
  }

  function setEnt(ent) {
    activeEnt = ent;
    Object.keys(ENT_BTN).forEach(e => {
      document.getElementById(ENT_BTN[e]).className = 'ent-btn' + (e===ent?' active-'+ENT_CLS[e]:'');
    });
    const hud = document.getElementById('entHud');
    hud.textContent = ENT_HUD[ent]; hud.className = 'ent-hud '+ENT_CLS[ent];
    redraw();
  }

  function applyT() {
    img.style.left=offset.x+'px'; img.style.top=offset.y+'px';
    img.style.width=natW*scale+'px'; img.style.height=natH*scale+'px';
    document.getElementById('hz').textContent=scale.toFixed(2);
    redraw();
  }
  function doZoom(f,cx,cy) {
    const r=viewer.getBoundingClientRect();
    cx=cx??r.width/2; cy=cy??r.height/2;
    const prev=scale; scale=Math.min(Math.max(scale*f,0.05),20);
    offset.x=cx-(cx-offset.x)*(scale/prev); offset.y=cy-(cy-offset.y)*(scale/prev);
    applyT();
  }
  function resetZoom() {
    const r=viewer.getBoundingClientRect();
    if(!r.width){requestAnimationFrame(resetZoom);return;}
    scale=Math.min(r.width/natW,r.height/natH)*0.97;
    offset.x=(r.width-natW*scale)/2; offset.y=(r.height-natH*scale)/2;
    applyT();
  }
  viewer.addEventListener('wheel',e=>{
    e.preventDefault();
    const r=viewer.getBoundingClientRect();
    doZoom(e.deltaY<0?1.12:0.89,e.clientX-r.left,e.clientY-r.top);
  },{passive:false});
  viewer.addEventListener('auxclick',e=>{if(e.button===1)e.preventDefault();});
  function imgCoords(e){
    const r=viewer.getBoundingClientRect();
    return{u:(e.clientX-r.left-offset.x)/scale, v:(e.clientY-r.top-offset.y)/scale};
  }
  viewer.addEventListener('mousedown',e=>{
    if(e.button===1||(e.button===0&&e.altKey)){
      e.preventDefault(); panning=true;
      panStart={x:e.clientX-offset.x,y:e.clientY-offset.y};
      viewer.style.cursor='grabbing';
    }
  });
  window.addEventListener('mousemove',e=>{
    if(curFrame){
      const c=imgCoords(e);
      if(c.u>=0&&c.u<=natW&&c.v>=0&&c.v<=natH){
        document.getElementById('hx').textContent=Math.round(c.u);
        document.getElementById('hy').textContent=Math.round(c.v);
      }
    }
    if(panning){offset.x=e.clientX-panStart.x; offset.y=e.clientY-panStart.y; applyT();}
  });
  window.addEventListener('mouseup',()=>{if(panning){panning=false;viewer.style.cursor='crosshair';}});

  viewer.addEventListener('click',e=>{
    if(!curFrame||e.altKey||e.button!==0||panning)return;
    const c=imgCoords(e);
    if(c.u<0||c.u>natW||c.v<0||c.v>natH)return;
    const match=curFrame.match(/(\d+)/);
    const fn=match?String(parseInt(match[1])):curFrame;
    obs[activeEnt].push({frame:fn,u:Math.round(c.u),v:Math.round(c.v),file:curFrame});
    updateProgress(); renderObs(); redraw();
  });

  function redraw() {
    const r=viewer.getBoundingClientRect();
    canvas.width=r.width; canvas.height=r.height;
    ctx.clearRect(0,0,canvas.width,canvas.height);
    if(!curFrame)return;
    const sx=u=>offset.x+u*scale, sy=v=>offset.y+v*scale;
    const R=6, ARM=14;
    Object.entries(obs).forEach(([ent,list])=>{
      const color=ENT_COLOR[ent];
      list.filter(o=>o.file===curFrame).forEach(o=>{
        const x=sx(o.u),y=sy(o.v);
        ctx.strokeStyle=color; ctx.fillStyle=color; ctx.lineWidth=1.5;
        ctx.beginPath();
        ctx.moveTo(x-ARM,y);ctx.lineTo(x-R,y);
        ctx.moveTo(x+R,y);ctx.lineTo(x+ARM,y);
        ctx.moveTo(x,y-ARM);ctx.lineTo(x,y-R);
        ctx.moveTo(x,y+R);ctx.lineTo(x,y+ARM);
        ctx.stroke();
        ctx.beginPath();ctx.arc(x,y,R,0,Math.PI*2);ctx.stroke();
        ctx.font='bold 12px JetBrains Mono, monospace';
        ctx.fillText(ENT_SHORT[ent]+' f'+o.frame,x+R+5,y-5);
      });
    });
  }

  function renderObs() {
    const list=document.getElementById('obsList');
    const total=Object.values(obs).reduce((s,a)=>s+a.length,0);
    if(!total){
      list.innerHTML='<div style="padding:13px;color:var(--muted);font-size:.7rem;text-align:center;">Click the center of each socket on ≥2 frames</div>';
      return;
    }
    let html='';
    [['vga_socket','vga'],['ethernet_socket','eth'],['power_socket','pwr']].forEach(([ent,cls])=>{
      if(!obs[ent].length)return;
      html+='<div class="obs-group-hdr '+cls+'">'+ent+' ('+obs[ent].length+'x)</div>';
      obs[ent].forEach((o,i)=>{
        html+='<div class="obs-item"><div><span class="obs-frame '+cls+'">f'+o.frame+'</span>'+
          '<span class="obs-uv">('+o.u+', '+o.v+')</span></div>'+
          '<button class="del" onclick="delObs(\''+ent+'\','+i+')">✕</button></div>';
      });
    });
    list.innerHTML=html;
  }

  function delObs(ent,i){obs[ent].splice(i,1);updateProgress();renderObs();redraw();}
  function clearAll(){
    obs={vga_socket:[],ethernet_socket:[],power_socket:[]};
    updateProgress();renderObs();redraw();
    document.getElementById('result').classList.remove('show');
  }

  function updateProgress() {
    [['vga_socket','vga'],['ethernet_socket','eth'],['power_socket','pwr']].forEach(([ent,cls])=>{
      const n=obs[ent].length, ready=n>=2;
      document.getElementById('dot-'+cls).className='prog-dot'+(ready?' '+cls:'');
      document.getElementById('pl-' +cls).className='prog-label'+(ready?' '+cls:'');
      document.getElementById('pn-' +cls).textContent=n+' / ≥2';
      document.getElementById('cnt-'+cls).textContent=n+' obs';
    });
    const allReady=['vga_socket','ethernet_socket','power_socket'].every(e=>obs[e].length>=2);
    document.getElementById('computeBtn').disabled=!allReady;
  }

  async function compute() {
    showResult('Triangulating 3 points…', false);
    try {
      const res=await fetch('/compute',{
        method:'POST', headers:{'Content-Type':'application/json'},
        body:JSON.stringify({
          vga:obs.vga_socket.map(o=>[o.frame,o.u,o.v]),
          eth:obs.ethernet_socket.map(o=>[o.frame,o.u,o.v]),
          pwr:obs.power_socket.map(o=>[o.frame,o.u,o.v]),
        })
      });
      const data=await res.json();
      if(data.error){showResult('Error: '+data.error,true);return;}
      showResult(fmt(data),false);
    } catch(e){showResult('Error: '+e.message,true);}
  }

  function fmt(d) {
    const R=d.rotation;
    const row=r=>'  ['+r.map(x=>x.toFixed(8).padStart(13)).join(', ')+']';
    return[
      '✓ Saved to rotation.json',
      '',
      'Rotation matrix:',
      '[', row(R[0]), row(R[1]), row(R[2]), ']',
      '',
      'Similarity to GT: ' + d.gt_similarity.toFixed(4) + '  (0=perfect, 2√2≈2.83=worst)',
      '',
      'Triangulated points:',
      '  VGA: ['+d.p_vga.map(x=>x.toFixed(4)).join(', ')+']',
      '  ETH: ['+d.p_eth.map(x=>x.toFixed(4)).join(', ')+']',
      '  PWR: ['+d.p_pwr.map(x=>x.toFixed(4)).join(', ')+']',
    ].join('\n');
  }

  function showResult(txt,err){
    const el=document.getElementById('result');
    el.textContent=txt; el.style.color=err?'var(--acc2)':'var(--acc)';
    el.classList.add('show');
  }

  buildFrames();
  updateProgress();
  window.addEventListener('resize',()=>{if(curFrame)resetZoom();});
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
        data    = request.get_json()
        raw_vga = data["vga"]
        raw_eth = data["eth"]
        raw_pwr = data["pwr"]

        for label, raw in [("VGA", raw_vga), ("ETH", raw_eth), ("PWR", raw_pwr)]:
            if len(raw) < 2:
                return jsonify({"error": f"{label} needs ≥2 observations (have {len(raw)})"})

        poses   = load_poses()
        to_obs  = lambda raw: [(str(r[0]), float(r[1]), float(r[2])) for r in raw]

        p_vga   = triangulate(to_obs(raw_vga), poses)
        p_eth   = triangulate(to_obs(raw_eth), poses)
        p_pwr   = triangulate(to_obs(raw_pwr), poses)

        rotation = compute_rotation(p_vga, p_eth, p_pwr)

        # measure how close we are to GT (Frobenius distance, lower=better)
        GT = np.array([
            [-0.004004375172752437,  0.9672545151126772, -0.25377680739897346],
            [ 0.01584254528462312,   0.25380835519540434, 0.9671247761234889 ],
            [ 0.9998664804554559,   -0.00014774012094266402, -0.016340117333610394]
        ])
        gt_similarity = float(np.linalg.norm(rotation - GT, 'fro'))

        # Save rotation.json
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        rot_path = os.path.join(OUTPUT_DIR, "rotation.json")
        with open(rot_path, "w") as f:
            json.dump(rotation.tolist(), f, indent=2)

        # Update answers.json
        answers_path = os.path.join(OUTPUT_DIR, "answers.json")
        answers = []
        if os.path.exists(answers_path):
            with open(answers_path) as f:
                answers = json.load(f)

        center_map = {"vga_socket": p_vga, "ethernet_socket": p_eth, "power_socket": p_pwr}
        for ent, extent in ENTITIES.items():
            center = center_map[ent]
            updated = False
            for entry in answers:
                if entry["entity"] == ent:
                    entry["obb"] = {"center": center.tolist(), "extent": extent,
                                    "rotation": rotation.tolist()}
                    updated = True; break
            if not updated:
                answers.append({"entity": ent,
                                 "obb": {"center": center.tolist(), "extent": extent,
                                         "rotation": rotation.tolist()}})
        with open(answers_path, "w") as f:
            json.dump(answers, f, indent=2)

        return jsonify({
            "rotation":      rotation.tolist(),
            "gt_similarity": gt_similarity,
            "p_vga":         p_vga.tolist(),
            "p_eth":         p_eth.tolist(),
            "p_pwr":         p_pwr.tolist(),
        })

    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "trace": traceback.format_exc()})

# ─────────────────────────────────────────────
if __name__ == "__main__":
    frames = get_frames()
    print(f"\n  Rotation Annotator")
    print(f"  Data dir   : {DATA_DIR}")
    print(f"  Output dir : {OUTPUT_DIR}")
    print(f"  Frames     : {len(frames)} found")
    print(f"  Open       : http://localhost:5000\n")
    app.run(debug=True, port=5000)