# Robotic Perception — Final Project (CP260-2026)

Metric-semantic reconstruction of a desktop scene using multi-view geometry. Given a set of posed RGB images and camera intrinsics, the system estimates the 6-DOF oriented bounding box (OBB) of hardware components on a motherboard I/O backplate.

---

## Project structure

```
Robotic_Perception/
├── src/
│   ├── annotator_with_all_ports.py   # Main annotation + OBB pipeline (Flask UI)
│   ├── comparison.py                 # Visual comparison: new vs old rotation matrix
│   └── validate_obb.py               # IoU validation against professor's ground truth
├── Rotation/
│   ├── rotation_matrix.py            # Computes panel rotation from 3 triangulated points
│   ├── rotation.json                 # Saved rotation matrix (output of rotation_matrix.py)
│   └── rotation_generation.txt       # Notes on rotation derivation
├── Data/                             # 16 posed RGB frames (frame_000XXX.png)
├── Camera_Properties/
│   ├── poses.json                    # Camera-to-world pose matrices (4×4) per frame
│   └── intrinsic.json                # Pinhole camera intrinsics (K matrix, undistorted)
├── Answers/
│   └── answers.json                  # OBB submission file (center, extent, rotation per entity)
├── Output_Figures/                   # Saved OBB comparison images
├── sample_answers.json               # Professor-provided ground truth for vga_socket
├── README.md
└── LICENSE
```
---

## How to run — step by step

### Step 1 — Compute the panel rotation matrix (run once)

```bash
cd Rotation
python rotation_matrix.py
```

Open `http://localhost:5000` in your browser.

- Click the **center** of the VGA, Ethernet, and Power sockets on at least 2 different frames each
- Hit **Compute Rotation Matrix**
- The matrix is saved to `Rotation/rotation.json` automatically
- Check the "Similarity to GT" score printed — lower is better (0 = identical to reference)

### Step 2 — Annotate OBBs for each entity

```bash
cd src
python annotator_with_all_ports.py
```

Open `http://localhost:5000`.

1. Select an entity from the dropdown (e.g. `vga_socket`)
2. Click the socket center on **≥ 3 frames** — more frames = better triangulation
3. Pick frames where the socket is clearly visible and unobstructed
4. Click **Triangulate + Compute OBB**
5. Check the reprojection errors printed — under 10px is good, under 5px is excellent
6. Click **Validate** on any frame to see the OBB drawn over the image (zoom with scroll, pan with click+drag)
7. Repeat for `ethernet_socket` and `power_socket`

The OBBs are saved automatically to `Answers/answers.json` after each Compute.

### Step 3 — Validate VGA against ground truth

```bash
# Edit FRAME = "390" at the top to any frame number you annotated on
python src/validate_obb.py
```

Prints IoU score and opens an interactive window showing:
- **Green box** = ground truth (from `sample_answers.json`)
- **Blue box** = predicted OBB (from `Answers/answers.json`)

Use scroll to zoom in, click+drag to pan, press Q or click X to close.

### Step 4 — Compare rotation matrices (optional)

```bash
python src/comparison.py
```

Open `http://localhost:5000`. Annotate all 3 sockets, pick a validation frame, click Compare. Saves an overlay image to `Output_Figures/` showing new rotation (green) vs ground truth rotation (red).

### Step 5 — Verify final answers.json before submission

```bash
python -c "
import json
d = json.load(open('Answers/answers.json'))
for e in d:
    obb = e['obb']
    print(f\"{e['entity']}: center={[round(x,4) for x in obb['center']]} | rot_rows={len(obb['rotation'])} | extent={obb['extent']}\")
"
```

All entities must have 3 rotation rows, numeric centers, and no placeholder values.

---

## Pipeline overview

```
Posed images + camera intrinsics
         │
         ▼
  Click socket centers          ← annotator_with_all_ports.py
  on ≥ 3 frames each
         │
         ▼
  SVD-based multi-view           ← triangulate()
  triangulation (DLT)
         │
         ▼
  3D center coordinates
         │
  + Panel rotation matrix        ← rotation_matrix.py (computed once from geometry)
  + Physical extents (prior)
         │
         ▼
  Oriented Bounding Box (OBB)
  center, extent, rotation
         │
         ▼
  answers.json  ──► Submission
```

---

## Submission format

`Answers/answers.json` — list of OBB entries:

```json
[
  {
    "entity": "vga_socket",
    "obb": {
      "center": [X, Y, Z],
      "extent": [W, H, L],
      "rotation": [
        [r00, r01, r02],
        [r10, r11, r12],
        [r20, r21, r22]
      ]
    }
  }
]
```

- `center`: 3D world coordinates of the socket center (meters)
- `extent`: half-widths along each axis (meters)
- `rotation`: 3×3 rotation matrix — columns are the OBB principal axes in world space

---

## Key design decisions

**Shared rotation matrix**: All sockets lie on the same planar I/O backplate. The panel's orientation is derived once by triangulating 3 reference points (VGA, Ethernet, Power centers) and computing an orthonormal basis from the panel geometry. This shared rotation is then used for all entities on the backplate.

**SVD triangulation**: The Direct Linear Transform (DLT) method. For each 2D pixel observation (u, v) in frame i, two linear equations are formed from the projection P·X = x. All equations are stacked into matrix A and solved via SVD - the 3D point is the last row of V^T, normalized by the homogeneous coordinate.

---

## Dependencies

| Package | Version | Purpose |
|---|---|---|
| flask | ≥ 2.0 | Web UI for annotation tools |
| numpy | ≥ 1.21 | Matrix math, SVD, projections |
| pillow | ≥ 9.0 | Image drawing in validate route |
| opencv-python | ≥ 4.5 | Interactive validation window |
