# Robotic Perception Final Project

Metric-semantic reconstruction of a desktop scene to estimate 
the 6-DOF pose (OBB) of hardware components on the motherboard 
backplate using multi-view triangulation.

## Folder structure
ROBOTIC_PERCEPTION_FINAL_PROJECT/
├── src/
│   ├── annotator_with_all_ports.py   # Main annotation + OBB tool
│   ├── compute_obb.py                # Standalone triangulation
│   ├── get_rotation.py               # Panel rotation from geometry
│   └── validate_obb.py               # IoU validation against GT
├── Data/                             # Frame images
├── Camera_Properties/                # poses.json, intrinsic.json
├── Answers/                          # answers.json (submission file)
├── docs/                             # Report PDF
└── sample_answers.json               # Prof-provided VGA ground truth

## Setup
pip install flask numpy pillow opencv-python shapely

## How to run

### 1. Annotate sockets and compute OBBs
python src/annotator_with_all_ports.py
# Open http://localhost:5000
# Select entity → click socket center on ≥3 frames → Compute

### 2. Validate against ground truth (VGA only)
python src/validate_obb.py   # edit FRAME = "319" at top

### 3. Standalone OBB computation (no UI)
python src/compute_obb.py    # edit ENTITY_NAME and OBSERVATIONS at top

## Submission format
answers.json in Answers/ folder — contains OBB for each entity:
center (3D world coords), extent (half-widths), rotation (3×3 matrix)
