import json
import numpy as np
import os
import cv2

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
CAM_DIR    = os.path.join(BASE_DIR, "Camera_Properties")
ANSWER_DIR = os.path.join(BASE_DIR, "Answers")
DATA_DIR   = os.path.join(BASE_DIR, "Data")

# ── CHANGE THIS to whichever frame you want to test on ──
FRAME = "390"
# ────────────────────────────────────────────────────────

poses = json.load(open(os.path.join(CAM_DIR,    "poses.json")))
K     = np.array(json.load(open(os.path.join(CAM_DIR, "intrinsic.json")))["camera_matrix"])
pred  = json.load(open(os.path.join(ANSWER_DIR, "answers.json")))
gt    = json.load(open(os.path.join(BASE_DIR,   "sample_answers.json")))

def get_obb(data, entity):
    return next(d["obb"] for d in data if d["entity"] == entity)

def corners_3d(obb):
    c, R, e = np.array(obb["center"]), np.array(obb["rotation"]), obb["extent"]
    offsets = np.array([[sx*e[0], sy*e[1], sz*e[2]]
                        for sx in (-1,1) for sy in (-1,1) for sz in (-1,1)])
    return (R @ offsets.T).T + c

def project(corners, frame_key):
    w2c = np.linalg.inv(np.array(poses[frame_key]))
    P   = K @ w2c[:3, :]
    pts = []
    for corner in corners:
        x = P @ np.append(corner, 1.0)
        pts.append((x[0]/x[2], x[1]/x[2]))
    return pts

def bbox_iou(pts_a, pts_b):
    def bbox(pts):
        xs, ys = zip(*pts)
        return min(xs), min(ys), max(xs), max(ys)
    ax1,ay1,ax2,ay2 = bbox(pts_a)
    bx1,by1,bx2,by2 = bbox(pts_b)
    ix = max(0, min(ax2,bx2) - max(ax1,bx1))
    iy = max(0, min(ay2,by2) - max(ay1,by1))
    inter = ix * iy
    union = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter
    return inter / union if union > 0 else 0.0

def draw_and_show(frame_key, pred_pts, gt_pts):
    """Draw GT (green) and Pred (blue) boxes on the frame and show with zoom support."""
    img_path = os.path.join(DATA_DIR, f"frame_{int(frame_key):06d}.png")
    img = cv2.imread(img_path)
    if img is None:
        print(f"Could not load image: {img_path}")
        return

    def bbox_rect(pts, color, label):
        xs = [int(p[0]) for p in pts]
        ys = [int(p[1]) for p in pts]
        x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        cv2.putText(img, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2, cv2.LINE_AA)

    bbox_rect(gt_pts,   (0, 255, 0),   "GT")
    bbox_rect(pred_pts, (255, 180, 0), "Pred")

    # ── Interactive zoom window ───────────────────────────────────────────────
    win = "Validation  |  scroll=zoom  |  click+drag=pan  |  Q=quit"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    H, W = img.shape[:2]
    disp_w, disp_h = 1280, 720
    cv2.resizeWindow(win, disp_w, disp_h)

    zoom   = 1.0
    pan    = [0, 0]
    drag   = False
    drag_s = [0, 0]

    def render():
        zh = int(H / zoom)
        zw = int(W / zoom)
        cx = max(zw//2, min(W - zw//2, pan[0]))
        cy = max(zh//2, min(H - zh//2, pan[1]))
        x1c = max(0, cx - zw//2);  x2c = min(W, cx + zw//2)
        y1c = max(0, cy - zh//2);  y2c = min(H, cy + zh//2)
        crop = img[y1c:y2c, x1c:x2c]
        return cv2.resize(crop, (disp_w, disp_h), interpolation=cv2.INTER_LINEAR)

    # Start pan centred on image
    pan[0], pan[1] = W // 2, H // 2

    def on_mouse(event, x, y, flags, _):
        nonlocal drag, zoom
        # Scale mouse coords back to full image space
        zh = int(H / zoom); zw = int(W / zoom)
        cx = max(zw//2, min(W - zw//2, pan[0]))
        cy = max(zh//2, min(H - zh//2, pan[1]))
        ix = cx - zw//2 + int(x * zw / disp_w)
        iy = cy - zh//2 + int(y * zh / disp_h)

        if event == cv2.EVENT_MOUSEWHEEL:
            factor = 1.15 if flags > 0 else 0.87
            zoom   = max(1.0, min(20.0, zoom * factor))
            pan[0], pan[1] = ix, iy
        elif event == cv2.EVENT_LBUTTONDOWN:
            drag = True; drag_s[0] = ix; drag_s[1] = iy
        elif event == cv2.EVENT_MOUSEMOVE and drag:
            pan[0] -= ix - drag_s[0]
            pan[1] -= iy - drag_s[1]
            drag_s[0] = ix; drag_s[1] = iy
        elif event == cv2.EVENT_LBUTTONUP:
            drag = False

    cv2.setMouseCallback(win, on_mouse)

    print(f"\nShowing frame {frame_key}  —  scroll to zoom, click+drag to pan, press Q to close")
    while True:
        cv2.imshow(win, render())
        key = cv2.waitKey(20) & 0xFF
        if key in (ord('q'), ord('Q'), 27):
            break
    cv2.destroyAllWindows()

# ── Run ───────────────────────────────────────────────────────────────────────
entity = "vga_socket"

pred_obb = get_obb(pred, entity)
gt_obb   = get_obb(gt,   entity)

pred_pts = project(corners_3d(pred_obb), FRAME)
gt_pts   = project(corners_3d(gt_obb),   FRAME)

iou = bbox_iou(pred_pts, gt_pts)

print(f"\nEntity : {entity}")
print(f"Frame  : {FRAME}")
print(f"IoU    : {iou:.4f}")
print(f"\nPred center : {pred_obb['center']}")
print(f"GT   center : {gt_obb['center']}")

draw_and_show(FRAME, pred_pts, gt_pts)
