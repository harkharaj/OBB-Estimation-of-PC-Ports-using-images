import json
import numpy as np
import cv2
import os

# --- Configuration ---
# Hardcoding the VGA rotation matrix to use as our flat panel normal
VGA_ROTATION = np.array([
    [-0.004004375172752437, 0.9672545151126772, -0.25377680739897346],
    [0.01584254528462312, 0.25380835519540434, 0.9671247761234889],
    [0.9998664804554559, -0.00014774012094266402, -0.016340117333610394]
])

# Physical extents from the handoff document (Half-extents in meters: [W, H, L])
EXTENTS = {
    "ethernet_socket": [0.008, 0.0065, 0.0055],
    "power_socket": [0.014, 0.011, 0.0075]
}

# === SET YOUR ABSOLUTE DIRECTORY PATH HERE ===
# Ensure you use raw strings (r"") or double backslashes for Windows paths
DATA_DIR = r"C:\Users\harkh\OneDrive\Desktop\ROBOTIC_PERCEPTION_FINAL_PROJECT\Data"

# === SET YOUR TARGET HERE ===
# Change this to "power_socket" when you are ready to do the other one!
ENTITY_NAME = "power_socket" 
TARGET_EXTENT = EXTENTS[ENTITY_NAME]

# FILL THIS IN: Your 2D pixel observations (frame_number_string, u_pixel, v_pixel)
OBSERVATIONS = [
    # This is the confirmed ethernet observation
    ("400", 1384, 906), 
    ("365",1530,1206),
    ("531",1141,1206)
    # Add your 2-3 extra frames here:
    # ("461", u, v),
    # ("468", u, v),
]

# The frame you want to draw the final 3D OBB onto to verify
TARGET_DRAW_FRAME = "319" 

def load_data():
    # Construct the full paths to the JSON files
    poses_path = os.path.join(DATA_DIR, 'poses.json')
    intrinsics_path = os.path.join(DATA_DIR, 'intrinsic.json')

    # Load poses
    with open(poses_path, 'r') as f:
        poses = json.load(f)
    
    # Load camera matrix K
    with open(intrinsics_path, 'r') as f:
        intrinsics_data = json.load(f)
        K = np.array(intrinsics_data['camera_matrix'])
        
    return poses, K

def triangulate(observations, poses, K):
    A = []
    for frame_key, u, v in observations:
        c2w = np.array(poses[frame_key])
        w2c = np.linalg.inv(c2w)
        R = w2c[:3, :3]
        t = w2c[:3, 3]
        Rt = np.hstack([R, t.reshape(3, 1)])
        P = K @ Rt
        
        A.append(v * P[2] - P[1])
        A.append(u * P[2] - P[0])
        
    A = np.array(A)
    _, _, Vt = np.linalg.svd(A)
    X = Vt[-1]
    return X[:3] / X[3]

def get_obb_corners(center, extent, rotation):
    corners = []
    dx, dy, dz = extent
    # Generate all 8 combinations of +/- extents
    for x in [-dx, dx]:
        for y in [-dy, dy]:
            for z in [-dz, dz]:
                corners.append([x, y, z])
                
    corners = np.array(corners).T # 3x8 matrix
    world_corners = rotation @ corners + center.reshape(3, 1)
    return world_corners.T # 8x3 matrix

def draw_obb_on_image(frame_key, poses, K, corners_3d):
    # Construct the full path to the specific image
    # Note: using .jpg based on your prompt, adjust if they are .png
    image_name = f"frame_{int(frame_key):06d}.png" 
    full_image_path = os.path.join(DATA_DIR, image_name)
    
    img = cv2.imread(full_image_path)
    
    if img is None:
        print(f"Error: Could not read {full_image_path} from disk. Check path and file extension (.jpg vs .png).")
        return

    c2w = np.array(poses[frame_key])
    w2c = np.linalg.inv(c2w)
    R = w2c[:3, :3]
    t = w2c[:3, 3]
    Rt = np.hstack([R, t.reshape(3, 1)])
    P = K @ Rt

    # Project 3D corners to 2D pixels
    pts_2d = []
    for corner in corners_3d:
        X_h = np.append(corner, 1)
        x_h = P @ X_h
        u = int(x_h[0] / x_h[2])
        v = int(x_h[1] / x_h[2])
        pts_2d.append((u, v))

    # Edges connecting the 8 corners of a box
    edges = [
        (0,1), (1,3), (3,2), (2,0), # back face
        (4,5), (5,7), (7,6), (6,4), # front face
        (0,4), (1,5), (2,6), (3,7)  # side connecting edges
    ]

    # Draw the lines
    for p1, p2 in edges:
        cv2.line(img, pts_2d[p1], pts_2d[p2], (0, 255, 0), 2)
        
    output_filename = f"OBB_Projection_{frame_key}.jpg"
    
    # Save the output file in the same Data directory
    output_path = os.path.join(DATA_DIR, output_filename)
    cv2.imwrite(output_path, img)
    print(f"Success! Drawn OBB saved to {output_path}")

def main():
    if len(OBSERVATIONS) < 2:
        print("Please add at least 2 observations to the OBSERVATIONS list.")
        return

    poses, K = load_data()
    
    # 1. Triangulate the 3D center
    center_3d = triangulate(OBSERVATIONS, poses, K)
    print(f"\nComputed 3D Center: {center_3d.tolist()}")
    
    # 2. Get the 3D corners
    corners_3d = get_obb_corners(center_3d, TARGET_EXTENT, VGA_ROTATION)
    
    # 3. Draw the result on the target frame
    draw_obb_on_image(TARGET_DRAW_FRAME, poses, K, corners_3d)

    # 4. Print the final JSON string ready for answers.json
    final_json = {
        "entity": ENTITY_NAME,
        "obb": {
            "center": center_3d.tolist(),
            "extent": TARGET_EXTENT,
            "rotation": VGA_ROTATION.tolist()
        }
    }
    print("\nFinal JSON block for answers.json:")
    print(json.dumps(final_json, indent=2))

if __name__ == "__main__":
    main()