import json
import numpy as np
import os

# === CONFIGURE YOUR ABSOLUTE DIRECTORY PATH ===
DATA_DIR = r"C:\Users\harkh\OneDrive\Desktop\ROBOTIC_PERCEPTION_FINAL_PROJECT\Data"

# --- 2D Observations for Calibration ---
# Pick 3 points that define the plane of the back panel.
# We use the centers of the VGA, Ethernet, and Power sockets.
OBS_VGA = [("365", 1508, 286), ("426", 1540, 367), ("515", 960, 548)]
OBS_ETH = [("515", 1091, 1076), ("333", 1574, 423), ("468", 1624, 574)]
OBS_PWR = [("426", 1550, 854), ("461", 1890, 850), ("471", 1300, 860)]

def load_calibration_data():
    with open(os.path.join(DATA_DIR, 'poses.json'), 'r') as f:
        poses = json.load(f)
    with open(os.path.join(DATA_DIR, 'intrinsic.json'), 'r') as f:
        K = np.array(json.load(f)['camera_matrix'])
    return poses, K

def triangulate(observations, poses, K):
    A = []
    for frame_key, u, v in observations:
        # P = K [R|t] where [R|t] is world-to-camera
        w2c = np.linalg.inv(np.array(poses[frame_key]))
        P = K @ w2c[:3, :]
        A.append(v * P[2] - P[1])
        A.append(u * P[2] - P[0])
    _, _, Vt = np.linalg.svd(np.array(A))
    X = Vt[-1]
    return X[:3] / X[3]

def main():
    poses, K = load_calibration_data()

    print("Triangulating reference points...")
    p_vga = triangulate(OBS_VGA, poses, K)
    p_eth = triangulate(OBS_ETH, poses, K)
    p_pwr = triangulate(OBS_PWR, poses, K)

    # 1. Define X-axis: Vector from VGA to Ethernet (Width direction)
    x_axis = p_eth - p_vga
    x_axis /= np.linalg.norm(x_axis)

    # 2. Define Z-axis: Normal to the panel[cite: 2]
    # We take the cross product of the vector to Ethernet and the vector to Power
    v_pwr = p_pwr - p_vga
    z_axis = np.cross(x_axis, v_pwr)
    z_axis /= np.linalg.norm(z_axis)

    # 3. Define Y-axis: Up direction (must be orthogonal to X and Z)[cite: 2]
    y_axis = np.cross(z_axis, x_axis)

    # Construct the 3x3 matrix where columns are the principal axes
    # Matrix = [x_axis | y_axis | z_axis]
    rotation_matrix = np.stack([x_axis, y_axis, z_axis], axis=1)

    print("\n====================================================")
    print("COMMON ROTATION MATRIX (Computed from Geometry)")
    print("====================================================")
    print(rotation_matrix.tolist())
    print("====================================================\n")
    
    print("You can now copy this list and use it as the 'rotation'")
    print("for any entity on the back panel.")

if __name__ == "__main__":
    main()