from flask import Flask, render_template, request, jsonify, send_file
import os
import json
import numpy as np

app = Flask(__name__)

# === CONFIGURE YOUR DATA DIRECTORY HERE ===
DATA_DIR = r"C:\Users\harkh\OneDrive\Desktop\ROBOTIC_PERCEPTION_FINAL_PROJECT\Data"

# Load Poses and Intrinsics once on startup
with open('poses.json', 'r') as f:
    POSES = json.load(f)
with open('intrinsic.json', 'r') as f:
    intrinsics_data = json.load(f)
    K = np.array(intrinsics_data['camera_matrix'])

VGA_ROTATION = np.array([
    [-0.004004375172752437, 0.9672545151126772, -0.25377680739897346],
    [0.01584254528462312, 0.25380835519540434, 0.9671247761234889],
    [0.9998664804554559, -0.00014774012094266402, -0.016340117333610394]
])

def triangulate(obs_list):
    A = []
    for frame_key, u, v in obs_list:
        if frame_key not in POSES:
            continue
        c2w = np.array(POSES[frame_key])
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

@app.route('/')
def index():
    # Find all images in the Data directory to populate the dropdown
    images = [f for f in os.listdir(DATA_DIR) if f.endswith(('.png', '.jpg', '.jpeg'))]
    images.sort()
    return render_template('index.html', images=images)

# Route to securely serve images from your local PC to the browser
@app.route('/image/<filename>')
def get_image(filename):
    safe_path = os.path.join(DATA_DIR, filename)
    if os.path.exists(safe_path):
        return send_file(safe_path)
    return "Image not found", 404

# Route that handles the math when you click "Compute"
@app.route('/compute', methods=['POST'])
def compute():
    data = request.json
    observations = data.get('observations', [])
    extent = data.get('extent', [0,0,0])
    entity_name = data.get('entity_name', 'unknown')

    if len(observations) < 2:
        return jsonify({"error": "Need at least 2 point observations to triangulate."}), 400

    # Convert the JS observations into the format our function expects
    clean_obs = []
    for obs in observations:
        # Extract just the frame number from the filename (e.g. "frame_000426.png" -> "426")
        frame_str = str(int(obs['image'].split('_')[1].split('.')[0]))
        clean_obs.append((frame_str, obs['x'], obs['y']))

    # Do the math
    center_3d = triangulate(clean_obs)

    # Format the final JSON
    result = {
        "entity": entity_name,
        "obb": {
            "center": center_3d.tolist(),
            "extent": extent,
            "rotation": VGA_ROTATION.tolist()
        }
    }
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, port=5000)