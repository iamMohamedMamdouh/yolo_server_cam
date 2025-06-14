from flask import Flask, request, jsonify
from ultralytics import YOLO
import cv2
import numpy as np
import base64

app = Flask(__name__)
model = YOLO("yolov8n-pose.pt")

def decode_image(b64_string):
    img_data = base64.b64decode(b64_string)
    np_arr = np.frombuffer(img_data, np.uint8)
    return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

def encode_image(img):
    _, buffer = cv2.imencode('.jpg', img)
    return base64.b64encode(buffer).decode('utf-8')

@app.route("/detect", methods=["POST"])
def detect_pose():
    data = request.json
    image = decode_image(data["image"])
    results = model(image)
    result = results[0]

    num_people = 0
    for kp in result.keypoints.xy:
        if len(kp) > 0:
            num_people += 1

    annotated = result.plot()
    encoded = encode_image(annotated)

    return jsonify({
        "people": num_people,
        "image": encoded
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860)
