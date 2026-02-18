import cv2
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
import pytesseract
from PIL import Image
from deepface import DeepFace


# Tesseract path
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

app = Flask(__name__)
CORS(app)

stored_faces = []

# REAL AI FACE MATCH
def compare_faces(id_path, selfie_path):
    try:
        result = DeepFace.verify(id_path, selfie_path, enforce_detection=False)
        confidence = (1 - result["distance"]) * 100

        if result["verified"]:
            return round(confidence,2)
        else:
            return round(confidence/2,2)

    except Exception as e:
        print("Face error:", e)
        return 20

# RISK ENGINE
def calculate_risk(match, duplicate):
    risk = 0
    if match < 60:
        risk += 50
    if duplicate:
        risk += 80
    return risk

@app.route("/verify", methods=["POST"])
def verify():
    id_file = request.files["id"]
    selfie_file = request.files["selfie"]

    id_path = "temp_id.jpg"
    selfie_path = "temp_selfie.jpg"

    id_file.save(id_path)
    selfie_file.save(selfie_path)

    match = compare_faces(id_path, selfie_path)

    selfie_img = cv2.imread(selfie_path)

    duplicate = False
    for face in stored_faces:
        diff = np.mean((face - selfie_img) ** 2)
        if diff < 200:
            duplicate = True

    stored_faces.append(selfie_img)

    risk = calculate_risk(match, duplicate)

    status = "VERIFIED"
    if risk > 70:
        status = "FRAUD DETECTED"

    print("MATCH:", match, "RISK:", risk, "STATUS:", status)

    return jsonify({
        "face_match": match,
        "duplicate": duplicate,
        "risk_score": risk,
        "status": status
    })

@app.route("/ocr", methods=["POST"])
def ocr():
    file = request.files["id"]
    img = Image.open(file)
    text = pytesseract.image_to_string(img)
    return jsonify({"extracted_text": text})

if __name__ == "__main__":
    app.run(port=5000, debug=True)
