from flask import Flask, request, jsonify
from flask_cors import CORS #for react and flask connection
from werkzeug.utils import secure_filename
import os

from src.pipeline.predict_pipeline import PredictPipeline
from src.pipeline.disease_predict_pipeline import DiseasePredictPipeline

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "static/uploads"
DISEASE_UPLOAD_FOLDER = "static/disease"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DISEASE_UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["DISEASE_UPLOAD_FOLDER"] = DISEASE_UPLOAD_FOLDER

CROP_MAP = {
    0: "Jute",
    1: "Maize",
    2: "Rice",
    3: "Sugarcane",
    4: "Wheat"
}

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["file"]
        filename = secure_filename(file.filename)

        file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(file_path)

        pipeline = PredictPipeline()
        crop_idx, confidence = pipeline.predict(file_path)

        return jsonify({
            "crop": CROP_MAP[crop_idx],
            "confidence": round(confidence, 2),
            "image": f"http://localhost:5000/static/uploads/{filename}"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


DISEASE_MAP = {
    0: "Bacterial Blight",
    1: "Blast",
    2: "Brown Spot",
    3: "Healthy",
    4: "Mosaic",
    5: "Red Rot",
    6: "Rust",
    7: "Tungro",
    8: "Yellow"
}

@app.route("/predict-disease", methods=["POST"])
def predict_disease():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["file"]
        filename = secure_filename(file.filename)

        file_path = os.path.join(app.config["DISEASE_UPLOAD_FOLDER"], filename)
        file.save(file_path)

        pipeline = DiseasePredictPipeline()
        disease_idx, confidence = pipeline.predict(file_path)

        return jsonify({
            "disease": DISEASE_MAP[disease_idx],
            "confidence": round(confidence, 2),
            "image": f"http://localhost:5000/static/disease/{filename}"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)

