from flask import Flask, request, jsonify
from flask_cors import CORS #for react and flask connection
from werkzeug.utils import secure_filename
import os
import json
from dotenv import load_dotenv

load_dotenv()

# LangSmith Tracing
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "CropAI"

from langchain_core.messages import HumanMessage, AIMessage
from src.pipeline.predict_pipeline import PredictPipeline
from src.pipeline.disease_predict_pipeline import DiseasePredictPipeline
from src.agents.brain_agent import BrainAgent

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


CHAT_UPLOAD_FOLDER = "static/chat"
os.makedirs(CHAT_UPLOAD_FOLDER, exist_ok=True)

brain = BrainAgent()

@app.route("/chat", methods=["POST"])
def chat():
    try:
        message = request.form.get("message", "")
        chat_history_raw = request.form.get("chat_history", "[]")

        # Parse chat history from frontend into LangChain message objects
        chat_history = []
        for msg in json.loads(chat_history_raw):
            if msg["role"] == "user":
                chat_history.append(HumanMessage(content=msg["text"]))
            elif msg["role"] == "bot":
                chat_history.append(AIMessage(content=msg["text"]))

        # Handle file upload
        image_path = None
        image_url = None
        has_file = "file" in request.files and request.files["file"].filename != ""
        if has_file:
            file = request.files["file"]
            filename = secure_filename(file.filename)
            file_path = os.path.join(CHAT_UPLOAD_FOLDER, filename)
            file.save(file_path)
            image_path = file_path
            image_url = f"http://localhost:5000/static/chat/{filename}"

        # Run the agent
        response_text = brain.run(
            user_message=message,
            chat_history=chat_history,
            image_path=image_path
        )

        result = {"response": response_text}
        if image_url:
            result["image"] = image_url

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)

