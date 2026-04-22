from flask import Flask, request, jsonify
from flask_cors import CORS #for react and flask connection
from werkzeug.utils import secure_filename
import os
import json
import threading
from dotenv import load_dotenv

load_dotenv()

# LangSmith Tracing
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "CropAI"

from src.logger import logging   # sets up the log file in logs/
from langchain_core.messages import HumanMessage, AIMessage
from src.pipeline.predict_pipeline import PredictPipeline
from src.pipeline.disease_predict_pipeline import DiseasePredictPipeline
from src.agents.brain_agent import BrainAgent
from src.agents.tools import get_last_rag_chunks, reset_last_rag_chunks
from typing import Optional, List
from src.RAG.faithfulness import FaithfulnessEvaluator, _format_context

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "static/uploads"
DISEASE_UPLOAD_FOLDER = "static/disease"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DISEASE_UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["DISEASE_UPLOAD_FOLDER"] = DISEASE_UPLOAD_FOLDER

ALLOWED_IMAGE_EXT = {"png", "jpg", "jpeg", "gif", "bmp", "webp"}


def _is_image(filename: str) -> bool:
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_IMAGE_EXT
    )


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
        if not filename or not _is_image(filename):
            return jsonify({"error": "Unsupported file type. Upload a PNG/JPG image."}), 400

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
        if not filename or not _is_image(filename):
            return jsonify({"error": "Unsupported file type. Upload a PNG/JPG image."}), 400

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

# Lazy faithfulness evaluator — built on first /chat that has RAG chunks so
# Flask boot stays fast.
_faithfulness_evaluator: Optional[FaithfulnessEvaluator] = None
_faithfulness_lock = threading.Lock()


def _get_faithfulness_evaluator() -> FaithfulnessEvaluator:
    global _faithfulness_evaluator
    if _faithfulness_evaluator is None:
        with _faithfulness_lock:
            if _faithfulness_evaluator is None:
                _faithfulness_evaluator = FaithfulnessEvaluator()
    return _faithfulness_evaluator


def _judge_async(query: str, chunks: list, answer: str) -> None:
    """Run the LLM-as-judge faithfulness check off the request thread.

    Why background: judge call adds ~1s of Groq latency that the user shouldn't
    pay for. Verdict goes to the log file alongside the chat trace.
    """
    try:
        evaluator = _get_faithfulness_evaluator()
        context = _format_context(chunks)
        verdict = evaluator.judge_answer(query, answer, context)
        logging.info(
            f"FAITHFULNESS  : verdict={verdict['verdict']} "
            f"reason={verdict['reason']!r}"
        )
    except Exception as exc:
        logging.error(f"Faithfulness check failed: {exc}")

@app.route("/chat", methods=["POST"])
def chat():
    try:
        message = request.form.get("message", "").strip()
        chat_history_raw = request.form.get("chat_history", "[]")

        # Parse chat history — tolerate malformed/missing JSON instead of crashing.
        chat_history = []
        try:
            parsed = json.loads(chat_history_raw) if chat_history_raw else []
            if not isinstance(parsed, list):
                parsed = []
        except (json.JSONDecodeError, TypeError):
            logging.warning("Invalid chat_history JSON — ignoring")
            parsed = []

        for msg in parsed:
            if not isinstance(msg, dict):
                continue
            role = msg.get("role")
            text = msg.get("text", "")
            if role == "user":
                chat_history.append(HumanMessage(content=text))
            elif role == "bot":
                chat_history.append(AIMessage(content=text))

        # Handle file upload (validate extension)
        image_path = None
        image_url = None
        filename = None
        has_file = "file" in request.files and request.files["file"].filename != ""
        if has_file:
            file = request.files["file"]
            filename = secure_filename(file.filename)
            if not filename or not _is_image(filename):
                return jsonify({
                    "error": "Unsupported file type. Upload a PNG/JPG image."
                }), 400
            file_path = os.path.join(CHAT_UPLOAD_FOLDER, filename)
            file.save(file_path)
            image_path = file_path
            image_url = f"http://localhost:5000/static/chat/{filename}"

        # Guard: nothing to respond to
        if not message and not has_file:
            return jsonify({
                "response": "Please ask a question or upload an image."
            })

        # ── Chat logging ────────────────────────────────────────────────────
        logging.info("=" * 60)
        logging.info("NEW CHAT REQUEST")
        logging.info(f"USER MESSAGE  : {message if message else '(no text)'}")
        if has_file:
            logging.info(f"IMAGE UPLOADED: {filename}  |  saved to: {file_path}")
        else:
            logging.info("IMAGE UPLOADED: None")
        # ────────────────────────────────────────────────────────────────────

        # Reset per-request RAG state so faithfulness only fires when this
        # request actually pulled chunks (not stale from a previous /chat).
        reset_last_rag_chunks()

        # Run the agent
        response_text = brain.run(
            user_message=message,
            chat_history=chat_history,
            image_path=image_path
        )

        # ── Log the bot response ─────────────────────────────────────────────
        logging.info(f"BOT RESPONSE  : {response_text}")
        logging.info("=" * 60)

        # ── Live faithfulness (async, only when RAG was used) ────────────────
        rag_chunks = get_last_rag_chunks()
        if rag_chunks and message:
            threading.Thread(
                target=_judge_async,
                args=(message, rag_chunks, response_text),
                daemon=True,
            ).start()
        # ─────────────────────────────────────────────────────────────────────
        # ─────────────────────────────────────────────────────────────────────

        result = {"response": response_text}
        if image_url:
            result["image"] = image_url

        return jsonify(result)

    except Exception as e:
        logging.error(f"CHAT ERROR: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)

