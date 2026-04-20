import contextvars
from typing import Optional

from langchain_core.tools import tool
from src.pipeline.predict_pipeline import PredictPipeline
from src.pipeline.disease_predict_pipeline import DiseasePredictPipeline
from src.logger import logging
from src.RAG.rerank import RerankRetriever

CROP_MAP = {0: "Jute", 1: "Maize", 2: "Rice", 3: "Sugarcane", 4: "Wheat"}
DISEASE_MAP = {
    0: "Bacterial Blight", 1: "Blast", 2: "Brown Spot", 3: "Healthy",
    4: "Mosaic", 5: "Red Rot", 6: "Rust", 7: "Tungro", 8: "Yellow"
}

# Which diseases belong to which crop (ground truth — never override)
DISEASE_CROP_MAP = {
    "Bacterial Blight": "Rice", "Blast": "Rice",
    "Brown Spot": "Rice",      "Tungro": "Rice",
    "Healthy": "Sugarcane",    "Mosaic": "Sugarcane",
    "Red Rot": "Sugarcane",    "Rust": "Sugarcane",
    "Yellow": "Sugarcane",
}

# Per-request state via ContextVar — safe across threads and concurrent requests.
# Why: module-level globals leaked state between users under multi-threaded Flask.
_current_image_path: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "cropai_current_image_path", default=None
)


def set_image_path(path: Optional[str]) -> None:
    _current_image_path.set(path)


@tool
def analyze_image(query: str) -> str:
    """Analyze an uploaded image by running BOTH the crop classifier and the disease classifier.

    Always use this tool whenever the user uploads any image — whether it is a full
    crop/field photo or a close-up leaf photo. Both models run together so the best
    answer can be determined.

    Pass any string as query — the uploaded image path is used automatically.

    Returns:
        Crop classifier result (crop name + confidence)
        Disease classifier result (disease name + confidence + which crop it belongs to)
    """
    image_path = _current_image_path.get()
    if image_path is None:
        return "Error: No image uploaded. Please ask the user to upload an image."

    logging.info(f"analyze_image tool called with image: {image_path}")

    # --- Crop classifier ---
    try:
        crop_pipeline = PredictPipeline()
        crop_idx, crop_conf = crop_pipeline.predict(image_path)
        crop_name = CROP_MAP[crop_idx]
        crop_result = f"Crop Classifier  → {crop_name} ({round(crop_conf, 2)}% confidence)"
    except Exception as e:
        crop_result = f"Crop Classifier  → Error: {e}"
        crop_name = None

    # --- Disease classifier ---
    try:
        disease_pipeline = DiseasePredictPipeline()
        disease_idx, disease_conf = disease_pipeline.predict(image_path)
        disease_name = DISEASE_MAP[disease_idx]
        disease_crop = DISEASE_CROP_MAP.get(disease_name, "Unknown")
        disease_result = (
            f"Disease Classifier → {disease_name} ({round(disease_conf, 2)}% confidence)"
            f" [This disease belongs to: {disease_crop}]"
        )
    except Exception as e:
        disease_result = f"Disease Classifier → Error: {e}"
        disease_name = None
        disease_crop = None

    return f"{crop_result}\n{disease_result}"


@tool
def answer_crop_question(question: str) -> str:
    """Answer farming questions about rice or sugarcane using a scraped knowledge base.

    Use this tool when the user asks about:
    - Cultivation practices (planting, land preparation, spacing, seedlings)
    - Pest and disease management (symptoms, treatments, chemicals)
    - Irrigation, nutrient management, fertilizers
    - Harvesting, post-harvest, marketing
    - Crop varieties, seasons, ratoon management
    - Any general rice or sugarcane farming knowledge

    Do NOT use this for image analysis — use analyze_image for uploaded images.
    Pass the user's exact question as the argument.
    Returns relevant passages from the knowledge base.
    """
    logging.info(f"answer_crop_question tool called with query: {question}")
    try:
        # Rerank retriever: hybrid (BM25 + dense + RRF) -> cross-encoder rerank.
        # Eval on 30 golden queries: Hit@5 100%, Hit@3 93%, MRR 0.80, ~630 ms.
        retriever = RerankRetriever.get_instance()
        results = retriever.retrieve(question, k=3)   # top-3 covers 93% of queries

        if not results:
            return "No relevant information found in the knowledge base for this question."

        context_parts = []
        for i, r in enumerate(results, 1):
            # Truncate each chunk to 400 chars to avoid bloating the prompt
            snippet = r["content"][:400].strip()
            context_parts.append(
                f"[{r['crop_type'].title()} | {r['source_file']}]\n{snippet}"
            )

        context = "\n\n---\n\n".join(context_parts)
        return f"Knowledge base:\n\n{context}"

    except Exception as e:
        logging.error(f"RAG tool error: {e}")
        return f"Error retrieving from knowledge base: {e}"
