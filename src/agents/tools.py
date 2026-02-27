from langchain_core.tools import tool
from src.pipeline.predict_pipeline import PredictPipeline
from src.pipeline.disease_predict_pipeline import DiseasePredictPipeline
from src.logger import logging
from src.RAG.rag_pipeline import RAGPipeline

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

# Shared state — set by brain_agent before each executor.invoke()
current_image_path = None
last_identified_crop = None


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
    if current_image_path is None:
        return "Error: No image uploaded. Please ask the user to upload an image."

    import src.agents.tools as _self
    logging.info(f"analyze_image tool called with image: {current_image_path}")

    # --- Crop classifier ---
    try:
        crop_pipeline = PredictPipeline()
        crop_idx, crop_conf = crop_pipeline.predict(current_image_path)
        crop_name = CROP_MAP[crop_idx]
        crop_result = f"Crop Classifier  → {crop_name} ({round(crop_conf, 2)}% confidence)"
        _self.last_identified_crop = crop_name
    except Exception as e:
        crop_result = f"Crop Classifier  → Error: {e}"
        crop_name = None

    # --- Disease classifier ---
    try:
        disease_pipeline = DiseasePredictPipeline()
        disease_idx, disease_conf = disease_pipeline.predict(current_image_path)
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
        pipeline = RAGPipeline.get_instance()
        results = pipeline.retrieve(question, k=2)   # 2 chunks keeps context tight

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
