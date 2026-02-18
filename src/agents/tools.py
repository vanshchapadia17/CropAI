from langchain_core.tools import tool
from src.pipeline.predict_pipeline import PredictPipeline
from src.pipeline.disease_predict_pipeline import DiseasePredictPipeline
from src.logger import logging

CROP_MAP = {0: "Jute", 1: "Maize", 2: "Rice", 3: "Sugarcane", 4: "Wheat"}
DISEASE_MAP = {
    0: "Bacterial Blight", 1: "Blast", 2: "Brown Spot", 3: "Healthy",
    4: "Mosaic", 5: "Red Rot", 6: "Rust", 7: "Tungro", 8: "Yellow"
}

# Shared state: the actual image path set by the /chat endpoint before agent runs
current_image_path = None

@tool
def classify_crop(query: str) -> str:
    """Classify a crop image to identify the crop type (Jute, Maize, Rice, Sugarcane, or Wheat).
    Use this when the user uploads a full crop or field image and wants to know what crop it is.
    Pass any string as query - the uploaded image is used automatically.
    Returns the crop name and confidence percentage."""
    if current_image_path is None:
        return "Error: No image has been uploaded. Please ask the user to upload a crop image."
    logging.info(f"classify_crop tool called with image: {current_image_path}")
    pipeline = PredictPipeline()
    crop_idx, confidence = pipeline.predict(current_image_path)
    crop_name = CROP_MAP[crop_idx]
    return f"Crop: {crop_name}, Confidence: {round(confidence, 2)}%"

@tool
def detect_disease(query: str) -> str:
    """Detect disease in a crop leaf image. Supports Rice and Sugarcane diseases only.
    Rice diseases: Bacterial Blight, Blast, Brown Spot, Tungro.
    Sugarcane diseases: Healthy, Mosaic, Red Rot, Rust, Yellow.
    Use this when the user uploads a leaf image and wants to check for disease.
    Pass any string as query - the uploaded image is used automatically.
    Returns the disease name and confidence percentage."""
    if current_image_path is None:
        return "Error: No image has been uploaded. Please ask the user to upload a leaf image."
    logging.info(f"detect_disease tool called with image: {current_image_path}")
    pipeline = DiseasePredictPipeline()
    disease_idx, confidence = pipeline.predict(current_image_path)
    disease_name = DISEASE_MAP[disease_idx]
    return f"Disease: {disease_name}, Confidence: {round(confidence, 2)}%"
