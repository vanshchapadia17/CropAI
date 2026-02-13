# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CropAI is a crop image classification system using EfficientNetB0 transfer learning. It classifies images into 5 crop types (Jute, Maize, Rice, Sugarcane, Wheat) via a Flask REST API backend and React frontend. A disease detection pipeline exists as stubs but is not yet implemented.

## Commands

### Backend (Python/Flask)
```bash
pip install -e .                              # Install package with dependencies
python app.py                                 # Start Flask dev server (localhost:5000, debug=True)
python src/pipeline/train_pipeline.py         # Run full training pipeline
```

### Frontend (React)
```bash
cd frontend
npm install                                   # Install dependencies
npm start                                     # Dev server on localhost:3000
npm run build                                 # Production build
npm test                                      # Run tests
```

## Architecture

### ML Pipeline (src/)

The training pipeline follows a sequential component pattern:

**DataIngestion** → **DataTransformation** → **ModelTrainer**

- `src/components/data_ingestion.py`: Reads CSV with image paths/labels, performs stratified 80/20 train/test split → outputs `data/train.csv`, `data/test.csv`
- `src/components/data_transformation.py`: Loads images as numpy arrays, resizes to 224x224, applies EfficientNet normalization, one-hot encodes labels → saves `artifacts/preprocessor.pkl` (dict with `img_size` and `num_classes`)
- `src/components/model_trainer.py`: Two-stage transfer learning on EfficientNetB0 — first 5 epochs with frozen base (lr=1e-4), then 10 epochs with last 50 layers unfrozen (lr=5e-6). Uses class weights (Rice: 1.8, Sugarcane: 1.2) for imbalanced data. Includes data augmentation (flip, rotation, zoom, brightness). → saves `artifacts/model.h5`

**Prediction pipeline** (`src/pipeline/predict_pipeline.py`): Loads model.h5 and preprocessor.pkl, reads image via OpenCV (BGR→RGB), resizes to 224x224, applies EfficientNet preprocessing, returns class index + confidence.

### Flask API (app.py)

Single endpoint: `POST /predict` — accepts multipart file upload, runs PredictPipeline, returns `{crop, confidence, image}` JSON. Uploaded files saved to `static/uploads/`. CORS enabled for React frontend.

### React Frontend (frontend/)

Single-page app in `frontend/src/App.js`. File upload form POSTs to `http://localhost:5000/predict`, displays crop name, confidence %, and image preview. Cyberpunk-themed UI.

### Cross-Cutting Concerns

- `src/logger.py`: Timestamped log files in `logs/` directory
- `src/exception.py`: CustomException with file/line context
- `src/utils.py`: `save_object()`/`load_object()` using dill serialization

### Artifacts

- `artifacts/model.h5` — trained EfficientNetB0 model
- `artifacts/preprocessor.pkl` — preprocessing config (image_size: 224, num_classes: 5)
- `artifacts/metrics.txt` — classification report from training

## Key Technical Details

- All images must be 224x224 for EfficientNetB0 compatibility
- Serialization uses **dill** (not pickle) for object persistence
- Crop index mapping: 0=Jute, 1=Maize, 2=Rice, 3=Sugarcane, 4=Wheat
- Disease pipeline files exist as empty stubs in `src/components/disease_*.py` and `src/pipeline/disease_*_pipeline.py`
