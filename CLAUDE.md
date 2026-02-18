# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CropAI is a multi-agent AI system for agriculture. A **Brain Agent** (LLM-powered via Groq/LangChain) acts as the central orchestrator — it receives user text + image through a chatbot interface, detects intent, and routes to the appropriate classifier:
- **Crop Classifier**: EfficientNetB0 classifying 5 crop types (Jute, Maize, Rice, Sugarcane, Wheat)
- **Disease Detector**: EfficientNetB0 classifying 9 disease types across Rice and Sugarcane

LangSmith tracing is enabled for all LLM calls.

## Commands

### Backend (Python/Flask)
```bash
pip install -e .                                        # Install package with dependencies
python app.py                                           # Start Flask dev server (localhost:5000)
python src/pipeline/train_pipeline.py                   # Train crop classifier
python -m src.pipeline.disease_train_pipeline            # Train disease classifier
```

### Frontend (React)
```bash
cd frontend
npm install                                   # Install dependencies
npm start                                     # Dev server on localhost:3000
npm run build                                 # Production build
npm test                                      # Run tests
```

### Environment Variables (.env)
```
GROQ_API_KEY=...              # Groq API key for Brain Agent LLM
LANGCHAIN_API_KEY=...         # LangSmith API key for tracing
```

## Architecture

### Multi-Agent Flow

```
User (text + image) → POST /chat → Brain Agent (intent detection via Groq LLM)
                                        ├── intent=crop    → PredictPipeline → crop result
                                        ├── intent=disease → DiseasePredictPipeline → disease result
                                        └── intent=general → LLM answers directly
                                    Brain Agent formats result → conversational response → React chatbot UI
```

### Brain Agent (`src/agents/brain_agent.py`)

Uses `ChatGroq` (model: `llama-3.1-8b-instant`) via LangChain. Three methods:
- `detect_intent()`: Classifies user message as "crop", "disease", or "general" — returns structured JSON
- `format_response()`: Takes classifier output and generates a natural language response
- `handle_general_query()`: Answers agriculture questions when no image is provided

### Crop ML Pipeline (src/components/)

**DataIngestion** → **DataTransformation** → **ModelTrainer**

- `data_ingestion.py`: Reads CSV with image paths/labels, stratified 80/20 split → `data/train.csv`, `data/test.csv`
- `data_transformation.py`: Loads images, resizes to 224x224, EfficientNet normalization, one-hot encodes → saves `artifacts/preprocessor.pkl`
- `model_trainer.py`: Two-stage EfficientNetB0 transfer learning — 5 epochs frozen (lr=1e-4), then 10 epochs fine-tuning last 50 layers (lr=5e-6). Class weights for Rice (1.8) and Sugarcane (1.2). → saves `artifacts/model.h5`

### Disease ML Pipeline (src/components/disease_*)

Same sequential pattern as crop pipeline:

- `disease_ingestion.py`: Scans `data/RD/` (Rice diseases) and `data/SD/` (Sugarcane diseases) folder structure, builds CSV with path + label → `data/train2.csv`, `data/test2.csv`
- `disease_transformation.py`: Same image preprocessing as crop → saves `artifacts/disease_preprocessor.pkl`
- `disease_trainer.py`: Same EfficientNetB0 architecture, 9-class output. Higher class weights (~2.5x) for sugarcane diseases (~500 imgs) vs rice diseases (~1500 imgs). → saves `artifacts/disease_model.h5`

### Flask API (app.py)

- `POST /chat` — Main chatbot endpoint. Accepts `message` (text) + `file` (image). Brain Agent detects intent and routes to classifier. Returns `{response, intent, crop/disease, confidence, image}`
- `POST /predict` — Direct crop classification. Returns `{crop, confidence, image}`
- `POST /predict-disease` — Direct disease detection. Returns `{disease, confidence, image}`

Environment loaded from `.env` via `python-dotenv`. LangSmith tracing auto-enabled.

### React Frontend (frontend/src/App.js)

Chatbot-style single-page app. Text input + image upload → sends to `/chat` endpoint. Displays conversation with message bubbles, classification badges (label + confidence), and image previews. Cyberpunk-themed UI.

### Cross-Cutting Concerns

- `src/logger.py`: Timestamped log files in `logs/`
- `src/exception.py`: CustomException with file/line context
- `src/utils.py`: `save_object()`/`load_object()` using dill serialization

### Artifacts

- `artifacts/model.h5` — trained crop classifier (EfficientNetB0)
- `artifacts/preprocessor.pkl` — crop preprocessing config (img_size: 224, num_classes: 5)
- `artifacts/disease_model.h5` — trained disease classifier (EfficientNetB0)
- `artifacts/disease_preprocessor.pkl` — disease preprocessing config (img_size: 224, num_classes: 9)
- `artifacts/metrics.txt` / `artifacts/disease_metrics.txt` — classification reports

## Key Technical Details

- All images must be 224x224 for EfficientNetB0 compatibility
- Models must be loaded with `compile=False` due to TensorFlow 2.12 serialization bug with `CategoricalCrossentropy(label_smoothing=...)`
- Serialization uses **dill** (not pickle) for object persistence
- Crop index mapping: 0=Jute, 1=Maize, 2=Rice, 3=Sugarcane, 4=Wheat
- Disease index mapping (alphabetical): 0=Bacterial Blight, 1=Blast, 2=Brown Spot, 3=Healthy, 4=Mosaic, 5=Red Rot, 6=Rust, 7=Tungro, 8=Yellow
- Disease data is folder-based (`data/RD/*/`, `data/SD/*/`) — not CSV-based like crop data
- Conda environment at `virtualEN/` (Python 3.8 + TensorFlow 2.12)
