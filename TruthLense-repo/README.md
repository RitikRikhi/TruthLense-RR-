# TruthLense

TruthLense is an AI-powered platform designed to detect, analyze, and explain misleading information circulating across digital platforms. In today’s digital age, misinformation spreads rapidly through social media, news platforms, and messaging apps, making it increasingly difficult for users to distinguish between reliable and misleading content.

This repository contains:

- `backend/` – Node.js Express API (`/api/analyze`) that orchestrates Gemini, NewsAPI, Wikipedia and a Python ML/RAG service.
- `ml/` – Python utilities to train and inspect a tabular ML model and expose it via FastAPI.
- `frontend/` – A minimal single-page UI that talks to the `/api/analyze` endpoint.

---

## 1. Backend (Node.js) – `/api/analyze`

### Install & run

```bash
cd backend
npm install
```

Create a `.env` file in `backend/` (do **not** commit it):

```env
PORT=5000
MONGO_URI=your_mongodb_connection_string   # optional (backend can run without DB)
GEMINI_API_KEY=your_gemini_key             # required for best analysis
NEWS_API_KEY=your_newsapi_key              # optional but recommended
PYTHON_API_URL=http://localhost:8000       # URL of the Python ML/RAG service
```

Then start the backend:

```bash
cd backend
npm start
```

This will:

- Expose the JSON API at `http://localhost:5000/api/analyze`.
- Serve the frontend from `http://localhost:5000/`.

---

## 2. Python ML utilities (`ml/`)

The `ml` folder contains a generic tabular ML pipeline (RandomForest + preprocessing) and a small FastAPI server if you want to expose it separately.

### Install Python dependencies

```bash
cd ml
python -m venv .venv
.venv\Scripts\activate  # on Windows
pip install -r requirements.txt
```

### Train a model

```bash
python train_model.py --data-path path/to/your.csv --target-col TARGET_COLUMN_NAME
```

This will create:

- `ml/models/truthlense_model.joblib` – trained model.
- `ml/models/truthlense_model_metadata.json` – metadata (feature names, metrics, etc.).

### (Optional) Inspect the model

```bash
python inspect_model.py --n-samples 3
```

This samples random rows from the dataset, runs them through the model, and prints predictions + metrics.

### (Optional) Python FastAPI server

The `backend/api.py` file (Python) exposes:

- `GET /` – health & model status
- `GET /model-info` – training metadata
- `POST /predict` – tabular prediction

Run it with:

```bash
uvicorn backend.api:app --reload
```

(Make sure the model is already trained and the paths in the metadata are valid.)

---

## 3. Frontend

The frontend is a lightweight single-page UI served by the Node backend.

- Location: `frontend/index.html`, `frontend/style.css`, `frontend/app.js`
- Served automatically when you run `npm start` in `backend/`
- Main interaction:
  - Textarea where you paste a claim
  - Calls `POST /api/analyze` and renders:
    - Credibility score
    - Verdict
    - Explanations
    - Sources with credibility scores

After cloning the repo, the minimal setup to see everything working is:

1. Set up Node backend (`backend/`) as described above.
2. (Optional but recommended) Configure `.env` with your API keys.
3. Visit `http://localhost:5000/` in your browser and start testing claims.

