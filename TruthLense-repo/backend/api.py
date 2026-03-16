"""
TruthLense API — Tabular ML Backend (Python)
===========================================

FastAPI server that exposes the trained TruthLense tabular ML model
from `ml/train_model.py` (RandomForest + preprocessing pipeline).

Usage (from repo root):
  uvicorn backend.api:app --reload

Make sure you have first trained the model from the `ml/` folder.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from joblib import load
from pydantic import BaseModel, Field


ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_METADATA_PATH = ROOT_DIR / "ml" / "models" / "truthlense_model_metadata.json"


class PredictRequest(BaseModel):
    features: Dict[str, Any] = Field(
        ...,
        description=(
            "Key-value mapping of feature name to value for a single example. "
            "Must include the same feature names used during training."
        ),
    )


class PredictResponse(BaseModel):
    prediction: Any
    problem_type: str
    target_col: str
    feature_names: List[str]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_path: Optional[str]
    data_path: Optional[str]
    target_col: Optional[str]
    problem_type: Optional[str]


def load_model_and_metadata():
    if not DEFAULT_METADATA_PATH.exists():
        return None, None

    import json

    with DEFAULT_METADATA_PATH.open("r", encoding="utf-8") as f:
        metadata = json.load(f)

    model_path = Path(metadata["model_path"])
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found at {model_path}. "
            f"Run ml/train_model.py to train and save the model."
        )

    pipeline = load(model_path)
    return pipeline, metadata


app = FastAPI(
    title="TruthLense Tabular ML API (Python)",
    description=(
        "Python FastAPI backend exposing the TruthLense tabular ML model trained via ml/train_model.py.\n\n"
        "- **GET /** — Health & model status\n"
        "- **GET /model-info** — Training metadata & metrics\n"
        "- **POST /predict** — Run the model on one example\n"
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


PIPELINE, METADATA = load_model_and_metadata()


@app.get("/", response_model=HealthResponse, tags=["Health"])
def health_check() -> HealthResponse:
    """Quick health check + basic model status."""
    if METADATA is None or PIPELINE is None:
        return HealthResponse(
            status="Model not loaded. Train the model with ml/train_model.py first.",
            model_loaded=False,
            model_path=None,
            data_path=None,
            target_col=None,
            problem_type=None,
        )

    metrics = METADATA.get("metrics", {})
    problem_type = metrics.get("problem_type")

    return HealthResponse(
        status="TruthLense Python ML API is running.",
        model_loaded=True,
        model_path=str(METADATA.get("model_path")),
        data_path=str(METADATA.get("data_path")),
        target_col=str(METADATA.get("target_col")),
        problem_type=str(problem_type) if problem_type is not None else None,
    )


@app.get("/model-info", tags=["Model"])
def model_info():
    """Return full training metadata and evaluation metrics."""
    if METADATA is None or PIPELINE is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Train the model with ml/train_model.py first.",
        )

    return METADATA


@app.post("/predict", response_model=PredictResponse, tags=["Prediction"])
def predict(request: PredictRequest) -> PredictResponse:
    """
    Run the trained model on a single example.

    - Body: `{ \"features\": { \"feature1\": value1, \"feature2\": value2, ... } }`
    - Returns: predicted value/class and some metadata.
    """
    if METADATA is None or PIPELINE is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Train the model with ml/train_model.py first.",
        )

    feature_names: List[str] = METADATA.get("feature_names", [])
    if not feature_names:
        raise HTTPException(
            status_code=500,
            detail="feature_names missing in metadata. Retrain the model with the latest ml/train_model.py.",
        )

    row = {name: request.features.get(name) for name in feature_names}
    df = pd.DataFrame([row])

    try:
        y_pred = PIPELINE.predict(df)[0]
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed. Check input features and types. Error: {str(e)}",
        )

    metrics = METADATA.get("metrics", {})
    problem_type = metrics.get("problem_type", "unknown")

    return PredictResponse(
        prediction=y_pred,
        problem_type=problem_type,
        target_col=str(METADATA.get("target_col")),
        feature_names=feature_names,
    )

