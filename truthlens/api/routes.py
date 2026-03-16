import os
import uuid
import datetime
from fastapi import APIRouter
from pydantic import BaseModel
from typing import List, Optional
from dotenv import load_dotenv

# Import our custom logic
from model.inference import get_inference
from model.train_model import run_training_async
from utils.rag_engine import get_rag_engine
from model.run_rag_ingestion import ingest_from_csv

# Load environment variables from .env file
load_dotenv()

MODEL_DIR = os.getenv("MODEL_PATH", "./model/saved_model")
DATASET_DIR = os.getenv("DATASET_PATH", "./data/training_data.csv")

router = APIRouter()

# -----------------
# Mock Database (For Hackathon Demo)
# -----------------
# We store past queries in memory to power the analytics and history endpoints.
mock_db_history = []

# -----------------
# Pydantic Schemas
# -----------------

class AnalyzeRequest(BaseModel):
    text: str
    
class AnalyzeResponse(BaseModel):
    id: str
    timestamp: str
    credibility_score: int
    prediction: str
    explanation: List[str]

class RAGAnalyzeResponse(BaseModel):
    id: str
    timestamp: str
    rag_score: int
    classification_prediction: str
    combined_explanation: List[str]
    retrieved_facts: List[str]
    
class HistoryResponse(BaseModel):
    total_queries: int
    history: List[AnalyzeResponse]

class AnalyticsResponse(BaseModel):
    total_analyzed: int
    total_misleading: int
    total_credible: int
    average_credibility_score: float

class SystemStatusResponse(BaseModel):
    status: str
    model_loaded: bool
    version: str
    uptime: str

# -----------------
# Endpoints
# -----------------

@router.get("/health", tags=["System"])
def health_check():
    """Simple endpoint to confirm the server is running."""
    return {"status": "TruthLens API running", "timestamp": datetime.datetime.now().isoformat()}

@router.get("/system/status", response_model=SystemStatusResponse, tags=["System"])
def system_status():
    """Returns detailed information about the system and model memory state."""
    inference_engine = get_inference(MODEL_DIR)
    
    return {
        "status": "Online",
        "model_loaded": inference_engine.is_loaded,
        "version": "1.0.0",
        "uptime": "Calculating based on server start"
    }

@router.post("/analyze", response_model=AnalyzeResponse, tags=["AI Analysis"])
def analyze_text(request: AnalyzeRequest):
    """
    Core AI endpoint. Analyzes text for misinformation using DistilBERT.
    Saves the result in the mock database for history tracking.
    """
    inference_engine = get_inference(MODEL_DIR)
    
    # Generate unique ID and timestamp for tracking
    query_id = str(uuid.uuid4())
    timestamp = datetime.datetime.now().isoformat()
    
    if inference_engine.is_loaded:
        # Run PyTorch pipeline
        result = inference_engine.predict(request.text)
        
        response = {
            "id": query_id,
            "timestamp": timestamp,
            **result
        }
    else:
        # Dynamic Hackathon fallback using 100% Free APIs!
        from utils.explanation import get_dynamic_fallback_score
        
        # Call the new dynamic engine
        credibility_score, prediction, dynamic_explanations = get_dynamic_fallback_score(request.text)
        
        response = {
            "id": query_id,
            "timestamp": timestamp,
            "credibility_score": credibility_score,
            "prediction": prediction,
            "explanation": dynamic_explanations
        }
            
    # Save to mock database
    mock_db_history.append(response)
    return response

@router.get("/history", response_model=HistoryResponse, tags=["Data Management"])
def get_history(limit: int = 10):
    """Retrieves the most recent queries made to the API."""
    # Return the most recent transactions up to the limit
    recent_history = list(reversed(mock_db_history))[:limit]
    return {
        "total_queries": len(mock_db_history),
        "history": recent_history
    }

@router.delete("/history", tags=["Data Management"])
def clear_history():
    """Clears all stored query history."""
    mock_db_history.clear()
    return {"status": "History cleared"}

@router.get("/analytics", response_model=AnalyticsResponse, tags=["Data Management"])
def get_analytics():
    """Returns aggregrated statistics on what the AI has processed."""
    total = len(mock_db_history)
    if total == 0:
         return {
            "total_analyzed": 0,
            "total_misleading": 0,
            "total_credible": 0,
            "average_credibility_score": 0.0
         }
         
    total_misleading = sum(1 for item in mock_db_history if item["prediction"] == "Potentially Misleading")
    total_credible = sum(1 for item in mock_db_history if item["prediction"] == "Credible")
    avg_score = float(sum(item["credibility_score"] for item in mock_db_history)) / total
    
    return {
        "total_analyzed": total,
        "total_misleading": total_misleading,
        "total_credible": total_credible,
        "average_credibility_score": round(avg_score, 2)
    }

@router.post("/train", tags=["AI Training"])
def train_model():
    """Triggers the async machine learning training pipeline."""
    if not os.path.exists(DATASET_DIR):
         return {"status": f"Training failed to start: Dataset not found at {DATASET_DIR}. Please place your CSV there."}
         
    run_training_async(data_path=DATASET_DIR, model_save_path=MODEL_DIR)
    
    return {"status": "Training started in background"}

# -----------------
# RAG Endpoints
# -----------------

@router.post("/rag/ingest", tags=["RAG Attention"])
def rag_ingest(file_path: str = "./data/rag_test_facts.csv"):
    """
    Ingests a CSV of known facts into the ChromaDB vector store.
    Provides the ground-truth knowledge base for RAG attention.
    """
    try:
        total_added = ingest_from_csv(file_path)
        return {"status": "Success", "facts_ingested": total_added}
    except Exception as e:
        return {"status": "Error", "message": str(e)}

@router.get("/rag/stats", tags=["RAG Attention"])
def rag_stats():
    """Returns the total number of verified facts in the Vector Database."""
    engine = get_rag_engine()
    count = engine.fact_collection.count()
    return {"status": "Online", "total_facts_stored": count}

@router.post("/rag/analyze", response_model=RAGAnalyzeResponse, tags=["RAG Attention"])
def rag_analyze(request: AnalyzeRequest):
    """
    Hybrid RAG Analysis.
    Combines classic classification (DistilBERT/SHAP) with Vector Retrieval (ChromaDB)
    and Generative AI (Gemini) to cross-reference statements against known facts.
    """
    engine = get_rag_engine()
    inference_engine = get_inference(MODEL_DIR)
    
    # Generate unique ID and timestamp 
    query_id = str(uuid.uuid4())
    timestamp = datetime.datetime.now().isoformat()
    
    # 1. RAG Vector Retrieval & Generative Verification
    rag_result = engine.query_rag(request.text)
    
    # 2. Classic Model Classification (if loaded)
    classification_pred = "Unknown"
    if inference_engine.is_loaded:
        clf_result = inference_engine.predict(request.text)
        classification_pred = clf_result["prediction"]
        
    response = {
        "id": query_id,
        "timestamp": timestamp,
        "rag_score": rag_result["rag_score"],
        "classification_prediction": classification_pred,
        "combined_explanation": [rag_result["rag_explanation"]],
        "retrieved_facts": rag_result["retrieved_context"]
    }
    
    return response
