import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routes import router

# The FastAPI app instance
app = FastAPI(
    title="TruthLens Backend API",
    description="An AI-powered service for detecting misinformation and evaluating credibility using a fine-tuned DistilBERT transformer.",
    version="1.0.0"
)

# Middleware configuration
# We allow CORS so the frontend web app can communicate with this backend easily.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Since it's a hackathon, allow all origins for easy testing
    allow_credentials=True,
    allow_methods=["*"], # Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"], # Allow all headers
)

# Attach routes from routes.py
app.include_router(router)
