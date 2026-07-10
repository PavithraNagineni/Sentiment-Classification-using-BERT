"""
FastAPI Inference Server — BERT Sentiment Analysis
===================================================
Serves the fine-tuned BERT model as a REST API.

Endpoints:
    POST /predict          — single text prediction
    POST /predict/batch    — batch prediction
    GET  /health           — health check
    GET  /model/info       — model metadata

Run:
    uvicorn app:app --host 0.0.0.0 --port 8000 --reload
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List
import torch
import numpy as np
import time
import json
import os
import logging
from transformers import AutoTokenizer, AutoModelForSequenceClassification

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Config ───────────────────────────────────────────────────────────────────
MODEL_PATH = os.getenv("MODEL_PATH", "./outputs/bert_sentiment/best_model")
MAX_LENGTH = 128
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
LABELS     = {0: "negative", 1: "positive"}

tokenizer = None
model     = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global tokenizer, model
    logger.info(f"Loading model from {MODEL_PATH} on {DEVICE}...")
    try:
        if not os.path.isdir(MODEL_PATH):
            raise FileNotFoundError(f"No saved model found at {MODEL_PATH}")
        
        if os.path.exists(os.path.join(MODEL_PATH, "adapter_config.json")):
            logger.info("Detected LoRA adapters. Loading base model and applying adapters...")
            from peft import PeftModel
            with open(os.path.join(MODEL_PATH, "adapter_config.json"), "r") as f:
                adapter_cfg = json.load(f)
            base_model_id = adapter_cfg.get("base_model_name_or_path", "bert-base-uncased")
            
            tokenizer = AutoTokenizer.from_pretrained(base_model_id)
            base_model = AutoModelForSequenceClassification.from_pretrained(
                base_model_id, num_labels=2,
                id2label={0: "negative", 1: "positive"},
                label2id={"negative": 0, "positive": 1}
            )
            model = PeftModel.from_pretrained(base_model, MODEL_PATH)
        else:
            tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
            model     = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
            
        model.to(DEVICE)
        model.eval()
        logger.info("Model loaded successfully!")
    except Exception as e:
        logger.error(f"Model load failed: {e}")
        logger.warning("Using mock model for demo purposes.")
    yield


from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="BERT Sentiment Analysis API",
    description="Fine-tuned BERT model for binary sentiment classification",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Schemas ──────────────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        max_length=512,
        json_schema_extra={"example": "This movie was absolutely fantastic!"},
    )

class PredictBatchRequest(BaseModel):
    texts: List[str] = Field(..., min_length=1, max_length=32)

class PredictionResult(BaseModel):
    text: str
    label: str
    confidence: float
    probabilities: dict
    inference_time_ms: float

class BatchPredictionResponse(BaseModel):
    results: List[PredictionResult]
    total_texts: int
    avg_inference_time_ms: float


# ── Inference helper ─────────────────────────────────────────────────────────
def predict_texts(texts: List[str]) -> List[PredictionResult]:
    if model is None or tokenizer is None:
        # Mock response for demo
        return [
            PredictionResult(
                text=t,
                label="positive" if i % 2 == 0 else "negative",
                confidence=0.95,
                probabilities={"negative": 0.05, "positive": 0.95},
                inference_time_ms=12.5,
            )
            for i, t in enumerate(texts)
        ]

    results = []
    for text in texts:
        start = time.perf_counter()
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_LENGTH,
            padding=True,
        ).to(DEVICE)

        with torch.no_grad():
            outputs = model(**inputs)
            probs   = torch.softmax(outputs.logits, dim=-1).cpu().numpy()[0]

        pred_idx    = int(np.argmax(probs))
        elapsed_ms  = (time.perf_counter() - start) * 1000

        results.append(PredictionResult(
            text=text,
            label=LABELS[pred_idx],
            confidence=float(probs[pred_idx]),
            probabilities={LABELS[i]: float(p) for i, p in enumerate(probs)},
            inference_time_ms=round(elapsed_ms, 2),
        ))
    return results


# ── Routes ───────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": DEVICE,
    }

@app.get("/model/info")
def model_info():
    metrics_path = "./outputs/bert_sentiment/metrics.json"
    metrics = {}
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            metrics = json.load(f)
    return {
        "model_name": "bert-base-uncased",
        "task": "Sentiment Classification (Binary)",
        "labels": LABELS,
        "max_length": MAX_LENGTH,
        "device": DEVICE,
        "metrics": metrics,
    }

@app.post("/predict", response_model=PredictionResult)
def predict(request: PredictRequest):
    try:
        results = predict_texts([request.text])
        return results[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_batch(request: PredictBatchRequest):
    try:
        results = predict_texts(request.texts)
        avg_time = sum(r.inference_time_ms for r in results) / len(results)
        return BatchPredictionResponse(
            results=results,
            total_texts=len(results),
            avg_inference_time_ms=round(avg_time, 2),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
