"""
SYNAPS AI Inference Microservice
Provides standalone PyTorch SignalTransformer model inference over HTTP.
"""

import os
from pathlib import Path
from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
import torch
import numpy as np

import sys
_SERVICE_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SERVICE_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from ai.models.transformer import SignalTransformer
from ai.classification.confidence import calculate_confidence, confidence_percent
from ai.classification.modulation import classify_modulation, CLASS_NAMES
from ai.classification.unknown_detection import get_detection_status

app = FastAPI(
    title="SYNAPS AI Inference Service",
    description="Standalone microservice for SignalTransformer modulation classification.",
    version="1.0.0",
)

# Global model state
MODEL = None
DEVICE = None
INPUT_TOKENS = 256
INPUT_FEATURES = 5


def load_transformer_model():
    global MODEL, DEVICE
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Locate checkpoint
    model_paths = [
        _SERVICE_DIR / "models" / "transformer.pth",
        _PROJECT_ROOT / "ai" / "models" / "transformer.pth",
        _PROJECT_ROOT / "models" / "transformer" / "transformer.pth",
    ]

    target_path = None
    for p in model_paths:
        if p.exists():
            target_path = p
            break

    if target_path is None:
        raise FileNotFoundError(f"transformer.pth not found in any of {model_paths}")

    checkpoint = torch.load(target_path, map_location=device, weights_only=False)

    in_features = 5
    if isinstance(checkpoint, dict) and "input_features" in checkpoint:
        in_features = checkpoint["input_features"]
    elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        in_proj_w = checkpoint["model_state_dict"].get("input_projection.weight")
        if in_proj_w is not None:
            in_features = in_proj_w.shape[1]
    elif isinstance(checkpoint, dict) and "input_projection.weight" in checkpoint:
        in_proj_w = checkpoint.get("input_projection.weight")
        if in_proj_w is not None:
            in_features = in_proj_w.shape[1]

    model = SignalTransformer(
        input_features=in_features,
        num_classes=len(CLASS_NAMES),
    )

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()

    MODEL = model
    DEVICE = device
    print(f"[INFO] AI Service: Loaded Transformer from {target_path} on {device}")


@app.on_event("startup")
def startup_event():
    load_transformer_model()


class PredictRequest(BaseModel):
    features: List[List[float]] = Field(
        ...,
        description="Feature matrix of shape (256, 5): 256 tokens by 5 features.",
    )


class PredictResponse(BaseModel):
    predicted_class: str
    confidence: float
    status: str
    probabilities: Dict[str, float]


@app.get("/health")
def health_check():
    return {
        "status": "ONLINE",
        "service": "SYNAPS AI Inference Microservice",
        "model_loaded": MODEL is not None,
        "device": str(DEVICE) if DEVICE else "none",
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if MODEL is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="AI Transformer model is not loaded.",
        )

    features = np.array(req.features, dtype=np.float32)

    if features.ndim != 2 or features.shape[0] != INPUT_TOKENS or features.shape[1] not in (5, 6):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Expected feature shape (256, 5) or (256, 6), got {features.shape}",
        )

    required_dim = MODEL.input_features
    if features.shape[1] < required_dim:
        pad_cols = required_dim - features.shape[1]
        features = np.pad(features, ((0, 0), (0, pad_cols)), mode="constant")
    elif features.shape[1] > required_dim:
        features = features[:, :required_dim]

    x_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = MODEL(x_tensor)

    probs, pred_idx, conf = calculate_confidence(logits)
    pred_class = classify_modulation(pred_idx)
    conf_pct = confidence_percent(conf)
    det_status = get_detection_status(conf)

    prob_dict = {
        name: float(probs[i].item() * 100.0)
        for i, name in enumerate(CLASS_NAMES)
    }

    return PredictResponse(
        predicted_class=pred_class,
        confidence=conf_pct,
        status=det_status,
        probabilities=prob_dict,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("ai_service.main:app", host="0.0.0.0", port=8001, reload=False)
