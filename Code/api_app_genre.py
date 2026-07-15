"""
Rhythmx Music Genre Classification — FastAPI Demo (CNN version)

Run with:
    uvicorn api_app_genre:app --reload

Note: run this from the same directory as model.py and best_genre_cnn.pt
(matches the "wrong directory" uvicorn issue you hit before — cd into this
folder first, don't launch from a parent dir).
"""

import io
import json
from contextlib import asynccontextmanager
from pathlib import Path

import librosa
import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import RedirectResponse
from pydantic import BaseModel


from model import GenreCNN

N_MFCC = 13
SAMPLE_RATE = 22050
TARGET_FRAMES = 130
CHECKPOINT_PATH = Path("best_genre_cnn.pt")

# TRAIN_MEAN = -0.4816
# TRAIN_STD = 65.9672


# Load normalization stats computed during CNN Training (same source of truth
# as streamlit_app_genre.py, instead of a second hardcoded copy of the numbers)
STATS_PATH = Path("train_norm_stats.json")
 
if not STATS_PATH.exists():
    raise RuntimeError(f"Normalization stats not found at {STATS_PATH}. Run CNN_Training.ipynb first.")
 
with open(STATS_PATH) as f:
    stats = json.load(f)
 
TRAIN_MEAN = stats["train_mean"]
TRAIN_STD = stats["train_std"]
print(f"Loaded normalization stats -> mean: {TRAIN_MEAN:.4f}, std: {TRAIN_STD:.4f}")
 
_model = None
_idx_to_genre = None
 
 
def load_model():
    global _model, _idx_to_genre
    if not CHECKPOINT_PATH.exists():
        raise RuntimeError(f"Checkpoint not found at {CHECKPOINT_PATH}. Run CNN_Training.ipynb first.")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    _model = GenreCNN(num_classes=checkpoint["num_classes"])
    _model.load_state_dict(checkpoint["model_state_dict"])
    _model.eval()
    _idx_to_genre = {v: k for k, v in checkpoint["genre_to_idx"].items()}
 
 
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    load_model()
    yield
    # (no shutdown steps needed)
 
 
app = FastAPI(title="Rhythmx Genre Classifier API", lifespan=lifespan)
 
 
@app.get("/")
def root():
    return RedirectResponse(url="/docs")
 
 
class GenrePrediction(BaseModel):
    genre: str
    confidence: float
 
 
class PredictionResponse(BaseModel):
    top_genre: str
    top_confidence: float
    all_predictions: list[GenrePrediction]
 
 
def extract_mfcc_array(y, sr, n_mfcc=N_MFCC, target_frames=TARGET_FRAMES):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    if mfcc.shape[1] < target_frames:
        mfcc = np.pad(mfcc, ((0, 0), (0, target_frames - mfcc.shape[1])), mode="constant")
    else:
        mfcc = mfcc[:, :target_frames]
    return mfcc.astype(np.float32)
 
 
@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _model is not None}
 
 
@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if not file.content_type == "audio/wav" and not file.filename.lower().endswith(".wav"):
        raise HTTPException(status_code=400, detail="Only .wav files are supported")
 
    raw_bytes = await file.read()
    try:
        # res_type='soxr_hq' pinned to match extraction in CNN_Training.ipynb —
        # keeps inference-time MFCCs consistent with what the model was trained on.
        y, sr = librosa.load(io.BytesIO(raw_bytes), sr=SAMPLE_RATE, mono=True, res_type="soxr_hq")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not decode audio: {e}")
 
    clip_len = 3 * sr
    if len(y) < clip_len:
        y = np.pad(y, (0, clip_len - len(y)))
    else:
        y = y[:clip_len]
 
    mfcc = extract_mfcc_array(y, sr)
    mfcc = (mfcc - TRAIN_MEAN) / (TRAIN_STD + 1e-8)
    x = torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # (1, 1, 13, 130)
 
    with torch.no_grad():
        logits = _model(x)
        probs = torch.softmax(logits, dim=1).squeeze(0).numpy()
 
    ranked = sorted(
        [(_idx_to_genre[i], float(p)) for i, p in enumerate(probs)],
        key=lambda t: t[1],
        reverse=True,
    )
 
    return PredictionResponse(
        top_genre=ranked[0][0],
        top_confidence=ranked[0][1],
        all_predictions=[GenrePrediction(genre=g, confidence=p) for g, p in ranked],
    )
 