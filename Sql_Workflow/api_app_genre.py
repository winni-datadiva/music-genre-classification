"""
Minimal FastAPI service for the GTZAN genre MLP.

Run locally with:
    pip install fastapi uvicorn python-multipart librosa torch --break-system-packages
    uvicorn api_app_genre:app --reload

Then POST an audio file to /predict, e.g.:
    curl -X POST "http://127.0.0.1:8000/predict" -F "file=@some_clip.wav"

Expects genre_mlp_model.pt (saved by MLP_Training.ipynb) in the same folder.
"""

import io

import librosa
import numpy as np
import torch
import torch.nn as nn
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel

# ── Config ────────────────────────────────────────────────────────────────
MODEL_PATH = "genre_mlp_model.pt"
DEVICE = torch.device("cpu")
N_MFCC = 13  # must match clean_preprocess_dataset.ipynb


# ── Model definition (must match MLP_Training.ipynb exactly) ─────────────
class GenreMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, num_classes, dropout=0.3):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev_dim, h), nn.ReLU(), nn.Dropout(dropout)]
            prev_dim = h
        layers.append(nn.Linear(prev_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

model = GenreMLP(
    input_dim=checkpoint["input_dim"],
    hidden_dims=checkpoint["hidden_dims"],
    num_classes=checkpoint["num_classes"],
)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

FEATURE_COLS = checkpoint["feature_cols"]
FEATURE_MEAN = np.array(checkpoint["feature_mean"], dtype=np.float32)
FEATURE_STD = np.array(checkpoint["feature_std"], dtype=np.float32)
LABEL_MAP = checkpoint["label_map"]
INV_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}

SAMPLE_RATE = checkpoint["sample_rate"]
DURATION_SEC = checkpoint["duration_sec"]
TARGET_SAMPLES = SAMPLE_RATE * DURATION_SEC


def extract_features(y, sr):
    """Same 18-feature extraction as clean_preprocess_dataset.ipynb. Order matters."""
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    tempo_val = float(np.atleast_1d(tempo)[0])

    chroma_mean = float(np.mean(librosa.feature.chroma_stft(y=y, sr=sr)))

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    mfcc_means = np.mean(mfcc, axis=1)

    spectral_centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
    spectral_rolloff = float(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)))
    zcr = float(np.mean(librosa.feature.zero_crossing_rate(y)))

    features = {
        "tempo": tempo_val,
        "chroma_mean": chroma_mean,
        **{f"mfcc{i + 1}_mean": mfcc_means[i] for i in range(N_MFCC)},
        "spectral_centroid": spectral_centroid,
        "spectral_rolloff": spectral_rolloff,
        "zero_crossing_rate": zcr,
    }
    return np.array([features[col] for col in FEATURE_COLS], dtype=np.float32)


def preprocess_audio(audio_bytes: bytes) -> torch.Tensor:
    y, sr = librosa.load(io.BytesIO(audio_bytes), sr=SAMPLE_RATE, mono=True)

    if len(y) > TARGET_SAMPLES:
        y = y[:TARGET_SAMPLES]
    elif len(y) < TARGET_SAMPLES:
        y = np.pad(y, (0, TARGET_SAMPLES - len(y)), mode="constant")

    raw_features = extract_features(y, sr)
    normalized = (raw_features - FEATURE_MEAN) / FEATURE_STD

    return torch.tensor(normalized, dtype=torch.float32).unsqueeze(0)


# ── API ───────────────────────────────────────────────────────────────────
class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    probabilities: dict[str, float]


app = FastAPI(title="GTZAN Genre Classifier API")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    audio_bytes = await file.read()
    feature_tensor = preprocess_audio(audio_bytes)

    with torch.no_grad():
        logits = model(feature_tensor)
        probabilities = torch.softmax(logits, dim=1)
        confidence, predicted_idx = torch.max(probabilities, dim=1)

    prediction = INV_LABEL_MAP[predicted_idx.item()]
    probs_by_genre = {
        INV_LABEL_MAP[i]: round(probabilities[0, i].item(), 4)
        for i in range(len(LABEL_MAP))
    }

    return PredictionResponse(
        prediction=prediction,
        confidence=round(confidence.item(), 4),
        probabilities=probs_by_genre,
    )
