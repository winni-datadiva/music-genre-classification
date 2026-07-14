"""
Rhythmx Music Genre Classification — Streamlit Demo (CNN version)

Run with:
    streamlit run streamlit_app_genre.py

Expects `best_genre_cnn.pt` (produced by CNN_Training.ipynb) and `model.py`
in the same directory.
"""

import json
from pathlib import Path

import librosa
import numpy as np
import streamlit as st
import torch

from model import GenreCNN

# --- Constants (must match preprocessing notebook) ---
N_MFCC = 13
SAMPLE_RATE = 22050
TARGET_FRAMES = 130
CHECKPOINT_PATH = Path("best_genre_cnn.pt")
# Precomputed on the train split during preprocessing — pull the actual
# values printed by Cell 21 of preprocess_clean_final.ipynb and hardcode
# them here so the app doesn't depend on recomputing stats at runtime.
TRAIN_MEAN = -0.4816   
TRAIN_STD = 65.9672   


@st.cache_resource
def load_model():
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")
    model = GenreCNN(num_classes=checkpoint["num_classes"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    idx_to_genre = {v: k for k, v in checkpoint["genre_to_idx"].items()}
    return model, idx_to_genre


def extract_mfcc_array(y, sr, n_mfcc=N_MFCC, target_frames=TARGET_FRAMES):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    if mfcc.shape[1] < target_frames:
        mfcc = np.pad(mfcc, ((0, 0), (0, target_frames - mfcc.shape[1])), mode="constant")
    else:
        mfcc = mfcc[:, :target_frames]
    return mfcc.astype(np.float32)


def predict(model, idx_to_genre, y, sr):
    mfcc = extract_mfcc_array(y, sr)
    mfcc = (mfcc - TRAIN_MEAN) / (TRAIN_STD + 1e-8)
    x = torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # (1, 1, 13, 130)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).squeeze(0).numpy()

    ranked = sorted(
        [(idx_to_genre[i], float(p)) for i, p in enumerate(probs)],
        key=lambda t: t[1],
        reverse=True,
    )
    return ranked


def main():
    st.set_page_config(page_title="Rhythmx Genre Classifier", page_icon="🎵")
    st.title("🎵 Rhythmx — Music Genre Classifier")
    st.caption("CNN trained on 3-second MFCC clips (GTZAN)")

    if not CHECKPOINT_PATH.exists():
        st.error(f"Checkpoint not found at `{CHECKPOINT_PATH}`. Run CNN_Training.ipynb first.")
        return

    model, idx_to_genre = load_model()

    uploaded_file = st.file_uploader("Upload a .wav clip", type=["wav"])

    if uploaded_file is not None:
        y, sr = librosa.load(uploaded_file, sr=SAMPLE_RATE, mono=True)

        # Use the first 3 seconds to match training clip length; pad if shorter
        clip_len = 3 * sr
        if len(y) < clip_len:
            y = np.pad(y, (0, clip_len - len(y)))
        else:
            y = y[:clip_len]

        st.audio(uploaded_file)

        with st.spinner("Classifying..."):
            ranked = predict(model, idx_to_genre, y, sr)

        top_genre, top_prob = ranked[0]
        st.subheader(f"Predicted genre: **{top_genre}** ({top_prob:.1%} confidence)")

        st.write("Full breakdown:")
        for genre, prob in ranked:
            st.progress(prob, text=f"{genre}: {prob:.1%}")


if __name__ == "__main__":
    main()
