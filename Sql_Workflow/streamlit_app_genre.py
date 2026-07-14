import streamlit as st
from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx
import time
from threading import Thread
import torch
import torch.nn as nn
import librosa
import numpy as np

# ── Thread / Subprocess for ScriptRunContext ───────────────────────────────
# Get results from a thread or subprocess in memory to mitigate Streamlit cache issues.
# Using Streamlit commands within the parent script/page to handle commands.
class WorkerThread(Thread):
    def __init__(self, delay, target):
        super().__init__()
        self.delay = delay
        self.target = target

    def run(self):
        # runs in custom thread but can also call Streamlit APIs
        start_time = time.time()
        time.sleep(self.delay)
        end_time = time.time()
        self.target.write(f"start: {start_time}, end: {end_time}, elapsed: {end_time - start_time:.2f} seconds")


delays = [5, 4, 3, 2, 1]
result_containers = []
for i, delay in enumerate(delays):
    st.header(f"Thread {i}")
    result_containers.append(st.container())

threads = [
    WorkerThread(delay, container)
    for delay, container in zip(delays, result_containers)
]
for thread in threads:
    add_script_run_ctx(thread, get_script_run_ctx())
    thread.start()

for thread in threads:
    thread.join()

st.button("Rerun")


# ── Config ────────────────────────────────────────────────────────────────
model_path = "genre_mlp_model.pt"
DEVICE = torch.device('cpu')

N_MFCC = 13  # must match clean_preprocess_dataset.ipynb


# ── Model definition ─────────────────────────────────────────────────────
# Must match the architecture used in MLP_Training.ipynb exactly, since we
# rebuild it from scratch here and then load the saved weights into it.
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


@st.cache_resource
def load_model():
    checkpoint = torch.load(model_path, map_location=DEVICE)

    model = GenreMLP(
        input_dim=checkpoint['input_dim'],
        hidden_dims=checkpoint['hidden_dims'],
        num_classes=checkpoint['num_classes'],
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return model, checkpoint


model, checkpoint = load_model()

FEATURE_COLS = checkpoint['feature_cols']
FEATURE_MEAN = np.array(checkpoint['feature_mean'], dtype=np.float32)
FEATURE_STD = np.array(checkpoint['feature_std'], dtype=np.float32)
LABEL_MAP = checkpoint['label_map']
INV_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}

SAMPLE_RATE = checkpoint['sample_rate']
DURATION_SEC = checkpoint['duration_sec']
TARGET_SAMPLES = SAMPLE_RATE * DURATION_SEC


def extract_features(y, sr):
    """Extract the same 18 mean-pooled features used in clean_preprocess_dataset.ipynb.

    Column order must match FEATURE_COLS exactly — do not reorder.
    """
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    tempo_val = float(np.atleast_1d(tempo)[0])

    chroma_mean = float(np.mean(librosa.feature.chroma_stft(y=y, sr=sr)))

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    mfcc_means = np.mean(mfcc, axis=1)

    spectral_centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
    spectral_rolloff = float(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)))
    zcr = float(np.mean(librosa.feature.zero_crossing_rate(y)))

    features = {
        'tempo': tempo_val,
        'chroma_mean': chroma_mean,
        **{f'mfcc{i + 1}_mean': mfcc_means[i] for i in range(N_MFCC)},
        'spectral_centroid': spectral_centroid,
        'spectral_rolloff': spectral_rolloff,
        'zero_crossing_rate': zcr,
    }

    # Enforce the exact column order the model was trained on
    return np.array([features[col] for col in FEATURE_COLS], dtype=np.float32)


def preprocess_audio(audio_file):
    """Load an uploaded clip and convert it into a model-ready, normalized tensor."""
    y, sr = librosa.load(audio_file, sr=SAMPLE_RATE, mono=True)

    # Pad or truncate to the fixed clip length the model was trained on
    if len(y) > TARGET_SAMPLES:
        y = y[:TARGET_SAMPLES]
    elif len(y) < TARGET_SAMPLES:
        y = np.pad(y, (0, TARGET_SAMPLES - len(y)), mode='constant')

    raw_features = extract_features(y, sr)

    # Normalize using the TRAINING-set statistics, not this clip's own stats
    normalized = (raw_features - FEATURE_MEAN) / FEATURE_STD

    # Shape: (num_features,) -> (1, num_features) for batch dim
    feature_tensor = torch.tensor(normalized, dtype=torch.float32).unsqueeze(0)
    return feature_tensor


# ── STREAMLIT APP ───────────────────────────────────────────────────────
st.title('GTZAN Music Genre Classifier')
st.caption('Upload a music clip and the model will predict its genre.')

uploaded_audio = st.file_uploader("Upload an audio clip...", type=["wav", "mp3"])

if uploaded_audio is not None:
    st.audio(uploaded_audio)

    if st.button('Classify'):
        with st.spinner('Extracting features and running the model...'):
            feature_tensor = preprocess_audio(uploaded_audio)

            with torch.no_grad():
                logits = model(feature_tensor)
                probabilities = torch.softmax(logits, dim=1)
                confidence, predicted_idx = torch.max(probabilities, dim=1)

        prediction = INV_LABEL_MAP[predicted_idx.item()]

        st.success(f'Prediction: **{prediction}**')
        st.write(f'Confidence: {confidence.item() * 100:.2f}%')

        # Show full probability breakdown across all genres
        probs_by_genre = {
            INV_LABEL_MAP[i]: probabilities[0, i].item()
            for i in range(len(LABEL_MAP))
        }
        st.bar_chart(probs_by_genre)


streamlit run Sql_Workflow/streamlit_app_genre.py