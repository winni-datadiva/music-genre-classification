# Rhythmx — Sprint 4: 3-Way Comparison Workflow
### Rhythmx `GenreCNN` vs. musicnn vs. MERT

This picks up after `Sprint4_Embedding_Extraction.ipynb` has produced:

```
embeddings/musicnn_train.npy  musicnn_train_labels.csv
embeddings/musicnn_val.npy    musicnn_val_labels.csv
embeddings/musicnn_test.npy   musicnn_test_labels.csv
embeddings/mert_train.npy     mert_train_labels.csv
embeddings/mert_val.npy       mert_val_labels.csv
embeddings/mert_test.npy      mert_test_labels.csv
embeddings/extraction_timing.csv
```

Run these steps in a new notebook (e.g. `Sprint4_Model_Comparison.ipynb`) inside your
regular `DS_class` env — this part is pure sklearn/PyTorch, no musicnn/MERT dependencies
needed, so no env-switching required. 

---

## Step 1 — Train a classifier head per embedding set

Keep the classifier head **identical in structure** across musicnn and MERT embeddings.
That isolates what you're actually testing: the quality of the pretrained embeddings,
not differences in classifier capacity. Two options — pick one and apply it to both:

### Option A: Logistic Regression (fast, strong baseline for linear separability)

```python
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

def load_split(model_name, split):
    X = np.load(f"embeddings/{model_name}_{split}.npy")
    y = pd.read_csv(f"embeddings/{model_name}_{split}_labels.csv")["genre"].values
    return X, y

def train_logreg_head(model_name, seed=42):
    X_train, y_train = load_split(model_name, "train")
    X_val, y_val = load_split(model_name, "val")
    X_test, y_test = load_split(model_name, "test")

    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_val_enc = le.transform(y_val)
    y_test_enc = le.transform(y_test)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    clf = LogisticRegression(max_iter=2000, random_state=seed, multi_class="multinomial")
    clf.fit(X_train_s, y_train_enc)

    val_preds = clf.predict(X_val_s)
    test_preds = clf.predict(X_test_s)

    return {
        "model_name": model_name,
        "clf": clf,
        "label_encoder": le,
        "val_acc": accuracy_score(y_val_enc, val_preds),
        "val_f1_macro": f1_score(y_val_enc, val_preds, average="macro"),
        "test_acc": accuracy_score(y_test_enc, test_preds),
        "test_f1_macro": f1_score(y_test_enc, test_preds, average="macro"),
        "y_test_true": y_test_enc,
        "y_test_pred": test_preds,
        "n_params": X_train.shape[1] * len(le.classes_),  # weight matrix size
    }

musicnn_results = train_logreg_head("musicnn")
mert_results = train_logreg_head("mert")
```

### Option B: Small MLP head (if linear separability looks weak in Option A)

```python
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

class ClassifierHead(nn.Module):
    def __init__(self, input_dim, n_classes, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, x):
        return self.net(x)

def train_mlp_head(model_name, epochs=30, lr=1e-3, seed=42):
    torch.manual_seed(seed)
    X_train, y_train = load_split(model_name, "train")
    X_val, y_val = load_split(model_name, "val")
    X_test, y_test = load_split(model_name, "test")

    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_val_enc = le.transform(y_val)
    y_test_enc = le.transform(y_test)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    train_ds = TensorDataset(
        torch.tensor(X_train_s, dtype=torch.float32),
        torch.tensor(y_train_enc, dtype=torch.long),
    )
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)

    n_classes = len(le.classes_)
    model = ClassifierHead(X_train.shape[1], n_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        val_preds = model(torch.tensor(X_val_s, dtype=torch.float32)).argmax(1).numpy()
        test_preds = model(torch.tensor(X_test_s, dtype=torch.float32)).argmax(1).numpy()

    n_params = sum(p.numel() for p in model.parameters())

    return {
        "model_name": model_name,
        "val_acc": accuracy_score(y_val_enc, val_preds),
        "val_f1_macro": f1_score(y_val_enc, val_preds, average="macro"),
        "test_acc": accuracy_score(y_test_enc, test_preds),
        "test_f1_macro": f1_score(y_test_enc, test_preds, average="macro"),
        "y_test_true": y_test_enc,
        "y_test_pred": test_preds,
        "label_encoder": le,
        "n_params": n_params,
    }
```

**Recommendation:** run Option A first — it's fast to iterate on and often sufficient
for well-separated pretrained embeddings. Only reach for Option B if the linear
classifier's val accuracy looks meaningfully worse than your CNN's, since that could
mean the genre boundaries are non-linear in embedding space rather than the embeddings
being weak.

---

## Step 2 — Bring in your CNN's Sprint 3 test results

Reuse the exact test-set predictions your `GenreCNN` produced in `CNN_Training.ipynb`
so all three models are scored identically. If you didn't save raw predictions there,
re-run inference on the same test split now:

```python
# Adjust to match your Sprint 3 model-loading code
cnn_model.eval()
cnn_test_preds = []
cnn_test_true = []

with torch.no_grad():
    for mfcc_batch, label_batch in cnn_test_loader:   # same test loader as Sprint 3
        outputs = cnn_model(mfcc_batch)
        preds = outputs.argmax(dim=1)
        cnn_test_preds.extend(preds.cpu().numpy())
        cnn_test_true.extend(label_batch.cpu().numpy())

cnn_results = {
    "model_name": "rhythmx_cnn",
    "test_acc": accuracy_score(cnn_test_true, cnn_test_preds),
    "test_f1_macro": f1_score(cnn_test_true, cnn_test_preds, average="macro"),
    "y_test_true": cnn_test_true,
    "y_test_pred": cnn_test_preds,
    "n_params": sum(p.numel() for p in cnn_model.parameters()),
}
```

> **Important:** confirm your CNN's label encoding order matches the `LabelEncoder`
> used for musicnn/MERT above (both should alphabetize genre strings the same way
> if built from the same source data — but verify with `le.classes_` before comparing
> confusion matrices).

---

## Step 3 — Metrics comparison table

```python
all_results = [cnn_results, musicnn_results, mert_results]

summary = pd.DataFrame([
    {
        "model": r["model_name"],
        "test_accuracy": round(r["test_acc"], 4),
        "test_f1_macro": round(r["test_f1_macro"], 4),
        "n_params": r["n_params"],
    }
    for r in all_results
]).sort_values("test_f1_macro", ascending=False)

print(summary.to_string(index=False))
```

Macro F1 matters more than raw accuracy here — GTZAN's genres aren't perfectly
balanced, and macro F1 won't let one dominant genre mask poor performance on others.

---

## Step 4 — Confusion matrices (per model)

```python
import matplotlib.pyplot as plt
import seaborn as sns

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for ax, r in zip(axes, all_results):
    cm = confusion_matrix(r["y_test_true"], r["y_test_pred"])
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax, cbar=False)
    ax.set_title(r["model_name"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

plt.tight_layout()
plt.savefig("embeddings/confusion_matrices_comparison.png", dpi=150)
plt.show()
```

Look specifically for **which genre pairs each model confuses** — e.g. rock/metal,
disco/pop, country/blues are classic GTZAN failure pairs. If your CNN and the
pretrained models fail on different pairs, that's a strong finding: it suggests
they're picking up different acoustic cues (MFCC texture vs. learned representations).

---

## Step 5 — Cost / efficiency comparison

Combine extraction timing (from Sprint 4 notebook) with parameter counts and
inference latency, since accuracy alone doesn't tell the deployment story:

```python
timing_df = pd.read_csv("embeddings/extraction_timing.csv")
test_timing = timing_df[timing_df["split"] == "test"][["model", "seconds_per_clip"]]

cost_summary = summary.merge(
    test_timing, left_on="model", right_on="model", how="left"
)
# Rhythmx CNN inference timing: measure directly if not already logged in Sprint 3
print(cost_summary.to_string(index=False))
```

This table — accuracy/F1 vs. params vs. per-clip latency — is exactly the kind of
tradeoff analysis that makes a capstone comparison feel like an engineering decision
rather than just a leaderboard.

---

## Step 6 — Grad-CAM on the CNN (optional but strong for the writeup)

Since musicnn/MERT are effectively black boxes here (frozen embeddings + linear head),
Grad-CAM on your own `GenreCNN` gives you an interpretability angle the pretrained
comparisons can't offer — worth including as a contrast point.

```python
import torch.nn.functional as F

def grad_cam(model, mfcc_input, target_layer, target_class):
    activations = {}
    gradients = {}

    def forward_hook(module, input, output):
        activations["value"] = output

    def backward_hook(module, grad_in, grad_out):
        gradients["value"] = grad_out[0]

    handle_fwd = target_layer.register_forward_hook(forward_hook)
    handle_bwd = target_layer.register_full_backward_hook(backward_hook)

    model.eval()
    output = model(mfcc_input)
    model.zero_grad()
    output[0, target_class].backward()

    handle_fwd.remove()
    handle_bwd.remove()

    acts = activations["value"].squeeze(0)      # (channels, H, W)
    grads = gradients["value"].squeeze(0)        # (channels, H, W)
    weights = grads.mean(dim=(1, 2))              # global-average-pool the gradients

    cam = torch.zeros(acts.shape[1:])
    for i, w in enumerate(weights):
        cam += w * acts[i]

    cam = F.relu(cam)
    cam = cam / (cam.max() + 1e-8)
    return cam.detach().cpu().numpy()

# Example usage — point target_layer at your last conv layer in GenreCNN
# cam = grad_cam(cnn_model, sample_mfcc, cnn_model.conv3, target_class=predicted_idx)
# plt.imshow(sample_mfcc.squeeze(), cmap="gray")
# plt.imshow(cam, cmap="jet", alpha=0.5)
```

---

## Step 7 — Deployment decision

Once Steps 3–5 are done, document the tradeoff explicitly for your capstone writeup
and for the Streamlit/FastAPI decision:

| Question | Answer once you have results |
|---|---|
| Which model wins on macro F1? | fill in |
| Which model is cheapest to run at inference (latency + params)? | fill in |
| Does the winner justify the extra complexity/dependencies (TF + HF)? | fill in |
| Final choice for `streamlit_app_genre.py` / `api_app_genre.py` | fill in |

If a pretrained model wins by a wide margin, consider shipping it instead of/alongside
the CNN — but note the added deployment weight (musicnn needs TensorFlow, MERT needs
a 330M-param download) against your original scope of a lightweight from-scratch model.
If the CNN holds its own, that's a legitimate and strong capstone result on its own —
"a compact custom model matches large pretrained baselines on this task" is a fine story.
