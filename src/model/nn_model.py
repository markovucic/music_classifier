import os
import json
import hashlib
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GroupShuffleSplit

from sklearn.ensemble import RandomForestClassifier

from src.model.evaluation import make_sample_weights, soft_vote_predictions, split_by_work, soft_vote_probs
from src.utils.paths import OUTPUT_DIR


def _spec_augment(x, freq_mask=20, time_mask=40, n_freq_masks=1, n_time_masks=1):
    x = x.copy()
    n_mels, t = x.shape
    fill = x.mean()

    for _ in range(n_freq_masks):
        f = np.random.randint(0, freq_mask)
        f0 = np.random.randint(0, max(1, n_mels - f))
        x[f0:f0 + f, :] = fill

    for _ in range(n_time_masks):
        w = np.random.randint(0, time_mask)
        t0 = np.random.randint(0, max(1, t - w))
        x[:, t0:t0 + w] = fill

    return x


class SpectrogramDataset(Dataset):
    def __init__(self, X, y, augment=False):
        self.X = X
        self.y = y
        self.augment = augment

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]

        if self.augment:
            x = _spec_augment(x)

        x = (x - x.mean()) / (x.std() + 1e-6)
        x = torch.from_numpy(x).unsqueeze(0).float()

        return x, int(self.y[idx])


class SimpleCNN(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(64, n_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).flatten(1)
        x = self.dropout(x)
        return self.fc(x)


class ConvGRUNet(nn.Module):
    """Configurable CNN, optionally with a GRU over the time axis before pooling."""

    def __init__(self, n_classes, conv_channels=(16, 32, 64), gru_hidden=None, dropout=0.5):
        super().__init__()
        layers = []
        in_ch = 1
        for out_ch in conv_channels:
            layers += [nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(), nn.MaxPool2d(2)]
            in_ch = out_ch
        self.features = nn.Sequential(*layers)

        self.gru_hidden = gru_hidden
        if gru_hidden:
            self.gru = nn.GRU(input_size=in_ch, hidden_size=gru_hidden, batch_first=True, bidirectional=True)
            fc_in = gru_hidden * 2
        else:
            self.pool = nn.AdaptiveAvgPool2d(1)
            fc_in = in_ch

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(fc_in, n_classes)

    def forward(self, x):
        x = self.features(x)  # (B, C, F, T)

        if self.gru_hidden:
            x = x.mean(dim=2)  # collapse frequency -> (B, C, T)
            x = x.transpose(1, 2)  # (B, T, C)
            x, _ = self.gru(x)
            x = x.mean(dim=1)  # average over time
        else:
            x = self.pool(x).flatten(1)

        x = self.dropout(x)
        return self.fc(x)


MODEL_REGISTRY = {
    "cnn_small": (ConvGRUNet, {"conv_channels": (8, 16, 32), "dropout": 0.5}),
    "crnn_small": (ConvGRUNet, {"conv_channels": (8, 16), "gru_hidden": 32, "dropout": 0.5}),
}


# def train_model(model, train_ds, val_ds, sample_weights=None, epochs=20, batch_size=32, lr=1e-3,
#                  device="cpu", checkpoint_path=None, history_path=None):
#     if sample_weights is not None:
#         sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
#         train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler)
#     else:
#         train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

#     val_loader = DataLoader(val_ds, batch_size=batch_size)

#     model.to(device)
#     optimizer = torch.optim.Adam(model.parameters(), lr=lr)
#     criterion = nn.CrossEntropyLoss()

#     history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
#     best_val_loss = float("inf")

#     for epoch in range(epochs):
#         try:
#             model.train()
#             total_loss, correct, total = 0.0, 0, 0

#             pbar = tqdm(train_loader, desc=f"epoch {epoch + 1}/{epochs}")

#             for xb, yb in pbar:
#                 xb, yb = xb.to(device), yb.to(device)

#                 optimizer.zero_grad()
#                 out = model(xb)
#                 loss = criterion(out, yb)
#                 loss.backward()
#                 optimizer.step()

#                 total_loss += loss.item() * len(yb)
#                 correct += (out.argmax(1) == yb).sum().item()
#                 total += len(yb)

#                 pbar.set_postfix(loss=total_loss / total, acc=correct / total)

#             train_loss, train_acc = total_loss / total, correct / total

#             model.eval()
#             val_loss, val_correct, val_total = 0.0, 0, 0

#             with torch.no_grad():
#                 for xb, yb in tqdm(val_loader, desc=f"epoch {epoch + 1}/{epochs} [val]", leave=False):
#                     xb, yb = xb.to(device), yb.to(device)
#                     out = model(xb)
#                     loss = criterion(out, yb)

#                     val_loss += loss.item() * len(yb)
#                     val_correct += (out.argmax(1) == yb).sum().item()
#                     val_total += len(yb)

#             val_loss, val_acc = val_loss / val_total, val_correct / val_total

#             if checkpoint_path is not None and val_loss < best_val_loss:
#                 best_val_loss = val_loss
#                 os.makedirs(os.path.dirname(checkpoint_path) or ".", exist_ok=True)
#                 torch.save(model.state_dict(), checkpoint_path)

#             history["train_loss"].append(train_loss)
#             history["train_acc"].append(train_acc)
#             history["val_loss"].append(val_loss)
#             history["val_acc"].append(val_acc)

#             print(
#                 f"epoch {epoch + 1}/{epochs}: "
#                 f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
#                 f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
#             )
#         except Exception as e:
#             print(f"epoch {epoch + 1}/{epochs} FAILED ({e}) — stopping, keeping progress so far")
#             history["error"] = str(e)
#             break

#         if history_path is not None:
#             os.makedirs(os.path.dirname(history_path) or ".", exist_ok=True)
#             with open(history_path, "w") as f:
#                 json.dump(history, f, indent=2)

#     return history


# def predict_probs(model, X, device="cpu", batch_size=64):
#     ds = SpectrogramDataset(X, np.zeros(len(X)), augment=False)
#     loader = DataLoader(ds, batch_size=batch_size)

#     model.eval()
#     probs = []

#     with torch.no_grad():
#         for xb, _ in tqdm(loader, desc="predicting"):
#             xb = xb.to(device)
#             out = torch.softmax(model(xb), dim=1)
#             probs.append(out.cpu().numpy())

#     return np.concatenate(probs, axis=0)


# def plot_history(history):
#     fig, ax = plt.subplots(1, 2, figsize=(12, 4))

#     ax[0].plot(history["train_loss"], label="train")
#     ax[0].plot(history["val_loss"], label="val")
#     ax[0].set_title("Loss")
#     ax[0].legend()

#     ax[1].plot(history["train_acc"], label="train")
#     ax[1].plot(history["val_acc"], label="val")
#     ax[1].set_title("Segment-level accuracy")
#     ax[1].legend()

#     plt.tight_layout()
#     plt.show()


# def _save_plots(history, work_true, work_pred, run_dir):
#     fig, ax = plt.subplots(1, 2, figsize=(12, 4))
#     ax[0].plot(history["train_loss"], label="train")
#     ax[0].plot(history["val_loss"], label="val")
#     ax[0].set_title("Loss")
#     ax[0].legend()
#     ax[1].plot(history["train_acc"], label="train")
#     ax[1].plot(history["val_acc"], label="val")
#     ax[1].set_title("Segment-level accuracy")
#     ax[1].legend()
#     fig.tight_layout()
#     fig.savefig(run_dir / "plots" / "loss_curve.png")
#     plt.close(fig)

#     fig2 = plt.figure(figsize=(6, 5))
#     metrics.ConfusionMatrixDisplay.from_predictions(work_true, work_pred, normalize="true", ax=fig2.gca())
#     fig2.tight_layout()
#     fig2.savefig(run_dir / "plots" / "confusion_matrix.png")
#     plt.close(fig2)


# def _save_report_pdf(run_dir, model_name, config, acc, f1):
#     with PdfPages(run_dir / "report.pdf") as pdf:
#         fig = plt.figure(figsize=(8.5, 11))
#         fig.text(0.1, 0.95, model_name, fontsize=16, weight="bold")
#         fig.text(0.1, 0.90, f"accuracy={acc:.4f}   macro_f1={f1:.4f}")
#         cfg_text = "\n".join(f"{k}: {v}" for k, v in config.items())
#         fig.text(0.1, 0.5, cfg_text, fontsize=9, va="top")
#         pdf.savefig(fig)
#         plt.close(fig)

#         for name in ("loss_curve.png", "confusion_matrix.png"):
#             img = plt.imread(run_dir / "plots" / name)
#             fig = plt.figure(figsize=(8.5, 11))
#             plt.imshow(img)
#             plt.axis("off")
#             pdf.savefig(fig)
#             plt.close(fig)


# def _log_error(run_dir, stage, e):
#     import traceback
#     with open(run_dir / "error.log", "a") as f:
#         f.write(f"--- failed at: {stage} ---\n")
#         f.write(traceback.format_exc())
#         f.write("\n")
#     print(f"ERROR ({stage}) in {run_dir}: {e}  -- see error.log, keeping whatever was already saved")


# def run_experiment(model_name, X, y, groups, epochs=20, batch_size=64, lr=1e-3,
#                     test_size=0.2, random_state=42, device=None, **arch_overrides):
#     """Builds a model from MODEL_REGISTRY, trains + evaluates it, and saves everything
#     (config, metrics, history, weights, plots, PDF report) under output/{model_name}_{hash}/.
#     Never raises: each stage is wrapped so a failure keeps whatever was already saved and
#     logs the error to error.log instead of crashing the whole run/notebook."""

#     builder, default_config = MODEL_REGISTRY[model_name]
#     config = {**default_config, **arch_overrides}
#     train_kwargs = {"epochs": epochs, "batch_size": batch_size, "lr": lr}
#     device = device or ("cuda" if torch.cuda.is_available() else "cpu")

#     run_key = {"model_name": model_name, **config, **train_kwargs, "test_size": test_size, "random_state": random_state}
#     run_hash = hashlib.sha1(json.dumps(run_key, sort_keys=True).encode()).hexdigest()[:8]
#     run_dir = OUTPUT_DIR / f"{model_name}_{run_hash}"
#     (run_dir / "plots").mkdir(parents=True, exist_ok=True)

#     with open(run_dir / "config.json", "w") as f:
#         json.dump({"model_name": model_name, **config, **train_kwargs}, f, indent=2)

#     le = LabelEncoder()
#     y_enc = le.fit_transform(y)

#     gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
#     train_idx, val_idx = next(gss.split(X, y_enc, groups))

#     X_train, y_train, groups_train = X[train_idx], y_enc[train_idx], groups[train_idx]
#     X_val, y_val, groups_val = X[val_idx], y[val_idx], groups[val_idx]

#     sample_weights = make_sample_weights(y[train_idx], groups_train)

#     train_ds = SpectrogramDataset(X_train, y_train, augment=True)
#     val_ds = SpectrogramDataset(X_val, le.transform(y_val), augment=False)

#     model = builder(n_classes=len(le.classes_), **config)

#     try:
#         history = train_model(
#             model, train_ds, val_ds,
#             sample_weights=sample_weights,
#             device=device,
#             checkpoint_path=str(run_dir / "model.pt"),
#             history_path=str(run_dir / "history.json"),
#             **train_kwargs
#         )
#     except Exception as e:
#         _log_error(run_dir, "training", e)
#         return {"run_dir": str(run_dir), "status": "failed_training", "error": str(e)}

#     try:
#         model.load_state_dict(torch.load(run_dir / "model.pt"))  # best val_loss weights, not last epoch

#         probs = predict_probs(model, X_val, device=device)
#         work_true, work_pred = soft_vote_predictions(y_val, probs, groups_val, le.classes_)

#         report = metrics.classification_report(work_true, work_pred, output_dict=True)
#         acc = metrics.accuracy_score(work_true, work_pred)
#         f1 = metrics.f1_score(work_true, work_pred, average="macro")

#         with open(run_dir / "metrics.json", "w") as f:
#             json.dump({"accuracy": acc, "macro_f1": f1, "report": report}, f, indent=2)
#     except Exception as e:
#         _log_error(run_dir, "evaluation", e)
#         return {"run_dir": str(run_dir), "status": "failed_evaluation", "history": history, "error": str(e)}

#     try:
#         _save_plots(history, work_true, work_pred, run_dir)
#         _save_report_pdf(run_dir, model_name, {**config, **train_kwargs}, acc, f1)
#     except Exception as e:
#         _log_error(run_dir, "plots/report", e)
#         return {"run_dir": str(run_dir), "status": "failed_plots", "accuracy": acc, "macro_f1": f1,
#                 "history": history, "error": str(e)}

#     print(f"saved to {run_dir}  accuracy={acc:.4f}  macro_f1={f1:.4f}")

#     return {"run_dir": str(run_dir), "status": "ok", "accuracy": acc, "macro_f1": f1, "history": history}


# def run_ensemble_experiment(cnn_model_name, X_spec, y, groups_spec, X_audio, y_audio, groups_audio,
#                             epochs=20, batch_size=64, lr=1e-3, test_size=0.2, random_state=42,
#                             device=None, **arch_overrides):
#     """Trains cnn_model_name (on spectrograms) and a Random Forest (on hand-crafted audio features)
#     over the SAME work-level split, then averages their work-level probabilities. X_spec/groups_spec
#     and X_audio/groups_audio can have different segment counts per work - split_by_work + soft_vote_probs
#     align everything at the work level regardless."""

#     device = device or ("cuda" if torch.cuda.is_available() else "cpu")
#     train_kwargs = {"epochs": epochs, "batch_size": batch_size, "lr": lr}

#     run_key = {"cnn_model_name": cnn_model_name, **arch_overrides, **train_kwargs,
#                "test_size": test_size, "random_state": random_state}
#     run_hash = hashlib.sha1(json.dumps(run_key, sort_keys=True).encode()).hexdigest()[:8]
#     run_dir = OUTPUT_DIR / f"ensemble_rf_{cnn_model_name}_{run_hash}"
#     (run_dir / "plots").mkdir(parents=True, exist_ok=True)

#     with open(run_dir / "config.json", "w") as f:
#         json.dump({"cnn_model_name": cnn_model_name, **arch_overrides, **train_kwargs}, f, indent=2)

#     train_works, val_works = split_by_work(groups_spec, test_size=test_size, random_state=random_state)

#     try:
#         # --- CNN branch ---
#         builder, default_config = MODEL_REGISTRY[cnn_model_name]
#         config = {**default_config, **arch_overrides}

#         spec_train_mask = np.isin(groups_spec, train_works)
#         spec_val_mask = np.isin(groups_spec, val_works)

#         le = LabelEncoder()
#         y_enc = le.fit_transform(y)

#         X_train, y_train, groups_train = X_spec[spec_train_mask], y_enc[spec_train_mask], groups_spec[spec_train_mask]
#         X_val, y_val, groups_val = X_spec[spec_val_mask], y[spec_val_mask], groups_spec[spec_val_mask]

#         sample_weights = make_sample_weights(y[spec_train_mask], groups_train)
#         train_ds = SpectrogramDataset(X_train, y_train, augment=True)
#         val_ds = SpectrogramDataset(X_val, le.transform(y_val), augment=False)

#         model = builder(n_classes=len(le.classes_), **config)
#         train_model(
#             model, train_ds, val_ds,
#             sample_weights=sample_weights, device=device,
#             checkpoint_path=str(run_dir / "cnn_model.pt"),
#             history_path=str(run_dir / "cnn_history.json"),
#             **train_kwargs
#         )
#         model.load_state_dict(torch.load(run_dir / "cnn_model.pt"))

#         cnn_probs_seg = predict_probs(model, X_val, device=device)
#         cnn_work_ids, cnn_probs = soft_vote_probs(cnn_probs_seg, groups_val, le.classes_)

#         # --- RF branch (same audio RF config as classic_models_raw.ipynb) ---
#         audio_train_mask = np.isin(groups_audio, train_works)
#         audio_val_mask = np.isin(groups_audio, val_works)

#         rf_sample_weights = make_sample_weights(y_audio[audio_train_mask], groups_audio[audio_train_mask])
#         rf = RandomForestClassifier(
#             n_estimators=300, max_depth=10, min_samples_leaf=1,
#             max_features="sqrt", random_state=42, n_jobs=-1
#         )
#         rf.fit(X_audio[audio_train_mask], y_audio[audio_train_mask], sample_weight=rf_sample_weights)

#         rf_probs_seg = rf.predict_proba(X_audio[audio_val_mask])
#         rf_work_ids, rf_probs = soft_vote_probs(rf_probs_seg, groups_audio[audio_val_mask], rf.classes_)

#         assert list(rf.classes_) == list(le.classes_), "class order mismatch between RF and CNN"

#         # --- combine at the work level ---
#         common_works = np.array(sorted(set(cnn_work_ids) & set(rf_work_ids)))
#         cnn_idx = {w: i for i, w in enumerate(cnn_work_ids)}
#         rf_idx = {w: i for i, w in enumerate(rf_work_ids)}

#         combined_probs = np.array([(cnn_probs[cnn_idx[w]] + rf_probs[rf_idx[w]]) / 2 for w in common_works])
#         work_true = np.array([y[groups_spec == w][0] for w in common_works])
#         work_pred = le.classes_[combined_probs.argmax(axis=1)]

#         report = metrics.classification_report(work_true, work_pred, output_dict=True)
#         acc = metrics.accuracy_score(work_true, work_pred)
#         f1 = metrics.f1_score(work_true, work_pred, average="macro")

#         with open(run_dir / "metrics.json", "w") as f:
#             json.dump({"accuracy": acc, "macro_f1": f1, "report": report,
#                        "n_common_works": len(common_works)}, f, indent=2)

#         fig = plt.figure(figsize=(6, 5))
#         metrics.ConfusionMatrixDisplay.from_predictions(work_true, work_pred, normalize="true", ax=fig.gca())
#         fig.tight_layout()
#         fig.savefig(run_dir / "plots" / "confusion_matrix.png")
#         plt.close(fig)
#     except Exception as e:
#         _log_error(run_dir, "ensemble", e)
#         return {"run_dir": str(run_dir), "status": "failed_ensemble", "error": str(e)}

#     print(f"saved to {run_dir}  accuracy={acc:.4f}  macro_f1={f1:.4f}  (n={len(common_works)} works)")

#     return {"run_dir": str(run_dir), "status": "ok", "accuracy": acc, "macro_f1": f1}


# def run_panns_experiment(X, y, groups, test_size=0.2, random_state=42, **rf_overrides):
#     """Trains a Random Forest on frozen PANNs embeddings (X from panns_features.make_panns_dataset) -
#     the 'transfer learning' branch: no NN training here, just a classical head on top of a
#     pretrained feature extractor."""

#     config = {"n_estimators": 300, "max_depth": 10, "min_samples_leaf": 1, "max_features": "sqrt", **rf_overrides}

#     run_key = {**config, "test_size": test_size, "random_state": random_state}
#     run_hash = hashlib.sha1(json.dumps(run_key, sort_keys=True).encode()).hexdigest()[:8]
#     run_dir = OUTPUT_DIR / f"panns_rf_{run_hash}"
#     (run_dir / "plots").mkdir(parents=True, exist_ok=True)

#     with open(run_dir / "config.json", "w") as f:
#         json.dump(config, f, indent=2)

#     train_works, val_works = split_by_work(groups, test_size=test_size, random_state=random_state)
#     train_mask = np.isin(groups, train_works)
#     val_mask = np.isin(groups, val_works)

#     try:
#         sample_weights = make_sample_weights(y[train_mask], groups[train_mask])
#         rf = RandomForestClassifier(random_state=42, n_jobs=-1, **config)
#         rf.fit(X[train_mask], y[train_mask], sample_weight=sample_weights)

#         probs = rf.predict_proba(X[val_mask])
#         work_true, work_pred = soft_vote_predictions(y[val_mask], probs, groups[val_mask], rf.classes_)

#         report = metrics.classification_report(work_true, work_pred, output_dict=True)
#         acc = metrics.accuracy_score(work_true, work_pred)
#         f1 = metrics.f1_score(work_true, work_pred, average="macro")

#         with open(run_dir / "metrics.json", "w") as f:
#             json.dump({"accuracy": acc, "macro_f1": f1, "report": report}, f, indent=2)

#         fig = plt.figure(figsize=(6, 5))
#         metrics.ConfusionMatrixDisplay.from_predictions(work_true, work_pred, normalize="true", ax=fig.gca())
#         fig.tight_layout()
#         fig.savefig(run_dir / "plots" / "confusion_matrix.png")
#         plt.close(fig)
#     except Exception as e:
#         _log_error(run_dir, "panns_rf", e)
#         return {"run_dir": str(run_dir), "status": "failed", "error": str(e)}

#     print(f"saved to {run_dir}  accuracy={acc:.4f}  macro_f1={f1:.4f}")

#     return {"run_dir": str(run_dir), "status": "ok", "accuracy": acc, "macro_f1": f1}
