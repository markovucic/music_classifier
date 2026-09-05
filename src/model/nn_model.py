import json
import hashlib

import joblib
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, TensorDataset, WeightedRandomSampler
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold, StratifiedKFold

from src.config import N_SPLITS, RANDOM_STATE
from src.model.evaluation import make_sample_weights, majority_vote_prediction, soft_vote_predictions, split_by_work
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
    "cnn_small": (ConvGRUNet, {"conv_channels": (8, 16), "dropout": 0.5}),
    "crnn_small": (ConvGRUNet, {"conv_channels": (8, 16), "gru_hidden": 32, "dropout": 0.5}),
}


# ============================================================
# Shared: used by both pipelines
# ============================================================

def plot_training_history(history, fold=None):

        epochs = range(
            1,
            len(history["train_loss"]) + 1
        )

        fig, axes = plt.subplots(
            1,
            2,
            figsize=(14, 5)
        )

        # LOSS
        axes[0].plot(
            epochs,
            history["train_loss"],
            label="Train loss"
        )

        axes[0].plot(
            epochs,
            history["val_loss"],
            label="Validation loss"
        )

        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].set_title(
            f"Fold {fold} - Loss"
            if fold is not None
            else "Loss"
        )
        axes[0].legend()
        axes[0].grid()


        # ACCURACY
        axes[1].plot(
            epochs,
            history["train_accuracy"],
            label="Train accuracy"
        )

        axes[1].plot(
            epochs,
            history["val_accuracy"],
            label="Validation accuracy"
        )

        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Accuracy")
        axes[1].set_title(
            f"Fold {fold} - Accuracy"
            if fold is not None
            else "Accuracy"
        )
        axes[1].legend()
        axes[1].grid()

        plt.tight_layout()
        plt.show()


# ============================================================
# Pipeline 1: 1D feature-vector input
# ============================================================

def train_and_evaluate_feature_model(
    model,
    train_loader,
    X_val,
    y_val,
    criterion,
    optimizer,
    device,
    epochs=60,
    print_every=5
):
    train_loss_history = []
    val_loss_history = []

    train_acc_history = []
    val_acc_history = []

    if not torch.is_tensor(X_val):
        X_val = torch.tensor(
            X_val,
            dtype=torch.float32
        )

    if not torch.is_tensor(y_val):
        y_val = torch.tensor(
            y_val,
            dtype=torch.long
        )

    X_val = X_val.to(device)
    y_val = y_val.to(device)

    for epoch in range(epochs):

        # =========================
        # TRAIN
        # =========================
        model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        for X_batch, y_batch in train_loader:

            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()

            logits = model(X_batch)

            loss = criterion(
                logits,
                y_batch
            )

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            predictions = torch.argmax(
                logits,
                dim=1
            )

            correct += (
                predictions == y_batch
            ).sum().item()

            total += y_batch.size(0)

        train_loss = (
            running_loss /
            len(train_loader)
        )

        train_accuracy = (
            correct /
            total
        )


        # =========================
        # EVALUATION
        # =========================
        model.eval()

        with torch.no_grad():

            val_logits = model(X_val)

            val_loss = criterion(
                val_logits,
                y_val
            ).item()

            val_predictions = torch.argmax(
                val_logits,
                dim=1
            )

            val_accuracy = (
                val_predictions == y_val
            ).float().mean().item()


        # =========================
        # HISTORY
        # =========================
        train_loss_history.append(
            train_loss
        )

        val_loss_history.append(
            val_loss
        )

        train_acc_history.append(
            train_accuracy
        )

        val_acc_history.append(
            val_accuracy
        )


        # =========================
        # PRINT
        # =========================
        if (epoch + 1) % print_every == 0:

            print(
                f"Epoch {epoch + 1}/{epochs} | "
                f"train_loss={train_loss:.4f} | "
                f"val_loss={val_loss:.4f} | "
                f"train_acc={train_accuracy:.4f} | "
                f"val_acc={val_accuracy:.4f}"
            )


    history = {
        "train_loss": train_loss_history,
        "val_loss": val_loss_history,
        "train_accuracy": train_acc_history,
        "val_accuracy": val_acc_history
    }

    return model, history


def cross_validate_feature_model(
    model_class,
    X,
    y,
    groups=None,
    *,
    n_splits=N_SPLITS,
    epochs=20,
    batch_size=64,
    learning_rate=1e-3,
    weight_decay=0.0,
    random_state=RANDOM_STATE,
    device=None,
    print_every=5,
    show_plots=True,
):
    """Train a fully connected network per fold and report held-out results.

    model_class must accept input_size and num_classes. X is a 2D feature matrix;
    y contains unencoded labels. With groups, use StratifiedGroupKFold and majority
    voting to score one prediction per work.

    Return fold_results (indices, histories, accuracy and macro F1), mean scores,
    decoded held-out y_true/y_pred, class names, and a classification report.
    """
    X = np.asarray(X)
    y = np.asarray(y)
    if X.ndim != 2 or y.ndim != 1 or len(X) != len(y) or len(y) == 0:
        raise ValueError("X must be a non-empty 2D matrix with one label per row")
    if epochs < 1 or batch_size < 1 or print_every < 1:
        raise ValueError("epochs, batch_size and print_every must be positive")

    if groups is not None:
        groups = np.asarray(groups)
        if groups.ndim != 1 or len(groups) != len(y):
            raise ValueError("groups must contain one work ID per row")
        for work_id in np.unique(groups):
            if len(np.unique(y[groups == work_id])) != 1:
                raise ValueError(f"Work {work_id!r} has more than one class label")

    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    splitter_class = StratifiedGroupKFold if groups is not None else StratifiedKFold
    splitter = splitter_class(n_splits=n_splits, shuffle=True, random_state=random_state)
    splits = splitter.split(X, y_encoded, groups) if groups is not None else splitter.split(X, y_encoded)

    print(f"\n{model_class.__name__} | device: {device}")
    print("Classes:", label_encoder.classes_)
    fold_results = []
    all_true, all_pred = [], []

    for fold, (train_idx, val_idx) in enumerate(splits, start=1):
        print(f"\n========== FOLD {fold} ==========")
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[train_idx])
        X_val = scaler.transform(X[val_idx])
        y_train, y_val = y_encoded[train_idx], y_encoded[val_idx]

        train_dataset = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.long),
        )
        if groups is not None:
            sample_weights = make_sample_weights(y[train_idx], groups[train_idx])
            sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
        else:
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        model = model_class(input_size=X.shape[1], num_classes=len(label_encoder.classes_)).to(device)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        model, history = train_and_evaluate_feature_model(
            model=model,
            train_loader=train_loader,
            X_val=X_val,
            y_val=y_val,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epochs=epochs,
            print_every=print_every,
        )
        if show_plots:
            plot_training_history(history, fold=fold)

        model.eval()
        with torch.no_grad():
            logits = model(torch.tensor(X_val, dtype=torch.float32, device=device))
            predictions = logits.argmax(dim=1).cpu().numpy()

        if groups is not None:
            fold_true, fold_pred = majority_vote_prediction(y_val, predictions, groups[val_idx])
        else:
            fold_true, fold_pred = y_val, predictions

        accuracy = metrics.accuracy_score(fold_true, fold_pred)
        macro_f1 = metrics.f1_score(fold_true, fold_pred, average="macro", zero_division=0)
        fold_results.append({
            "fold": fold,
            "train_indices": train_idx,
            "val_indices": val_idx,
            "history": history,
            "accuracy": accuracy,
            "macro_f1": macro_f1,
        })
        all_true.extend(fold_true)
        all_pred.extend(fold_pred)
        print(f"Fold {fold}: accuracy={accuracy:.4f}, macro F1={macro_f1:.4f}")

    y_true = label_encoder.inverse_transform(np.asarray(all_true, dtype=int))
    y_pred = label_encoder.inverse_transform(np.asarray(all_pred, dtype=int))
    mean_accuracy = float(np.mean([r["accuracy"] for r in fold_results]))
    mean_macro_f1 = float(np.mean([r["macro_f1"] for r in fold_results]))
    report = metrics.classification_report(
        y_true, y_pred, labels=label_encoder.classes_, output_dict=True, zero_division=0,
    )
    print("\n========== UKUPNI REZULTATI ==========")
    print(f"Mean accuracy: {mean_accuracy:.4f}")
    print(f"Mean macro F1: {mean_macro_f1:.4f}")
    print(metrics.classification_report(y_true, y_pred, labels=label_encoder.classes_, zero_division=0))

    if show_plots:
        metrics.ConfusionMatrixDisplay.from_predictions(
            y_true, y_pred, labels=label_encoder.classes_, normalize="true",
        )
        plt.title(f"{model_class.__name__} — cross-validation")
        plt.tight_layout()
        plt.show()

    return {
        "model_name": model_class.__name__,
        "classes": label_encoder.classes_,
        "fold_results": fold_results,
        "mean_accuracy": mean_accuracy,
        "mean_macro_f1": mean_macro_f1,
        "y_true": y_true,
        "y_pred": y_pred,
        "classification_report": report,
    }


def save_feature_model(model_name, model_class, X, y, groups, cv_results, epochs=20,
                           batch_size=64, learning_rate=1e-3, weight_decay=0.0,
                           random_state=42, device=None):
    """Refit on (nearly) all data for the same number of epochs used in CV -
    then save it + cv_results (performance estimate)"""
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    gss = GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=random_state)
    train_idx, val_idx = next(gss.split(X, y_encoded, groups))

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X[train_idx])
    X_val = scaler.transform(X[val_idx])
    y_train, y_val = y_encoded[train_idx], y_encoded[val_idx]

    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long)
    )
    sample_weights = make_sample_weights(y[train_idx], groups[train_idx])
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)

    model = model_class(input_size=X.shape[1], num_classes=len(le.classes_)).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    model, history = train_and_evaluate_feature_model(
        model=model, train_loader=train_loader, X_val=X_val, y_val=y_val,
        criterion=criterion, optimizer=optimizer, device=device,
        epochs=epochs, print_every=max(1, epochs // 4),
    )

    run_key = {
        "model": model_class.__name__, "epochs": epochs, "batch_size": batch_size,
        "learning_rate": learning_rate, "weight_decay": weight_decay,
        "n_splits": len(cv_results["fold_results"]),
    }
    run_hash = hashlib.sha1(json.dumps(run_key, sort_keys=True, default=str).encode()).hexdigest()[:8]
    run_dir = OUTPUT_DIR / f"{model_name}_{run_hash}"
    (run_dir / "plots").mkdir(parents=True, exist_ok=True)

    with open(run_dir / "config.json", "w") as f:
        json.dump(run_key, f, indent=2)

    metrics_out = {
        "accuracy": cv_results["mean_accuracy"],
        "macro_f1": cv_results["mean_macro_f1"],
        "classification_report": cv_results["classification_report"],
        "fold_results": [
            {"fold": r["fold"], "accuracy": r["accuracy"], "macro_f1": r["macro_f1"]}
            for r in cv_results["fold_results"]
        ],
    }
    with open(run_dir / "metrics.json", "w") as f:
        json.dump(metrics_out, f, indent=2, default=str)

    torch.save(model.state_dict(), run_dir / "model.pt")
    joblib.dump(scaler, run_dir / "scaler.joblib")
    joblib.dump(le, run_dir / "label_encoder.joblib")

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(history["train_loss"], label="train")
    ax[0].plot(history["val_loss"], label="val")
    ax[0].set_title("Loss")
    ax[0].legend()
    ax[1].plot(history["train_accuracy"], label="train")
    ax[1].plot(history["val_accuracy"], label="val")
    ax[1].set_title("Segment-level accuracy")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(run_dir / "plots" / "loss_curve.png")
    plt.close(fig)

    fig2 = plt.figure(figsize=(6, 5))
    metrics.ConfusionMatrixDisplay.from_predictions(
        cv_results["y_true"], cv_results["y_pred"], labels=cv_results["classes"],
        normalize="true", ax=fig2.gca(),
    )
    fig2.tight_layout()
    fig2.savefig(run_dir / "plots" / "confusion_matrix.png")
    plt.close(fig2)

    print(f"saved to {run_dir}  accuracy={metrics_out['accuracy']:.4f}  macro_f1={metrics_out['macro_f1']:.4f}")
    return str(run_dir)


def run_feature_experiment(model_name, model_class, X, y, groups, n_splits=5, epochs=20,
                               batch_size=64, learning_rate=1e-3, weight_decay=0.0,
                               random_state=42, device=None, print_every=5, show_plots=True,
                               save_model=True):
    """Runs evaluation.cross_validate_feature_model for the honest estimate, then (if
    save_model) refits on (nearly) all data and saves under output/{model_name}_{hash}/."""
    cv_results = cross_validate_feature_model(
        model_class, X, y, groups, n_splits=n_splits, epochs=epochs, batch_size=batch_size,
        learning_rate=learning_rate, weight_decay=weight_decay, random_state=random_state,
        device=device, print_every=print_every, show_plots=show_plots,
    )

    run_dir = None
    if save_model:
        run_dir = save_feature_model(
            model_name, model_class, X, y, groups, cv_results, epochs=epochs,
            batch_size=batch_size, learning_rate=learning_rate, weight_decay=weight_decay,
            random_state=random_state, device=device,
        )

    return cv_results, run_dir


# ============================================================
# Pipeline 2: 2D-matrix models (e.g. spectrograms)
# ============================================================

def _train_conv_model(model, train_loader, val_loader, epochs=20, lr=1e-3, device="cpu", print_every=5):
    """Shared train/eval loop for 2D-matrix-input models (e.g. spectrograms) - used both inside
    cross_validate_conv_model (per fold) and by save_conv_model (final full-data refit)."""
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(epochs):
        model.train()
        total_loss, correct, total = 0.0, 0, 0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)

            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(yb)
            correct += (out.argmax(1) == yb).sum().item()
            total += len(yb)

        train_loss, train_acc = total_loss / total, correct / total

        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0

        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                out = model(xb)
                loss = criterion(out, yb)

                val_loss += loss.item() * len(yb)
                val_correct += (out.argmax(1) == yb).sum().item()
                val_total += len(yb)

        val_loss, val_acc = val_loss / val_total, val_correct / val_total

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if (epoch + 1) % print_every == 0:
            print(
                f"epoch {epoch + 1}/{epochs}: train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
            )

    return model, history


def _predict_probs_conv(model, dataset, device="cpu", batch_size=64):
    loader = DataLoader(dataset, batch_size=batch_size)
    model.eval()
    probs = []

    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(device)
            out = torch.softmax(model(xb), dim=1)
            probs.append(out.cpu().numpy())

    return np.concatenate(probs, axis=0)


def cross_validate_conv_model(model_name, X, y, groups, n_splits=5, epochs=20, batch_size=64,
                               lr=1e-3, random_state=42, device=None, print_every=5,
                               show_plots=True, **arch_overrides):
    """CV for 2D-matrix input:
    fresh model per fold, SpecAugment + WeightedRandomSampler instead of scaling, soft-vote
    instead of majority-vote (we have real class probabilities here)."""
    builder, default_config = MODEL_REGISTRY[model_name]
    config = {**default_config, **arch_overrides}
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    print(f"\n{model_name} | device: {device}")
    print("Classes:", le.classes_)

    fold_results = []
    all_true, all_pred = [], []

    for fold, (train_idx, val_idx) in enumerate(sgkf.split(X, y_encoded, groups), start=1):
        print(f"\n========== FOLD {fold} ==========")

        y_train, y_val = y_encoded[train_idx], y_encoded[val_idx]
        groups_train, groups_val = groups[train_idx], groups[val_idx]

        sample_weights = make_sample_weights(y[train_idx], groups_train)
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

        train_ds = SpectrogramDataset(X[train_idx], y_train, augment=True)
        val_ds = SpectrogramDataset(X[val_idx], y_val, augment=False)

        train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler)
        val_loader = DataLoader(val_ds, batch_size=batch_size)

        model = builder(n_classes=len(le.classes_), **config)
        model, history = _train_conv_model(
            model, train_loader, val_loader, epochs=epochs, lr=lr,
            device=device, print_every=print_every,
        )

        if show_plots:
            plot_training_history({
                "train_loss": history["train_loss"], "val_loss": history["val_loss"],
                "train_accuracy": history["train_acc"], "val_accuracy": history["val_acc"],
            }, fold=fold)

        probs = _predict_probs_conv(model, val_ds, device=device, batch_size=batch_size)
        fold_true, fold_pred = soft_vote_predictions(y[val_idx], probs, groups_val, le.classes_)

        accuracy = metrics.accuracy_score(fold_true, fold_pred)
        macro_f1 = metrics.f1_score(fold_true, fold_pred, average="macro", zero_division=0)
        fold_results.append({"fold": fold, "history": history, "accuracy": accuracy, "macro_f1": macro_f1})
        all_true.extend(fold_true)
        all_pred.extend(fold_pred)
        print(f"Fold {fold}: accuracy={accuracy:.4f}, macro F1={macro_f1:.4f}")

    mean_accuracy = float(np.mean([r["accuracy"] for r in fold_results]))
    mean_macro_f1 = float(np.mean([r["macro_f1"] for r in fold_results]))
    all_true, all_pred = np.array(all_true), np.array(all_pred)
    report = metrics.classification_report(
        all_true, all_pred, labels=le.classes_, output_dict=True, zero_division=0,
    )

    print("\n========== UKUPNI REZULTATI ==========")
    print(f"Mean accuracy: {mean_accuracy:.4f}")
    print(f"Mean macro F1: {mean_macro_f1:.4f}")
    print(metrics.classification_report(all_true, all_pred, labels=le.classes_, zero_division=0))

    if show_plots:
        metrics.ConfusionMatrixDisplay.from_predictions(all_true, all_pred, labels=le.classes_, normalize="true")
        plt.title(f"{model_name} — cross-validation")
        plt.tight_layout()
        plt.show()

    return {
        "model_name": model_name, "classes": le.classes_, "fold_results": fold_results,
        "mean_accuracy": mean_accuracy, "mean_macro_f1": mean_macro_f1,
        "y_true": all_true, "y_pred": all_pred, "classification_report": report,
    }


def save_conv_model(model_name, X, y, groups, cv_results, epochs=20, batch_size=64, lr=1e-3,
                     random_state=42, device=None, **arch_overrides):
    """Refit one final model on (nearly) all data - then save it + cv_results under"""
    builder, default_config = MODEL_REGISTRY[model_name]
    config = {**default_config, **arch_overrides}
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    train_works, val_works = split_by_work(groups, test_size=0.1, random_state=random_state)
    train_mask = np.isin(groups, train_works)
    val_mask = np.isin(groups, val_works)

    sample_weights = make_sample_weights(y[train_mask], groups[train_mask])
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    train_ds = SpectrogramDataset(X[train_mask], y_encoded[train_mask], augment=True)
    val_ds = SpectrogramDataset(X[val_mask], y_encoded[val_mask], augment=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    model = builder(n_classes=len(le.classes_), **config)
    model, history = _train_conv_model(model, train_loader, val_loader, epochs=epochs, lr=lr, device=device)

    run_key = {
        "model_name": model_name, **config, "epochs": epochs, "batch_size": batch_size,
        "lr": lr, "n_splits": len(cv_results["fold_results"]),
    }
    run_hash = hashlib.sha1(json.dumps(run_key, sort_keys=True, default=str).encode()).hexdigest()[:8]
    run_dir = OUTPUT_DIR / f"{model_name}_{run_hash}"
    (run_dir / "plots").mkdir(parents=True, exist_ok=True)

    with open(run_dir / "config.json", "w") as f:
        json.dump(run_key, f, indent=2, default=str)

    metrics_out = {
        "accuracy": cv_results["mean_accuracy"], "macro_f1": cv_results["mean_macro_f1"],
        "classification_report": cv_results["classification_report"],
        "fold_results": [
            {"fold": r["fold"], "accuracy": r["accuracy"], "macro_f1": r["macro_f1"]}
            for r in cv_results["fold_results"]
        ],
    }
    with open(run_dir / "metrics.json", "w") as f:
        json.dump(metrics_out, f, indent=2, default=str)

    torch.save(model.state_dict(), run_dir / "model.pt")

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(history["train_loss"], label="train")
    ax[0].plot(history["val_loss"], label="val")
    ax[0].set_title("Loss")
    ax[0].legend()
    ax[1].plot(history["train_acc"], label="train")
    ax[1].plot(history["val_acc"], label="val")
    ax[1].set_title("Segment-level accuracy")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(run_dir / "plots" / "loss_curve.png")
    plt.close(fig)

    fig2 = plt.figure(figsize=(6, 5))
    metrics.ConfusionMatrixDisplay.from_predictions(
        cv_results["y_true"], cv_results["y_pred"], labels=cv_results["classes"],
        normalize="true", ax=fig2.gca(),
    )
    fig2.tight_layout()
    fig2.savefig(run_dir / "plots" / "confusion_matrix.png")
    plt.close(fig2)

    print(f"saved to {run_dir}  accuracy={metrics_out['accuracy']:.4f}  macro_f1={metrics_out['macro_f1']:.4f}")
    return str(run_dir)


def run_conv_experiment(model_name, X, y, groups, n_splits=5, epochs=20, batch_size=64, lr=1e-3,
                    random_state=42, device=None, save_model=True, **arch_overrides):
    """Runs cross_validate_conv_model for the honest estimate then save"""
    cv_results = cross_validate_conv_model(
        model_name, X, y, groups, n_splits=n_splits, epochs=epochs, batch_size=batch_size,
        lr=lr, random_state=random_state, device=device, **arch_overrides,
    )

    run_dir = None
    if save_model:
        run_dir = save_conv_model(
            model_name, X, y, groups, cv_results, epochs=epochs, batch_size=batch_size, lr=lr,
            random_state=random_state, device=device, **arch_overrides,
        )

    return cv_results, run_dir
