from collections import Counter

import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn import metrics
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from src.config import N_SPLITS, RANDOM_STATE


def make_sample_weights(y, groups):
    """Weight = 1 / (works of that composer x segments of that work), rescaled to mean 1."""
    segments_per_work = Counter(groups)

    work_to_composer = {}

    for composer, work in zip(y, groups):
        work_to_composer[work] = composer

    works_per_composer = Counter(work_to_composer.values())

    weights = []

    for composer, work in zip(y, groups):

        weight = 1.0 / (
            works_per_composer[composer]
            * segments_per_work[work]
        )

        weights.append(weight)

    weights = np.array(weights)

    weights *= len(weights) / weights.sum()

    return weights


def majority_vote_prediction(y_true, y_pred, groups):
    """Collapses segment-level predictions to one prediction per work_id by majority vote."""
    work_true = []
    work_pred = []

    for work_id in np.unique(groups):
        mask = groups == work_id

        true_composer = y_true[mask][0]

        segment_predictions = y_pred[mask]

        predicted_composer = Counter(
            segment_predictions
        ).most_common(1)[0][0]

        work_true.append(true_composer)
        work_pred.append(predicted_composer)

    return np.array(work_true), np.array(work_pred)


def split_by_work(groups, test_size=0.2, random_state=42):
    """Plain (non-stratified) work-level split, reusable across differently-shaped feature sets
    (e.g. audio vs spectrogram segments) that share the same work_id strings."""
    works = np.unique(groups)
    rng = np.random.RandomState(random_state)
    shuffled = works.copy()
    rng.shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * test_size))
    return shuffled[n_val:], shuffled[:n_val]  # train_works, val_works


def soft_vote_probs(probabilities, groups, classes):
    """Like soft_vote_predictions but returns the averaged per-work probability vector
    instead of the argmax label - needed to combine multiple models' predictions."""
    work_ids = np.unique(groups)
    avg_probs = np.array([np.mean(probabilities[groups == w], axis=0) for w in work_ids])
    return work_ids, avg_probs


def soft_vote_predictions(y_true, probabilities, groups, classes):
    """Collapses segment-level class probabilities to one prediction per work_id by averaging."""
    work_true = []
    work_pred = []

    for work_id in np.unique(groups):
        mask = groups == work_id

        true_composer = y_true[mask][0]

        work_probabilities = probabilities[mask]

        mean_probabilities = np.mean(
            work_probabilities,
            axis=0
        )

        predicted_composer = classes[
            np.argmax(mean_probabilities)
        ]

        work_true.append(true_composer)
        work_pred.append(predicted_composer)

    return np.array(work_true), np.array(work_pred)


def train_and_evaluate_neural_network(
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


def cross_validate_neural_network(
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
    """Train a fresh fully connected network per fold and report held-out results.

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
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        model = model_class(input_size=X.shape[1], num_classes=len(label_encoder.classes_)).to(device)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        model, history = train_and_evaluate_neural_network(
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

