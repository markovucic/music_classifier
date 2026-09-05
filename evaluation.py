from collections import Counter

import numpy as np
import matplotlib.pyplot as plt
import torch


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
        # ISPIS
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