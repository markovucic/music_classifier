from collections import Counter

import numpy as np


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
