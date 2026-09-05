import random
from itertools import product

import numpy as np
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder

from evaluation import majority_vote_prediction, make_sample_weights


def get_random_search_params(param_distributions, count):
    return [
        {param: random.choice(vals) for param, vals in param_distributions.items()}
        for _ in range(count)
    ]


def get_grid_search_params(param_distributions):
    return [
        {hp: p[i] for i, hp in enumerate(param_distributions)}
        for p in product(*param_distributions.values())
    ]


def train_classical_model(X, y, groups, model, cv, param_grid=None, randomized_search=False,
                           count=10, label_encode=False):
    """Grid/randomized search over param_grid, evaluated with group-aware CV + majority vote
    per combination. Returns (results, best) where results is a list of
    (params, mean_acc, mean_f1) tuples and best is the entry with the highest mean_f1."""

    param_grid = param_grid or {}

    le = None
    if label_encode:
        le = LabelEncoder()
        le.fit(y)

    if randomized_search:
        params_list = get_random_search_params(param_grid, count)
    else:
        params_list = get_grid_search_params(param_grid)

    results = []

    for params in params_list:
        work_acc_scores = []
        work_f1_scores = []

        for train_idx, val_idx in cv.split(X, y, groups):
            X_train_fold = X[train_idx]
            y_train_fold = y[train_idx]
            groups_train = groups[train_idx]

            X_val_fold = X[val_idx]
            y_val_fold = y[val_idx]
            groups_val = groups[val_idx]

            if label_encode:
                y_train_fold = le.transform(y_train_fold)

            sample_weights = make_sample_weights(y[train_idx], groups_train)

            model.set_params(**params)
            model.fit(X_train_fold, y_train_fold, sample_weight=sample_weights)

            pred = model.predict(X_val_fold)
            if label_encode:
                pred = le.inverse_transform(pred)

            work_true, work_pred = majority_vote_prediction(y_val_fold, pred, groups_val)

            work_acc_scores.append(metrics.accuracy_score(work_true, work_pred))
            work_f1_scores.append(metrics.f1_score(work_true, work_pred, average="macro"))

        mean_acc = np.mean(work_acc_scores)
        mean_f1 = np.mean(work_f1_scores)

        results.append((params, mean_acc, mean_f1))

        print(
            ", ".join(f"{param}={val}" for param, val in params.items()) +
            f": accuracy={mean_acc:.4f}, macro_f1={mean_f1:.4f}"
        )

    best = max(results, key=lambda r: r[2])

    print("\nBest params:", best[0])
    print("Best accuracy:", best[1])
    print("Best macro F1:", best[2])

    return results, best
