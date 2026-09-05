import random
from itertools import product

import numpy as np
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from src.model.evaluation import majority_vote_prediction, make_sample_weights


def _sample_weight_kwarg(model, sample_weights):
    """Pipeline.fit needs sample_weight prefixed with the last step's name; a plain estimator
    takes it directly."""
    if isinstance(model, Pipeline):
        return {f"{model.steps[-1][0]}__sample_weight": sample_weights}
    return {"sample_weight": sample_weights}


def _normalize_param_grid(model, param_grid):
    """Auto-prefix param_grid keys for a Pipeline (e.g. "C" -> "svc__C"), so the same
    param_grid works whether model is a plain estimator or wrapped in a pipeline."""
    if not isinstance(model, Pipeline):
        return param_grid

    valid = set(model.get_params())
    last_step = model.steps[-1][0]

    return {
        key if key in valid else f"{last_step}__{key}": vals
        for key, vals in param_grid.items()
    }


def _validate_param_grid(model, param_grid):
    """Fail fast on unknown/unprefixed param names (e.g. "C" instead of "svc__C" for a
    Pipeline) instead of erroring deep inside the CV loop."""
    unknown = set(param_grid) - set(model.get_params())
    if unknown:
        raise ValueError(
            f"param_grid has keys not valid for {type(model).__name__}: {sorted(unknown)} "
            f"(for a Pipeline, prefix with the step name, e.g. \"svc__C\")"
        )


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


def get_best_result(results, thresholds=(0.01, 0.05, 0.1, 0.2)):
    """Among results within `th` of the best accuracy AND best macro F1, return the one with
    the highest macro F1. Guards against picking hyperparameters that only won by CV noise;
    progressively relaxes th until at least one candidate qualifies."""
    max_acc = max(r[1] for r in results)
    max_f1 = max(r[2] for r in results)

    candidates = results
    for th in thresholds:
        near_best = [r for r in results if r[1] > (1 - th) * max_acc and r[2] > (1 - th) * max_f1]
        if near_best:
            candidates = near_best
            break

    return max(candidates, key=lambda r: r[2])


def train_classical_model(X, y, groups, model, cv, param_grid=None, randomized_search=False,
                           count=10, label_encode=False):
    """Grid/randomized search over param_grid, evaluated with group-aware CV + majority vote
    per combination. Returns (results, best) where results is a list of
    (params, mean_acc, mean_f1) tuples and best is the entry with the highest mean_f1."""

    param_grid = _normalize_param_grid(model, param_grid or {})
    _validate_param_grid(model, param_grid)

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
            model.fit(X_train_fold, y_train_fold, **_sample_weight_kwarg(model, sample_weights))

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

    best = get_best_result(results)

    print("\nBest params:", best[0])
    print("Best accuracy:", best[1])
    print("Best macro F1:", best[2], "\n")

    return results, best


def nested_cv_classical(X, y, groups, model, param_grid, outer_cv, inner_cv,
                         randomized_search=False, count=10, label_encode=False):
    """Outer loop = generalization estimate, inner loop (train_classical_model) = tuning.
    Returns (fold_results, all_work_true, all_work_pred):
    fold_results is [(best_params, outer_acc, outer_f1), ...] per outer fold."""

    fold_results = []
    all_work_true = []
    all_work_pred = []

    for train_idx, test_idx in outer_cv.split(X, y, groups):
        X_train, y_train, groups_train = X[train_idx], y[train_idx], groups[train_idx]
        X_test, y_test, groups_test = X[test_idx], y[test_idx], groups[test_idx]

        _, best = train_classical_model(
            X_train, y_train, groups_train, model, inner_cv,
            param_grid=param_grid, randomized_search=randomized_search,
            count=count, label_encode=label_encode,
        )
        best_params = best[0]

        le = LabelEncoder().fit(y_train) if label_encode else None
        y_train_fit = le.transform(y_train) if label_encode else y_train

        sample_weights = make_sample_weights(y_train, groups_train)
        model.set_params(**best_params)
        model.fit(X_train, y_train_fit, **_sample_weight_kwarg(model, sample_weights))

        pred = model.predict(X_test)
        if label_encode:
            pred = le.inverse_transform(pred)

        work_true, work_pred = majority_vote_prediction(y_test, pred, groups_test)
        acc = metrics.accuracy_score(work_true, work_pred)
        f1 = metrics.f1_score(work_true, work_pred, average="macro")

        fold_results.append((best_params, acc, f1))
        all_work_true.extend(work_true)
        all_work_pred.extend(work_pred)

        print(f"outer fold: best_params={best_params} -> accuracy={acc:.4f}, macro_f1={f1:.4f}\n")

    return fold_results, all_work_true, all_work_pred


def report_metrics(all_work_true, all_work_pred):

    acc = metrics.accuracy_score(
        all_work_true,
        all_work_pred
    )

    f1 = metrics.f1_score(
        all_work_true,
        all_work_pred,
        average = "macro"
    )

    classification_report = metrics.classification_report(
        all_work_true,
        all_work_pred
    )

    return acc, f1, classification_report


def _refit_and_collect(X, y, groups, model, params, cv, label_encode):
    """Fit `model` with fixed `params` on each cv fold, collecting out-of-fold work-level
    predictions. Used for the non-nested case, where the same cv also picked the params."""
    all_work_true = []
    all_work_pred = []

    le = LabelEncoder().fit(y) if label_encode else None

    for train_idx, val_idx in cv.split(X, y, groups):
        X_train, y_train, groups_train = X[train_idx], y[train_idx], groups[train_idx]
        X_val, y_val, groups_val = X[val_idx], y[val_idx], groups[val_idx]

        y_train_fit = le.transform(y_train) if label_encode else y_train
        sample_weights = make_sample_weights(y_train, groups_train)

        model.set_params(**params)
        model.fit(X_train, y_train_fit, **_sample_weight_kwarg(model, sample_weights))

        pred = model.predict(X_val)
        if label_encode:
            pred = le.inverse_transform(pred)

        work_true, work_pred = majority_vote_prediction(y_val, pred, groups_val)
        all_work_true.extend(work_true)
        all_work_pred.extend(work_pred)

    return all_work_true, all_work_pred


def run_classical_train_pipeline(X, y, groups, model, cv, param_grid=None, outer_cv=None,
                  randomized_search=False, count=10, label_encode=False):
    """Runs hyperparameter search + evaluation, then prints the report and plots the confusion
    matrix. If outer_cv is given, runs nested CV (cv is treated as the inner search loop);
    otherwise cv is used both to pick hyperparameters and to produce the final report.

    For scaled models (e.g. SVM), pass model=make_pipeline(StandardScaler(), SVC(...)) and
    prefix param_grid keys accordingly (e.g. "svc__C") - the scaler then gets refit on each
    fold's train split automatically, never touching that fold's validation data."""

    if not param_grid:
        raise ValueError("param_grid must not be empty")

    if outer_cv is not None:
        fold_results, all_work_true, all_work_pred = nested_cv_classical(
            X, y, groups, model, param_grid, outer_cv, cv,
            randomized_search=randomized_search, count=count, label_encode=label_encode,
        )
    else:
        _, best = train_classical_model(
            X, y, groups, model, cv, param_grid=param_grid,
            randomized_search=randomized_search, count=count, label_encode=label_encode,
        )
        fold_results = [best]
        all_work_true, all_work_pred = _refit_and_collect(
            X, y, groups, model, best[0], cv, label_encode
        )

    acc, f1, report = report_metrics(all_work_true, all_work_pred)

    print(report)
    print("Overall work accuracy:", acc)
    print("Overall work macro F1:", f1)

    metrics.ConfusionMatrixDisplay.from_predictions(
        all_work_true,
        all_work_pred,
        normalize="true"
    )
    plt.show()

    return fold_results, all_work_true, all_work_pred