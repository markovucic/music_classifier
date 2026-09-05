import hashlib
import json
import random
from itertools import product

import joblib
import numpy as np
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from src.model.evaluation import majority_vote_prediction, make_sample_weights
from src.utils.paths import OUTPUT_DIR


def _sample_weight_kwarg(model, sample_weights):
    if isinstance(model, Pipeline):
        return {f"{model.steps[-1][0]}__sample_weight": sample_weights}
    return {"sample_weight": sample_weights}


def _normalize_param_grid(model, param_grid):
    if not isinstance(model, Pipeline):
        return param_grid

    valid = set(model.get_params())
    last_step = model.steps[-1][0]

    return {
        key if key in valid else f"{last_step}__{key}": vals
        for key, vals in param_grid.items()
    }


def _validate_param_grid(model, param_grid):
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
    """Among results within thresholds of the best accuracy AND best macro F1, return the one with
    the highest macro F1. Guards against picking hyperparameters that only won by CV noise;
    progressively relaxes thresholds until at least one candidate qualifies."""
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
    """Nested cross-validation - because of small dataset training size.
    Outer loop = generalization estimate, inner loop (train_classical_model) = tuning.
    Returns (fold_results, all_work_true, all_work_pred):
    fold_results is [(best_params, outer_acc, outer_f1), ...] per outer fold."""

    fold_results = []
    all_work_true = []
    all_work_pred = []

    for i, (train_idx, test_idx) in enumerate(outer_cv.split(X, y, groups), start=1):
        print(f"Fold {i}:\n")
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
        print(f"==============================================================================\n")

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


def save_classical_model(model_name, model, X, y, groups, cv, param_grid, fold_results,
                          acc, f1, report, confusion_matrix_fig, randomized_search=False,
                          count=10, label_encode=False):
    """Refits on the FULL dataset to get one deployable model - nested CV's fold_results are
    per-outer-fold best_params (possibly different each fold), not a single answer, so tuning
    is redone once more on everything if a param_grid was given. Saves that model plus this
    run's metrics/plot under output/{model_name}_{hash}/."""

    if param_grid:
        _, best = train_classical_model(
            X, y, groups, model, cv, param_grid=param_grid,
            randomized_search=randomized_search, count=count, label_encode=label_encode,
        )
        best_params = best[0]
    else:
        best_params = {}

    le = LabelEncoder().fit(y) if label_encode else None
    y_fit = le.transform(y) if label_encode else y
    sample_weights = make_sample_weights(y, groups)

    model.set_params(**best_params)
    model.fit(X, y_fit, **_sample_weight_kwarg(model, sample_weights))

    run_key = {
        "model": type(model).__name__,
        "param_grid": {k: list(v) for k, v in (param_grid or {}).items()},
        "cv_splits": getattr(cv, "n_splits", None),
    }
    run_hash = hashlib.sha1(json.dumps(run_key, sort_keys=True, default=str).encode()).hexdigest()[:8]
    run_dir = OUTPUT_DIR / f"{model_name}_{run_hash}"
    (run_dir / "plots").mkdir(parents=True, exist_ok=True)

    config = {**run_key, "best_params": best_params, "label_encode": label_encode}
    with open(run_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2, default=str)

    metrics_out = {
        "accuracy": acc,
        "macro_f1": f1,
        "classification_report": report,
        "fold_results": [
            {"best_params": p, "accuracy": a, "macro_f1": fs} for p, a, fs in fold_results
        ],
    }
    with open(run_dir / "metrics.json", "w") as f:
        json.dump(metrics_out, f, indent=2, default=str)

    joblib.dump(model, run_dir / "model.joblib")
    confusion_matrix_fig.savefig(run_dir / "plots" / "confusion_matrix.png")

    print(f"saved to {run_dir}")
    return str(run_dir)


def run_classical_train_pipeline(model_name, X, y, groups, model, cv, param_grid=None, outer_cv=None,
                  randomized_search=False, count=10, label_encode=False, save_model=True):
    """Full classical model training pipeline.
    Evaluates the model with cross-validation. If supplied - runs hyperparameter search,
    in that case performing nested cross-validation, then prints the report and plots the confusion
    matrix.

    To perform hyperparameter search, both outer_cv and param_grid must be supplied, otherwise
    error is raised.

    For scaled models (e.g. SVM), pass model=make_pipeline(StandardScaler(), SVC(...)) -
    the scaler then gets refit on each fold's train split automatically,
    never touching that fold's validation data.

    If save_model=True, refits on the full dataset and saves the model + metrics + confusion
    matrix under output/{model_name}_{hash}/ (see save_classical_model)."""

    if outer_cv is None and param_grid is not None:
        raise ValueError("outer_cv must be supplied when performing hyperparameter tuning")

    if outer_cv is not None and param_grid is None:
        raise ValueError("param_grid must not be empty when performing hyperparameter tuning")

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
    print(f"Overall work accuracy: {acc:.4f}")
    print(f"Overall work macro F1: {f1:.4f}")

    disp = metrics.ConfusionMatrixDisplay.from_predictions(
        all_work_true,
        all_work_pred,
        normalize="true"
    )
    plt.show()

    run_dir = None
    if save_model:
        run_dir = save_classical_model(
            model_name, model, X, y, groups, cv, param_grid, fold_results, acc, f1, report,
            disp.figure_, randomized_search=randomized_search, count=count, label_encode=label_encode,
        )

    return fold_results, all_work_true, all_work_pred, run_dir