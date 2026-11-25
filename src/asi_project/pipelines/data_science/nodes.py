from pathlib import Path

import pandas as pd

from asi_project.data import make_xy, split_train_test
from asi_project.metrics import compute_metrics
from asi_project.model import build_baseline_pipeline
from asi_project.model_package import build_metadata, save_model_package
from asi_project.pipelines.data_processing.nodes import get_contract
from asi_project.tracking import (
    dataframe_fingerprint,
    init_wandb_run,
    log_test_metrics,
    raw_file_sha256,
)


def split_data(
    model_input: pd.DataFrame,
    dataset_id: str,
    numeric_features: list[str],
    categorical_features: list[str],
    test_size: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:

    contract = get_contract(dataset_id)

    features = numeric_features + categorical_features

    X, y = make_xy(model_input, contract, features)

    return split_train_test(X, y, contract, test_size, random_state)


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    numeric_features: list[str],
    categorical_features: list[str],
):
    model = build_baseline_pipeline(
        numeric_features,
        categorical_features,
    )

    model.fit(X_train, y_train)

    return model


def evaluate_model(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict:

    y_pred = model.predict(X_test)

    metrics = compute_metrics(y_test, y_pred)

    return metrics


BASELINE_NAME = "baseline_logreg"
BASELINE_PARAMS = {"model": "LogisticRegression", "class_weight": "balanced", "max_iter": 1000}


def compute_data_fingerprint(model_input: pd.DataFrame) -> dict:
    """Identyfikator wersji danych: hash danych wejściowych modelu + SHA-256 surowego pliku."""
    return {
        "model_input_sha256": dataframe_fingerprint(model_input),
        "raw_file_sha256": raw_file_sha256(),
        "n_rows": int(len(model_input)),
    }


def log_baseline_run(
    metrics: dict,
    data_fingerprint: dict,
    dataset_id: str,
    test_size: float,
    random_state: int,
    numeric_features: list[str],
    categorical_features: list[str],
    wandb_params: dict,
) -> str:
    config = {
        "model_name": BASELINE_NAME,
        **BASELINE_PARAMS,
        "dataset": dataset_id,
        "test_size": test_size,
        "random_state": random_state,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "data_fingerprint": data_fingerprint,
    }
    run = init_wandb_run(
        name=BASELINE_NAME,
        job_type="baseline",
        config=config,
        tags=["s3", "baseline"],
        project=wandb_params.get("project"),
        entity=wandb_params.get("entity"),
    )
    with run:
        log_test_metrics(run, metrics)
        return str(getattr(run, "id", "") or "")


def package_baseline(
    model,
    metrics: dict,
    data_fingerprint: dict,
    wandb_run_id: str,
    dataset_id: str,
    numeric_features: list[str],
    categorical_features: list[str],
    model_packages_dir: str,
) -> dict:
    metadata = build_metadata(
        model_name=BASELINE_NAME,
        model_type="sklearn",
        contract=get_contract(dataset_id),
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        metrics=metrics,
        data_fingerprint=data_fingerprint,
        params=BASELINE_PARAMS,
        wandb_run_id=wandb_run_id or None,
    )
    save_model_package(model, Path(model_packages_dir) / BASELINE_NAME, metadata)
    return metadata
