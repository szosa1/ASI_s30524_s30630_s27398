import pandas as pd

from asi_project.data import make_xy, split_train_test
from asi_project.metrics import compute_metrics
from asi_project.model import build_baseline_pipeline
from asi_project.pipelines.data_processing.nodes import get_contract


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
