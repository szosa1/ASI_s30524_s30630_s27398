import pytest

from asi_project.data import filter_training_rows, normalize_input
from asi_project.pipelines.data_science.nodes import (
    evaluate_model,
    split_data,
    train_model,
)


@pytest.fixture
def split_parts(project):
    df = normalize_input(project.sample)
    df = filter_training_rows(df, project.contract)

    return split_data(
        df,
        project.config.dataset_id,
        list(project.config.numeric_features),
        list(project.config.categorical_features),
        project.config.test_size,
        project.config.random_state,
    )


def test_split_data(split_parts, project):
    X_train, X_test, y_train, y_test = split_parts

    assert len(X_train) > 0
    assert len(X_test) > 0
    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)
    assert set(X_train.columns) == set(project.config.model_features)
    assert set(X_train.index).isdisjoint(X_test.index)


def test_train_model(split_parts, project):
    X_train, _, y_train, _ = split_parts

    model = train_model(
        X_train,
        y_train,
        list(project.config.numeric_features),
        list(project.config.categorical_features),
    )

    assert hasattr(model, "predict")
    assert hasattr(model, "predict_proba")
    assert list(model.classes_) == [0, 1]


def test_evaluate_model(split_parts, project):
    X_train, X_test, y_train, y_test = split_parts

    model = train_model(
        X_train,
        y_train,
        list(project.config.numeric_features),
        list(project.config.categorical_features),
    )

    metrics = evaluate_model(model, X_test, y_test)

    assert 0 <= metrics["f1"] <= 1
    assert 0 <= metrics["precision"] <= 1
    assert 0 <= metrics["recall"] <= 1
    assert 0 <= metrics["accuracy"] <= 1
    assert sum(metrics["confusion_matrix"].values()) == len(y_test)
