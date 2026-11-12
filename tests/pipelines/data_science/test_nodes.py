import pandas as pd
from sklearn.dummy import DummyClassifier

from asi_project.pipelines.data_science.nodes import (
    evaluate_model,
    split_data,
    train_model,
)


def test_split_poprawna_strukture():
    df = pd.read_csv("data/01_raw/sample.csv")

    X_train, X_test, y_train, y_test = split_data(
        df,
        dataset_id="online_shoppers",
        numeric_features=["Administrative"],
        categorical_features=["VisitorType"],
        test_size=0.2,
        random_state=42,
    )

    assert len(X_train) > 0
    assert len(X_test) > 0
    assert len(X_train) + len(X_test) == len(df)


def test_treningowy_zwraca_model():
    X_train = pd.DataFrame(
        {"Administrative": [1, 2], "VisitorType": ["Returning_Visitor", "New_Visitor"]}
    )
    y_train = pd.Series([True, False], name="Revenue")

    model = train_model(
        X_train,
        y_train,
        numeric_features=["Administrative"],
        categorical_features=["VisitorType"],
    )

    assert hasattr(model, "predict")


def test_ewaluacyjny_zwraca_oczekiwane_metryki():
    X_test = pd.DataFrame({"Administrative": [1, 2]})
    y_test = pd.Series([True, False], name="Revenue")

    dummy = DummyClassifier(strategy="constant", constant=True)
    dummy.fit(X_test, y_test)

    metrics = evaluate_model(dummy, X_test, y_test)

    assert isinstance(metrics, dict)
    assert "f1" in metrics
