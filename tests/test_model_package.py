import json
import subprocess
import sys

import pytest

from asi_project.data import filter_training_rows, normalize_input
from asi_project.model_package import (
    METADATA_FILE,
    MODEL_DIR,
    REQUIRED_KEYS,
    build_metadata,
    load_model_package,
    package_size_bytes,
    predict_frame,
    save_model_package,
    validate_metadata,
)
from asi_project.pipelines.data_science.nodes import (
    compute_data_fingerprint,
    evaluate_model,
    split_data,
    train_model,
)


@pytest.fixture
def trained(project):
    df = filter_training_rows(normalize_input(project.sample), project.contract)
    numeric = list(project.config.numeric_features)
    categorical = list(project.config.categorical_features)
    X_train, X_test, y_train, y_test = split_data(
        df,
        project.config.dataset_id,
        numeric,
        categorical,
        project.config.test_size,
        project.config.random_state,
    )
    model = train_model(X_train, y_train, numeric, categorical)
    metadata = build_metadata(
        model_name="test_baseline",
        model_type="sklearn",
        contract=project.contract,
        numeric_features=numeric,
        categorical_features=categorical,
        metrics=evaluate_model(model, X_test, y_test),
        data_fingerprint=compute_data_fingerprint(df),
    )
    return model, metadata, X_test


def test_package_structure_and_metadata(trained, tmp_path, project):
    model, metadata, _ = trained
    package_dir = save_model_package(model, tmp_path / "pkg", metadata)

    assert (package_dir / MODEL_DIR).is_dir()
    saved = json.loads((package_dir / METADATA_FILE).read_text(encoding="utf-8"))
    assert set(REQUIRED_KEYS) <= set(saved)
    assert saved["target"] == project.contract.target
    assert saved["positive_class"] == project.contract.positive_class
    assert saved["features"] == list(project.config.model_features)
    assert saved["git_sha"]
    assert "scikit-learn" in saved["library_versions"]
    assert package_size_bytes(package_dir) > 0


def test_roundtrip_gives_same_predictions(trained, tmp_path):
    model, metadata, X_test = trained
    package = load_model_package(save_model_package(model, tmp_path / "pkg", metadata))
    result = predict_frame(package, X_test)

    assert list(result.columns) == ["prediction", "probability"]
    assert result["prediction"].isin([0, 1]).all()
    assert result["probability"].between(0, 1).all()
    assert (result["probability"].to_numpy() == model.predict_proba(X_test)[:, 1]).all()


def test_package_loads_in_clean_process(trained, tmp_path):
    """Model da się wczytać w osobnym procesie, bez treningu (wymóg Sprintu 3)."""
    model, metadata, X_test = trained
    package_dir = save_model_package(model, tmp_path / "pkg", metadata)
    rows = tmp_path / "rows.csv"
    X_test.head(5).to_csv(rows, index=False)
    code = (
        "import sys, pandas as pd\n"
        "from asi_project.model_package import load_model_package, predict_frame\n"
        "package = load_model_package(sys.argv[1])\n"
        "print(predict_frame(package, pd.read_csv(sys.argv[2])).to_json(orient='records'))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code, str(package_dir), str(rows)],
        capture_output=True,
        text=True,
        check=True,
    )
    records = json.loads(result.stdout)
    assert len(records) == 5
    assert all(set(record) == {"prediction", "probability"} for record in records)


def test_validate_metadata_reports_missing_keys():
    assert validate_metadata({}) != []


def test_load_without_metadata_fails(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_model_package(tmp_path)


def test_predict_frame_requires_all_features(trained, tmp_path):
    model, metadata, X_test = trained
    package = load_model_package(save_model_package(model, tmp_path / "pkg", metadata))
    with pytest.raises(ValueError):
        predict_frame(package, X_test.drop(columns=package.features[0]))
