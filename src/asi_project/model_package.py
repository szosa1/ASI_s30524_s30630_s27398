import json
import platform
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from asi_project.config import DatasetContract
from asi_project.tracking import git_is_dirty, git_sha

PACKAGE_SCHEMA_VERSION = 1
MODEL_DIR = "model"
METADATA_FILE = "metadata.json"
SKLEARN_FILE = "model.joblib"
MODEL_TYPES = ("sklearn", "autogluon")
TRACKED_LIBRARIES = (
    "scikit-learn",
    "pandas",
    "numpy",
    "joblib",
    "autogluon.tabular",
    "kedro",
    "wandb",
)
REQUIRED_KEYS = (
    "schema_version",
    "model_name",
    "model_type",
    "created_at_utc",
    "dataset",
    "target",
    "positive_class",
    "features",
    "numeric_features",
    "categorical_features",
    "threshold",
    "metrics",
    "git_sha",
    "data_fingerprint",
    "library_versions",
)


@dataclass(frozen=True)
class ModelPackage:
    path: Path
    model: Any
    metadata: dict[str, Any]

    @property
    def features(self) -> list[str]:
        return list(self.metadata["features"])


def library_versions() -> dict[str, str]:
    versions = {"python": platform.python_version()}
    for library in TRACKED_LIBRARIES:
        try:
            versions[library] = version(library)
        except PackageNotFoundError:
            continue
    return versions


def build_metadata(
    *,
    model_name: str,
    model_type: str,
    contract: DatasetContract,
    numeric_features: Sequence[str],
    categorical_features: Sequence[str],
    metrics: Mapping[str, Any],
    data_fingerprint: Mapping[str, Any],
    params: Mapping[str, Any] | None = None,
    threshold: float = 0.5,
    wandb_run_id: str | None = None,
) -> dict[str, Any]:
    if model_type not in MODEL_TYPES:
        raise ValueError(f"model_type: oczekiwano jednej z {MODEL_TYPES}, jest {model_type!r}.")
    return {
        "schema_version": PACKAGE_SCHEMA_VERSION,
        "model_name": model_name,
        "model_type": model_type,
        "created_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "dataset": contract.dataset_id,
        "target": contract.target,
        "positive_class": contract.positive_class,
        "features": [*numeric_features, *categorical_features],
        "numeric_features": list(numeric_features),
        "categorical_features": list(categorical_features),
        "threshold": float(threshold),
        "metrics": dict(metrics),
        "params": dict(params or {}),
        "git_sha": git_sha(),
        "git_dirty": git_is_dirty(),
        "data_fingerprint": dict(data_fingerprint),
        "library_versions": library_versions(),
        "wandb_run_id": wandb_run_id,
    }


def validate_metadata(metadata: Mapping[str, Any]) -> list[str]:
    """Lista problemów z ``metadata.json`` (pusta lista = OK)."""
    problems = [f"brak klucza {key!r}" for key in REQUIRED_KEYS if key not in metadata]
    if metadata.get("schema_version") not in (None, PACKAGE_SCHEMA_VERSION):
        problems.append(f"schema_version: oczekiwano {PACKAGE_SCHEMA_VERSION}")
    if "model_type" in metadata and metadata["model_type"] not in MODEL_TYPES:
        problems.append(f"model_type: oczekiwano jednej z {MODEL_TYPES}")
    if "features" in metadata and not metadata["features"]:
        problems.append("features: pusta lista cech")
    return problems


def save_model_package(model: Any, package_dir: Path | str, metadata: Mapping[str, Any]) -> Path:
    problems = validate_metadata(metadata)
    if problems:
        raise ValueError("metadata.json nie spełnia kontraktu: " + "; ".join(problems))
    package_dir = Path(package_dir)
    if package_dir.exists():
        if not (package_dir / METADATA_FILE).is_file():
            raise FileExistsError(f"{package_dir} istnieje i nie jest pakietem modelu.")
        shutil.rmtree(package_dir)
    model_dir = package_dir / MODEL_DIR
    model_dir.mkdir(parents=True)
    if metadata["model_type"] == "sklearn":
        joblib.dump(model, model_dir / SKLEARN_FILE)
    else:
        # TabularPredictor jest już zapisany po fit() w predictor.path kopiujemy
        shutil.copytree(model.path, model_dir, dirs_exist_ok=True)
    text = json.dumps(dict(metadata), indent=2, ensure_ascii=False) + "\n"
    (package_dir / METADATA_FILE).write_text(text, encoding="utf-8")
    return package_dir


def load_model_package(package_dir: Path | str) -> ModelPackage:
    package_dir = Path(package_dir)
    metadata_path = package_dir / METADATA_FILE
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Brak {METADATA_FILE} w {package_dir}.")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    problems = validate_metadata(metadata)
    if problems:
        raise ValueError("metadata.json nie spełnia kontraktu: " + "; ".join(problems))
    model_dir = package_dir / MODEL_DIR
    if metadata["model_type"] == "sklearn":
        model = joblib.load(model_dir / SKLEARN_FILE)
    else:
        from autogluon.tabular import TabularPredictor

        model = TabularPredictor.load(str(model_dir))
    return ModelPackage(path=package_dir, model=model, metadata=metadata)


def predict_frame(package: ModelPackage, X: pd.DataFrame) -> pd.DataFrame:
    """Predykcja w jednym schemacie: ``prediction`` (0/1) i ``probability`` klasy pozytywnej."""
    missing = [name for name in package.features if name not in X.columns]
    if missing:
        raise ValueError(f"Brak cech wymaganych przez model: {missing}")
    X = X.loc[:, package.features]
    if package.metadata["model_type"] == "sklearn":
        probability = package.model.predict_proba(X)[:, 1]
    else:
        probability = package.model.predict_proba(X, as_multiclass=False).to_numpy()
    prediction = (probability >= package.metadata["threshold"]).astype(int)
    return pd.DataFrame(
        {"prediction": prediction, "probability": probability.astype(float)},
        index=X.index,
    )


def package_size_bytes(package_dir: Path | str) -> int:
    return sum(path.stat().st_size for path in Path(package_dir).rglob("*") if path.is_file())