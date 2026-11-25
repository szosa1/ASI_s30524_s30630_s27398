import hashlib
import json
import os
import subprocess
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from asi_project.config import load_project_config, load_registry
from asi_project.data import raw_data_path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WANDB_PROJECT = "asi-online-shoppers"


def _git(*args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def git_sha() -> str:
    """SHA bieżącego commita; w CI bez .git: GITHUB_SHA; w ostateczności unknown."""
    return _git("rev-parse", "HEAD") or os.environ.get("GITHUB_SHA", "unknown")


def git_is_dirty() -> bool:
    """Czy w śledzonych plikach są niezacommitowane zmiany (wtedy SHA nie opisuje kodu w pełni)."""
    return bool(_git("status", "--porcelain", "--untracked-files=no"))


def dataframe_fingerprint(df: pd.DataFrame) -> str:
    """SHA-256 zawartości DataFrame (wartości, indeks i nazwy kolumn)."""
    digest = hashlib.sha256(pd.util.hash_pandas_object(df, index=True).to_numpy().tobytes())
    digest.update("|".join(map(str, df.columns)).encode("utf-8"))
    return digest.hexdigest()


def raw_file_sha256() -> str | None:
    """SHA-256 surowego pliku z ``fingerprint.json`` (zapisany przez download_data.py)."""
    config = load_project_config()
    contract = load_registry()[config.dataset_id]
    path = raw_data_path(config, contract).parent / "fingerprint.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8")).get("sha256")


def init_wandb_run(
    *,
    name: str,
    job_type: str,
    config: Mapping[str, Any],
    tags: Iterable[str] = (),
    project: str | None = None,
    entity: str | None = None,
):
    """Start runu W&B z Git SHA w configu. ``WANDB_MODE=disabled`` w CI → run bez sieci."""
    import wandb

    full_config = {**config, "git_sha": git_sha(), "git_dirty": git_is_dirty()}
    return wandb.init(
        project=os.environ.get("WANDB_PROJECT", project or DEFAULT_WANDB_PROJECT),
        entity=os.environ.get("WANDB_ENTITY", entity),
        name=name,
        job_type=job_type,
        config=full_config,
        tags=list(tags),
    )


def log_test_metrics(run, metrics: Mapping[str, Any]) -> None:
    """Metryki z ``compute_metrics`` jako ``test/<nazwa>`` (macierz pomyłek spłaszczona)."""
    flat = {f"test/{key}": value for key, value in metrics.items() if key != "confusion_matrix"}
    for key, value in metrics.get("confusion_matrix", {}).items():
        flat[f"test/cm_{key}"] = value
    run.log(flat)