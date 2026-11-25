import pandas as pd
import pytest

from asi_project.tracking import dataframe_fingerprint, git_sha, init_wandb_run, log_test_metrics


def test_dataframe_fingerprint_is_deterministic_and_sensitive():
    df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    assert dataframe_fingerprint(df) == dataframe_fingerprint(df.copy())
    changed = df.copy()
    changed.loc[0, "a"] = 3
    assert dataframe_fingerprint(df) != dataframe_fingerprint(changed)


def test_git_sha_format():
    sha = git_sha()
    assert sha == "unknown" or len(sha) == 40


def test_wandb_run_works_offline(monkeypatch):
    pytest.importorskip("wandb")
    monkeypatch.setenv("WANDB_MODE", "disabled")
    run = init_wandb_run(name="test", job_type="test", config={"x": 1})
    with run:
        log_test_metrics(run, {"f1": 0.5, "confusion_matrix": {"tn": 1, "fp": 0, "fn": 0, "tp": 1}})
