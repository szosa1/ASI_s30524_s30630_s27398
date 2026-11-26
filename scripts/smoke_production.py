from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
import yaml

from asi_project.data import normalize_input
from asi_project.model_package import load_model_package, predict_frame


def main():
    import wandb

    params = yaml.safe_load(Path("conf/base/parameters.yml").read_text(encoding="utf-8"))

    artifact_name = (
        f"{params['wandb']['entity']}/{params['wandb']['project']}/online-shoppers-model:production"
    )

    api = wandb.Api()
    artifact = api.artifact(artifact_name, type="model")

    with TemporaryDirectory() as temp_dir:
        package_dir = artifact.download(root=temp_dir)
        package = load_model_package(package_dir)

        print("Wczytano model:", package.metadata["model_name"])

        data = pd.read_csv("data/01_raw/sample.csv").head(5)
        data = normalize_input(data)

        result = predict_frame(package, data)

        assert len(result) == 5
        assert list(result.columns) == ["prediction", "probability"]
        assert result["prediction"].isin([0, 1]).all()
        assert result["probability"].between(0, 1).all()
        assert result.notna().all().all()

        print(result)
        print("Smoke test: PASS")


if __name__ == "__main__":
    main()
