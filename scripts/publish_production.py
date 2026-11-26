from pathlib import Path

import yaml

from asi_project.tracking import init_wandb_run, log_production_artifact


def main():
    params = yaml.safe_load(Path("conf/base/parameters.yml").read_text(encoding="utf-8"))

    model_name = params["production_model"]
    package_dir = Path(params["model_packages_dir"]) / model_name

    with init_wandb_run(
        name="publish-production",
        job_type="model-release",
        config={"model_name": model_name},
        project=params["wandb"]["project"],
        entity=params["wandb"]["entity"],
    ) as run:
        log_production_artifact(run, package_dir)


if __name__ == "__main__":
    main()
