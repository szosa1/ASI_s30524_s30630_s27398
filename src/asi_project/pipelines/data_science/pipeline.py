from kedro.pipeline import Pipeline, node

from .nodes import (
    compute_data_fingerprint,
    evaluate_model,
    log_baseline_run,
    package_baseline,
    split_data,
    train_model,
)


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=split_data,
                inputs=[
                    "model_input",
                    "params:dataset_id",
                    "params:numeric_features",
                    "params:categorical_features",
                    "params:test_size",
                    "params:random_state",
                ],
                outputs=["X_train", "X_test", "y_train", "y_test"],
                name="split_data_node",
            ),
            node(
                func=train_model,
                inputs=[
                    "X_train",
                    "y_train",
                    "params:numeric_features",
                    "params:categorical_features",
                ],
                outputs="model",
                name="train_model_node",
            ),
            node(
                func=evaluate_model,
                inputs=["model", "X_test", "y_test"],
                outputs="metrics",
                name="evaluate_model_node",
            ),
            node(
                func=compute_data_fingerprint,
                inputs="model_input",
                outputs="data_fingerprint",
                name="data_fingerprint_node",
            ),
            node(
                func=log_baseline_run,
                inputs=[
                    "metrics",
                    "data_fingerprint",
                    "params:dataset_id",
                    "params:test_size",
                    "params:random_state",
                    "params:numeric_features",
                    "params:categorical_features",
                    "params:wandb",
                ],
                outputs="baseline_wandb_run_id",
                name="log_baseline_wandb_node",
            ),
            node(
                func=package_baseline,
                inputs=[
                    "model",
                    "metrics",
                    "data_fingerprint",
                    "baseline_wandb_run_id",
                    "params:dataset_id",
                    "params:numeric_features",
                    "params:categorical_features",
                    "params:model_packages_dir",
                ],
                outputs="baseline_package_metadata",
                name="package_baseline_node",
            ),
        ]
    )
