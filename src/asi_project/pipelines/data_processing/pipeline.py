from kedro.pipeline import Pipeline, node

from asi_project.data import normalize_input

from .nodes import filter_rows, validate_raw_data


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=validate_raw_data,
                inputs=["raw_data", "params:dataset_id"],
                outputs="validated_data",
                name="validate_raw_data_node",
            ),
            node(
                func=normalize_input,
                inputs="validated_data",
                outputs="normalized_data",
                name="normalize_input_node",
            ),
            node(
                func=filter_rows,
                inputs=["normalized_data", "params:dataset_id"],
                outputs="model_input",
                name="filter_training_rows_node",
            ),
        ]
    )