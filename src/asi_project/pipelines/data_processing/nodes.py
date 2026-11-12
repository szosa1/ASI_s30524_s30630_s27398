import pandas as pd

from asi_project.config import DatasetContract, load_registry
from asi_project.data import (
    DataContractError,
    assert_valid,
    filter_training_rows,
    normalize_column_names,
)


def get_contract(dataset_id: str) -> DatasetContract:
    registry = load_registry()
    if dataset_id not in registry:
        known = ", ".join(sorted(registry))
        raise DataContractError(f"Nieznany dataset '{dataset_id}'. Dozwolone: {known}.")
    return registry[dataset_id]


def validate_raw_data(raw_data: pd.DataFrame, dataset_id: str) -> pd.DataFrame:
    df = normalize_column_names(raw_data)
    assert_valid(df, get_contract(dataset_id))
    return df


def filter_rows(normalized_data: pd.DataFrame, dataset_id: str) -> pd.DataFrame:
    return filter_training_rows(normalized_data, get_contract(dataset_id))
