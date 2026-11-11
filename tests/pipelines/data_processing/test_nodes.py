import pandas as pd
from asi_project.pipelines.data_processing.nodes import validate_raw_data

def test_przygotowanie_danych_zwraca_oczekiwany_wynik():
    df = pd.read_csv("data/01_raw/sample.csv")
    
    result = validate_raw_data(df, "online_shoppers")
    
    assert isinstance(result, pd.DataFrame)
    assert not result.empty