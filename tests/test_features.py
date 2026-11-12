import pandas as pd
import pytest
from src.features import FEATURES, TARGET, validate_columns

def test_validate_columns_accepts_expected_schema():
    row = {c: 0.0 for c in FEATURES}
    row[TARGET] = 0
    validate_columns(pd.DataFrame([row]))

def test_validate_columns_rejects_missing_column():
    row = {c: 0.0 for c in FEATURES}
    row[TARGET] = 0
    del row["V1"]
    with pytest.raises(ValueError):
        validate_columns(pd.DataFrame([row]))
