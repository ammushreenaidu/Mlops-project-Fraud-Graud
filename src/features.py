FEATURES = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
TARGET = "Class"

def validate_columns(df):
    required = set(FEATURES + [TARGET])
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    if df[FEATURES + [TARGET]].isnull().any().any():
        raise ValueError("Required columns contain missing values.")
    if not set(df[TARGET].unique()).issubset({0, 1}):
        raise ValueError("Class must contain only 0 and 1.")
