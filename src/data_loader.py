from pathlib import Path
import pandas as pd

from src.config import DATA_RAW_PATH, TARGET_COLUMN, INPUT_COLUMNS


def load_raw_data(file_path=None):
    source_path = Path(file_path) if file_path is not None else DATA_RAW_PATH
    source_path = source_path if source_path.is_absolute() else source_path.resolve()

    if not source_path.exists():
        raise FileNotFoundError(f"Missing dataset at: {source_path}")

    df = pd.read_csv(source_path)
    missing_columns = set(INPUT_COLUMNS + [TARGET_COLUMN]) - set(df.columns)
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing_columns)}")

    print(f"Data loaded from {source_path}. Shape: {df.shape}")
    return df
