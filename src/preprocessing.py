import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, TargetEncoder
from sklearn.model_selection import train_test_split

from src.config import (
    INPUT_COLUMNS,
    MODEL_FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    TARGET_ENCODING_COLUMNS,
    ONE_HOT_COLUMNS,
    TARGET_COLUMN,
    RANDOM_STATE,
    TEST_SIZE,
)
from src.data_loader import load_raw_data
from src.features import prepare_feature_matrix


def build_preprocessing_pipeline():
    numeric_pipeline = Pipeline([
        ("scaler", StandardScaler()),
    ])

    transformer = ColumnTransformer([
        ("brand_target_encoding", TargetEncoder(smooth="auto", random_state=RANDOM_STATE), TARGET_ENCODING_COLUMNS),
        (
            "onehot_encoding",
            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ONE_HOT_COLUMNS,
        ),
        ("numeric_scaling", numeric_pipeline, NUMERIC_FEATURES),
    ], remainder="drop")

    pipeline = Pipeline([
        ("transformer", transformer),
    ])
    return pipeline


def clean_and_prepare_data(file_path=None, return_pipeline=False):
    df = load_raw_data(file_path)
    df = prepare_feature_matrix(df)

    X = df[MODEL_FEATURE_COLUMNS].copy()
    y = np.log1p(df[TARGET_COLUMN])

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    pipeline = build_preprocessing_pipeline()
    X_train_prepared = pipeline.fit_transform(X_train, y_train)
    X_test_prepared = pipeline.transform(X_test)

    print(
        f"Preprocessing finished. Train shape: {X_train_prepared.shape}, Test shape: {X_test_prepared.shape}"
    )

    if return_pipeline:
        return X_train_prepared, X_test_prepared, y_train, y_test, pipeline

    return X_train_prepared, X_test_prepared, y_train, y_test


if __name__ == "__main__":
    print("Running preprocessing execution...")
    X_train, X_test, y_train, y_test, pipeline = clean_and_prepare_data(return_pipeline=True)
    print("Execution completed successfully.")
