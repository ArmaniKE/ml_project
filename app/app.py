import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from src.config import INPUT_COLUMNS
from src.features import prepare_feature_matrix
from src.utils import load_artifact, load_keras_model, load_model


def get_dataset_options():
    dataset_path = ROOT_DIR / "data" / "raw" / "myntra_products_catalog.csv"
    if not dataset_path.exists():
        return {
            "brand": ["Unknown"],
            "gender": ["Unknown"],
            "color": ["Unknown"],
        }

    df = pd.read_csv(dataset_path)
    df["PrimaryColor"] = df["PrimaryColor"].fillna("Unknown")

    return {
        "brand": sorted(df["ProductBrand"].fillna("Unknown").unique()),
        "gender": sorted(df["Gender"].fillna("Unknown").unique()),
        "color": sorted(df["PrimaryColor"].unique()),
    }


def get_preprocessor():
    return load_artifact("preprocessor")


def get_available_models():
    model_dir = ROOT_DIR / "models"
    if not model_dir.exists():
        return []

    report_path = ROOT_DIR / "reports" / "training_report.json"
    if report_path.exists():
        with report_path.open("r", encoding="utf-8") as fp:
            report = json.load(fp)
        model_names = list(report.get("test_metrics", {}).keys())
    else:
        model_names = [path.stem for path in model_dir.glob("*.pkl")]

    available_models = []
    for model_name in model_names:
        model_stem = model_name.lower()
        pkl_path = model_dir / f"{model_stem}.pkl"
        keras_path = model_dir / f"{model_stem}.keras"

        if pkl_path.exists():
            available_models.append({
                "label": model_name,
                "stem": model_stem,
                "kind": "pickle",
            })
        elif keras_path.exists():
            available_models.append({
                "label": model_name,
                "stem": model_stem,
                "kind": "keras",
            })

    return available_models


def load_selected_model(model_info):
    if model_info["kind"] == "keras":
        return load_keras_model(model_info["stem"])
    return load_model(model_info["stem"])


def build_input_dataframe(brand, gender, color, num_images, description):
    raw_input = pd.DataFrame([
        {
            "ProductBrand": brand,
            "Gender": gender,
            "NumImages": num_images,
            "PrimaryColor": color,
            "Description": description or "",
        }
    ])[INPUT_COLUMNS]
    return prepare_feature_matrix(raw_input)


st.title("Product Price Predictor")

try:
    preprocessor = get_preprocessor()
    available_models = get_available_models()
    dataset_options = get_dataset_options()

    if not available_models:
        st.warning("No trained models were found. Run the training pipeline first.")
    else:
        model_options = {model["label"]: model for model in available_models}
        selected_model_name = st.selectbox("Choose ML Model", list(model_options.keys()))
        brand = st.selectbox("Product Brand", dataset_options["brand"])
        gender = st.selectbox("Gender Category", dataset_options["gender"])
        color = st.selectbox("Primary Color", dataset_options["color"])
        description = st.text_area("Product Description", "")
        num_images = st.number_input("Number of Images", min_value=0, max_value=20, value=5)

        if st.button("Predict Price"):
            input_df = build_input_dataframe(brand, gender, color, num_images, description)
            input_transformed = preprocessor.transform(input_df)
            model = load_selected_model(model_options[selected_model_name])
            log_prediction = model.predict(input_transformed)[0]
            final_price = np.expm1(log_prediction)
            st.success(f"Estimated Price: {final_price:.2f} INR")

except Exception as error:
    st.error(f"Initialization error: {error}")
