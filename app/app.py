import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = ROOT_DIR / "data" / "raw" / "myntra_products_catalog.csv"
TEST_DATA_PATH = ROOT_DIR / "data" / "test" / "app_test_products.txt"
sys.path.append(str(ROOT_DIR))

from src.config import INPUT_COLUMNS, TARGET_COLUMN
from src.features import prepare_feature_matrix
from src.utils import load_artifact, load_keras_model, load_model


@st.cache_data
def load_dataset():
    if not DATASET_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(DATASET_PATH)


@st.cache_data
def load_test_products():
    if not TEST_DATA_PATH.exists():
        return pd.DataFrame()

    products = []
    current_product = {}

    for raw_line in TEST_DATA_PATH.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line == "---":
            if current_product:
                products.append(current_product)
                current_product = {}
            continue
        if ":" not in line:
            continue

        key, value = line.split(":", 1)
        current_product[key.strip()] = value.strip()

    if current_product:
        products.append(current_product)

    test_products = pd.DataFrame(products)
    if test_products.empty:
        return test_products

    test_products[TARGET_COLUMN] = pd.to_numeric(test_products[TARGET_COLUMN], errors="coerce")
    test_products["NumImages"] = pd.to_numeric(test_products["NumImages"], errors="coerce").fillna(0).astype(int)
    return test_products


def get_dataset_options():
    df = load_dataset()
    if df.empty:
        return {
            "brand": ["Unknown"],
            "gender": ["Unknown"],
            "color": ["Unknown"],
        }

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


def predict_price(model, preprocessor, input_df):
    input_transformed = preprocessor.transform(input_df)
    log_prediction = np.ravel(model.predict(input_transformed))[0]
    return float(np.expm1(log_prediction))


def get_option_index(options, value):
    normalized_value = str(value).strip()
    for index, option in enumerate(options):
        if str(option).strip() == normalized_value:
            return index
    return 0


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


def select_reference_product(dataset, brand, gender, color, num_images):
    if dataset.empty:
        return None, None

    df = dataset.copy()
    df["ProductBrand"] = df["ProductBrand"].fillna("Unknown")
    df["Gender"] = df["Gender"].fillna("Unknown")
    df["PrimaryColor"] = df["PrimaryColor"].fillna("Unknown")

    match_levels = [
        (
            (df["ProductBrand"] == brand)
            & (df["Gender"] == gender)
            & (df["PrimaryColor"] == color)
            & (df["NumImages"] == num_images),
            "Matched by brand, gender, color, and image count",
        ),
        (
            (df["ProductBrand"] == brand)
            & (df["Gender"] == gender)
            & (df["PrimaryColor"] == color),
            "Matched by brand, gender, and color",
        ),
        (
            (df["ProductBrand"] == brand)
            & (df["Gender"] == gender),
            "Matched by brand and gender",
        ),
        (
            df["ProductBrand"] == brand,
            "Matched by brand",
        ),
    ]

    candidates = pd.DataFrame()
    match_note = None
    for mask, note in match_levels:
        candidates = df[mask]
        if not candidates.empty:
            match_note = note
            break

    if candidates.empty:
        candidates = df
        match_note = "No close catalog match; using dataset representative"

    median_price = candidates[TARGET_COLUMN].median()
    reference_index = (candidates[TARGET_COLUMN] - median_price).abs().idxmin()
    return df.loc[reference_index], match_note


def get_test_product_options(test_products):
    if test_products.empty:
        return {"Manual input": None}
    options = {"Manual input": None}
    for index, row in test_products.iterrows():
        label = f"{row['TestCaseName']} | {row[TARGET_COLUMN]:.0f} INR"
        options[label] = index
    return options


st.title("Product Price Predictor")

try:
    preprocessor = get_preprocessor()
    available_models = get_available_models()
    dataset_options = get_dataset_options()
    dataset = load_dataset()
    test_products = load_test_products()

    if not available_models:
        st.warning("No trained models were found. Run the training pipeline first.")
    else:
        model_options = {model["label"]: model for model in available_models}
        selected_model_name = st.selectbox("Choose ML Model", list(model_options.keys()))

        test_product_options = get_test_product_options(test_products)
        selected_test_product = st.selectbox("Test Example", list(test_product_options.keys()))
        selected_test_index = test_product_options[selected_test_product]

        if selected_test_index is not None:
            test_row = test_products.loc[selected_test_index]
            default_brand = test_row["ProductBrand"]
            default_gender = test_row["Gender"]
            default_color = test_row["PrimaryColor"]
            default_description = test_row["Description"]
            default_num_images = int(test_row["NumImages"])
        else:
            test_row = None
            default_brand = dataset_options["brand"][0]
            default_gender = dataset_options["gender"][0]
            default_color = dataset_options["color"][0]
            default_description = ""
            default_num_images = 5

        brand = st.selectbox(
            "Product Brand",
            dataset_options["brand"],
            index=get_option_index(dataset_options["brand"], default_brand),
            key=f"brand_{selected_test_product}",
        )
        gender = st.selectbox(
            "Gender Category",
            dataset_options["gender"],
            index=get_option_index(dataset_options["gender"], default_gender),
            key=f"gender_{selected_test_product}",
        )
        color = st.selectbox(
            "Primary Color",
            dataset_options["color"],
            index=get_option_index(dataset_options["color"], default_color),
            key=f"color_{selected_test_product}",
        )
        description = st.text_area(
            "Product Description",
            default_description,
            key=f"description_{selected_test_product}",
        )
        num_images = st.number_input(
            "Number of Images",
            min_value=0,
            max_value=20,
            value=default_num_images,
            key=f"num_images_{selected_test_product}",
        )

        if st.button("Predict Price"):
            input_df = build_input_dataframe(brand, gender, color, num_images, description)
            model = load_selected_model(model_options[selected_model_name])
            final_price = predict_price(model, preprocessor, input_df)
            st.success(f"Estimated Price: {final_price:.2f} INR")

            if test_row is not None:
                reference_row = test_row
                match_note = "Compared with selected test product"
            else:
                reference_row, match_note = select_reference_product(
                    dataset,
                    brand,
                    gender,
                    color,
                    num_images,
                )

            if reference_row is not None:
                actual_price = float(reference_row[TARGET_COLUMN])
                difference = final_price - actual_price
                absolute_error = abs(difference)
                percentage_error = absolute_error / actual_price * 100 if actual_price else 0

                st.caption(match_note)
                st.write(f"Reference product: {reference_row.get('ProductName', 'Unknown product')}")

                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Real Price", f"{actual_price:.2f} INR")
                col2.metric("Predicted", f"{final_price:.2f} INR")
                col3.metric("Difference", f"{difference:+.2f} INR")
                col4.metric("Error", f"{percentage_error:.2f}%")
            else:
                st.info("No dataset product is available for real-price comparison.")

except Exception as error:
    st.error(f"Initialization error: {error}")
