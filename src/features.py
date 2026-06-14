import re
import pandas as pd

BASIC_COLOR_GROUPS = {
    "neutral": ["black", "white", "gray", "grey", "beige", "ivory", "cream", "brown"],
    "warm": ["red", "orange", "yellow", "maroon", "gold", "peach", "copper"],
    "cool": ["blue", "green", "teal", "aqua", "navy", "cyan"],
    "pastel": ["pink", "lavender", "mauve", "cream"],
    "metallic": ["silver", "bronze", "metallic"],
}

DEFAULT_COLOR_GROUP = "other"


def infer_color_group(color_value: str) -> str:
    if not isinstance(color_value, str):
        return DEFAULT_COLOR_GROUP
    normalized = re.sub(r"[^a-zA-Z]", " ", color_value or "").lower().strip()
    for group, keywords in BASIC_COLOR_GROUPS.items():
        for word in keywords:
            if word in normalized.split():
                return group
    return DEFAULT_COLOR_GROUP


def infer_brand_length(brand_value: str) -> int:
    return len(brand_value.strip()) if isinstance(brand_value, str) else 0


def build_text_features(df: pd.DataFrame) -> pd.DataFrame:
    output = df.copy()
    output["Description"] = output["Description"].fillna("")
    output["DescriptionLength"] = output["Description"].astype(str).str.len()
    output["HasDescription"] = (output["DescriptionLength"] > 0).astype(int)
    output["BrandNameLength"] = output["ProductBrand"].fillna("Unknown").astype(str).apply(infer_brand_length)
    output["ColorGroup"] = output["PrimaryColor"].fillna("Unknown").astype(str).apply(infer_color_group)
    return output


def prepare_feature_matrix(df: pd.DataFrame):
    frame = df.copy()
    frame["PrimaryColor"] = frame["PrimaryColor"].fillna("Unknown")
    frame["Gender"] = frame["Gender"].fillna("Unknown")
    frame["ProductBrand"] = frame["ProductBrand"].fillna("Unknown")
    frame = build_text_features(frame)
    return frame
