from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW_PATH = PROJECT_ROOT / "data" / "raw" / "myntra_products_catalog.csv"
MODELS_DIR = PROJECT_ROOT / "models"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
REPORTS_DIR = PROJECT_ROOT / "reports"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

RANDOM_STATE = 42
TEST_SIZE = 0.2

TARGET_COLUMN = "Price (INR)"
INPUT_COLUMNS = ["ProductBrand", "Gender", "NumImages", "PrimaryColor", "Description"]
NUMERIC_FEATURES = ["NumImages", "DescriptionLength", "BrandNameLength", "HasDescription"]
TARGET_ENCODING_COLUMNS = ["ProductBrand"]
ONE_HOT_COLUMNS = ["Gender", "PrimaryColor", "ColorGroup"]
MODEL_FEATURE_COLUMNS = INPUT_COLUMNS + ["DescriptionLength", "BrandNameLength", "HasDescription", "ColorGroup"]
