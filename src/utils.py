from pathlib import Path
import json
import joblib

from src.config import MODELS_DIR, ARTIFACTS_DIR, REPORTS_DIR, PROCESSED_DIR


def ensure_directories():
    directories = [MODELS_DIR, ARTIFACTS_DIR, REPORTS_DIR, PROCESSED_DIR]
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


def save_model(model, name):
    ensure_directories()
    target_path = MODELS_DIR / f"{name}.pkl"
    joblib.dump(model, target_path)
    print(f"Model saved to {target_path}")


def load_model(name):
    target_path = MODELS_DIR / f"{name}.pkl"
    if not target_path.exists():
        raise FileNotFoundError(f"No model found at {target_path}")
    return joblib.load(target_path)


def load_keras_model(name):
    target_path = MODELS_DIR / f"{name}.keras"
    if not target_path.exists():
        raise FileNotFoundError(f"No TensorFlow model found at {target_path}")

    import importlib

    try:
        tensorflow = importlib.import_module("tensorflow")
    except ImportError as exc:
        raise ImportError("TensorFlow is required to load Keras models.") from exc

    return tensorflow.keras.models.load_model(target_path)


def save_keras_model(model, name):
    ensure_directories()
    target_path = MODELS_DIR / f"{name}.keras"
    model.save(target_path)
    print(f"TensorFlow model saved to {target_path}")


def save_artifact(obj, name):
    ensure_directories()
    target_path = ARTIFACTS_DIR / f"{name}.pkl"
    joblib.dump(obj, target_path)
    print(f"Artifact saved to {target_path}")


def load_artifact(name):
    target_path = ARTIFACTS_DIR / f"{name}.pkl"
    if not target_path.exists():
        raise FileNotFoundError(f"No artifact found at {target_path}")
    return joblib.load(target_path)


def save_report(report, filename):
    ensure_directories()
    target_path = REPORTS_DIR / filename
    with open(target_path, "w", encoding="utf-8") as fp:
        json.dump(report, fp, indent=2)
    print(f"Report saved to {target_path}")
