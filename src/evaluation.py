import json
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_validate


def evaluate_predictions(y_true, y_pred):
    """Evaluate prediction vectors and return a consolidated metric report."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return {"MAE": mae, "RMSE": rmse, "R2": r2}


def run_cross_validation(models, X, y, cv=5):
    """Run multi-metric cross-validation for a set of models."""
    scoring = ["r2", "neg_mean_absolute_error", "neg_root_mean_squared_error"]
    cv_results = {}

    for name, model in models.items():
        scores = cross_validate(
            model,
            X,
            y,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            return_train_score=False,
        )
        cv_results[name] = {
            "CV_R2_Mean": np.mean(scores["test_r2"]),
            "CV_R2_Std": np.std(scores["test_r2"]),
            "CV_MAE_Mean": -np.mean(scores["test_neg_mean_absolute_error"]),
            "CV_RMSE_Mean": -np.mean(scores["test_neg_root_mean_squared_error"]),
        }
        print(
            f"{name} CV R2: {cv_results[name]['CV_R2_Mean']:.4f} "
            f"(+/- {cv_results[name]['CV_R2_Std']:.4f}), "
            f"MAE: {cv_results[name]['CV_MAE_Mean']:.4f}, "
            f"RMSE: {cv_results[name]['CV_RMSE_Mean']:.4f}"
        )

    return cv_results


def summarize_cv_results(cv_results):
    return json.dumps(cv_results, indent=2)
