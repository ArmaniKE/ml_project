from datetime import datetime
from pathlib import Path

from src.preprocessing import clean_and_prepare_data
from src.models import init_models, train_models
from src.deep_learning import init_mlp, train_mlp, init_keras_regressor, train_keras_regressor
from src.evaluation import evaluate_predictions, run_cross_validation
from src.reporting import create_enterprise_comparison_report, export_metrics_csv
from src.utils import save_model, save_artifact, save_report, save_keras_model


def run_training_pipeline(file_path=None):
    X_train, X_test, y_train, y_test, preprocessor = clean_and_prepare_data(file_path, return_pipeline=True)

    baseline_models = init_models()
    cv_results = run_cross_validation(baseline_models, X_train, y_train, cv=5)
    trained_models = train_models(baseline_models, X_train, y_train)

    deep_learning_results = {}

    mlp_model = init_mlp()
    mlp_trained = train_mlp(mlp_model, X_train, y_train)
    trained_models["MLP"] = mlp_trained

    keras_model = init_keras_regressor(input_dim=X_train.shape[1])
    keras_trained, keras_history = train_keras_regressor(keras_model, X_train, y_train)
    trained_models["TensorFlowDeepRegressor"] = keras_trained
    deep_learning_results["TensorFlowDeepRegressor"] = {
        "epochs": len(keras_history.history.get("loss", [])),
        "final_train_loss": float(keras_history.history["loss"][-1]),
        "final_val_loss": float(keras_history.history["val_loss"][-1]),
    }

    test_results = {}
    for name, model in trained_models.items():
        y_pred = model.predict(X_test)
        metrics = evaluate_predictions(y_test, y_pred)
        test_results[name] = metrics
        if name == "TensorFlowDeepRegressor":
            save_keras_model(model, name.lower())
        else:
            save_model(model, name.lower())

    save_artifact(preprocessor, "preprocessor")

    comparison_report = create_enterprise_comparison_report(cv_results, test_results, dl_history=deep_learning_results)
    export_metrics_csv(test_results, Path("reports") / "test_metrics.csv")

    historic_report = {
        "created_at": datetime.utcnow().isoformat() + "Z",
        "cv_results": cv_results,
        "test_metrics": test_results,
        "deep_learning": deep_learning_results,
        "summary": comparison_report,
    }
    save_report(historic_report, "training_report.json")

    return trained_models, preprocessor, historic_report
