# Myntra Product Price Prediction

This project trains and compares machine learning models for predicting Myntra product prices in INR from product metadata such as brand, gender category, image count, primary color, and description text.

## Project Structure

- `main.py` - runs the full training pipeline.
- `src/` - data loading, feature engineering, preprocessing, model training, evaluation, reporting, and artifact utilities.
- `app/app.py` - Streamlit app for interactive price prediction.
- `data/raw/myntra_products_catalog.csv` - source dataset.
- `models/` - trained model artifacts.
- `artifacts/preprocessor.pkl` - fitted preprocessing pipeline.
- `reports/` - generated metrics and training report.
- `notebooks/` - exploratory and phase-based project notebooks.

## Setup

From the project root:

```powershell
cd C:\Users\Arman\Desktop\ml_project
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If PowerShell blocks activation in the current session:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## Train Models

Use the virtual environment Python explicitly:

```powershell
.\.venv\Scripts\python.exe main.py
```

The pipeline:

- loads `data/raw/myntra_products_catalog.csv`
- engineers text and color features
- applies target encoding, one-hot encoding, and numeric scaling
- trains KNN, SVM, RandomForest, HistogramGradientBoosting, MLP, and TensorFlow deep regressor models
- saves models to `models/`
- saves the fitted preprocessor to `artifacts/preprocessor.pkl`
- writes metrics to `reports/test_metrics.csv`
- writes the full report to `reports/training_report.json`

## Run the App

```powershell
.\.venv\Scripts\python.exe -m streamlit run app/app.py
```

The app reads the latest training report and only shows models that have matching saved artifacts. It supports both `.pkl` scikit-learn models and the `.keras` TensorFlow model.

## Current Results

Latest test metrics from `reports/test_metrics.csv`:

| Model | MAE | RMSE | R2 |
| --- | ---: | ---: | ---: |
| RandomForest | 0.2597 | 0.3749 | 0.7181 |
| HistogramGradientBoosting | 0.2650 | 0.3776 | 0.7141 |
| TensorFlowDeepRegressor | 0.2819 | 0.3945 | 0.6879 |
| MLP | 0.2937 | 0.4072 | 0.6674 |
| SVM | 0.2929 | 0.4109 | 0.6614 |
| KNN | 0.2927 | 0.4194 | 0.6472 |

The current best model is `RandomForest`.

## Notes

- The target is modeled as `log1p(Price (INR))`; app predictions are converted back with `expm1`.
- TensorFlow may print oneDNN/CPU optimization messages during startup. These are informational messages, not failures.
- Run notebooks from the `notebooks/` directory or project root after installing dependencies. They import project code through the `src` package.
