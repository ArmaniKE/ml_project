from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor

from src.config import RANDOM_STATE


def init_models():
    """Initialize regression models with tuned settings for a broader benchmark."""
    models = {
        "KNN": KNeighborsRegressor(n_neighbors=7, weights="distance"),
        "SVM": SVR(C=1.5, epsilon=0.2, kernel="rbf"),
        "RandomForest": RandomForestRegressor(
            n_estimators=150,
            max_depth=18,
            min_samples_leaf=3,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "HistogramGradientBoosting": HistGradientBoostingRegressor(
            max_iter=300,
            learning_rate=0.1,
            random_state=RANDOM_STATE,
        ),
    }
    return models


def train_models(models, X_train, y_train):
    trained_models = {}
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        trained_models[name] = model
    print("All models trained successfully.")
    return trained_models
