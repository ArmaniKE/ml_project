from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any

import numpy as np
from sklearn.neural_network import MLPRegressor

from src.config import RANDOM_STATE

try:
    import tensorflow as tf
    keras = tf.keras
    layers = tf.keras.layers
    regularizers = tf.keras.regularizers
except Exception as import_error:  
    keras = None
    layers = None
    regularizers = None
    TF_IMPORT_ERROR = import_error
else:
    TF_IMPORT_ERROR = None


@dataclass
class DeepLearningArtifacts:
    keras_model: Any
    mlp_model: MLPRegressor
    history: Dict[str, Any]


class KerasRegressionModel:
    def __init__(self, input_dim: int):
        if keras is None:
            raise RuntimeError(
                "TensorFlow is not available in this environment. "
                "Install a working TensorFlow build in the active interpreter."
            ) from TF_IMPORT_ERROR
        self.input_dim = input_dim
        self.model = self._build_model(input_dim)

    def _build_model(self, input_dim: int) -> Any:
        inputs = keras.Input(shape=(input_dim,))
        x = layers.BatchNormalization()(inputs)
        x = layers.Dense(512, activation="relu", kernel_regularizer=regularizers.l2(1e-4))(x)
        x = layers.Dropout(0.35)(x)
        x = layers.Dense(256, activation="relu", kernel_regularizer=regularizers.l2(1e-4))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.25)(x)
        x = layers.Dense(128, activation="relu", kernel_regularizer=regularizers.l2(1e-4))(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(64, activation="relu")(x)
        outputs = layers.Dense(1, activation="linear")(x)

        model = keras.Model(inputs, outputs, name="myntra_price_regressor")
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-3),
            loss="mse",
            metrics=[
                keras.metrics.MeanAbsoluteError(name="mae"),
                keras.metrics.RootMeanSquaredError(name="rmse"),
            ],
        )
        return model

    def fit(self, X_train, y_train, validation_split=0.15, epochs=200, batch_size=64, verbose=0):
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=20,
                restore_best_weights=True,
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=8,
                min_lr=1e-6,
            ),
        ]
        history = self.model.fit(
            X_train,
            y_train,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose,
            shuffle=True,
        )
        return history

    def predict(self, X):
        predictions = self.model.predict(X, verbose=0).reshape(-1)
        return predictions

    def save(self, path):
        self.model.save(path)

    @classmethod
    def load(cls, path):
        model = keras.models.load_model(path)
        wrapper = cls(input_dim=model.input_shape[-1])
        wrapper.model = model
        return wrapper


def init_mlp():
    return MLPRegressor(
        hidden_layer_sizes=(128, 64, 32),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        learning_rate="adaptive",
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=20,
        max_iter=1000,
        random_state=RANDOM_STATE,
    )


def train_mlp(model, X_train, y_train):
    print("Training MLP Neural Network...")
    model.fit(X_train, y_train)
    print("MLP trained successfully.")
    return model


def init_keras_regressor(input_dim: int) -> KerasRegressionModel:
    return KerasRegressionModel(input_dim=input_dim)


def train_keras_regressor(model: KerasRegressionModel, X_train, y_train):
    print("Training TensorFlow deep learning model...")
    history = model.fit(X_train, y_train, verbose=0)
    print("TensorFlow model trained successfully.")
    return model, history
