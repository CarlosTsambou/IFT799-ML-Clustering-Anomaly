import time
import numpy as np

from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers


# =========================
# 0. Petit décorateur pour mesurer le temps
# =========================

def timed(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        return result, end - start
    return wrapper


# =========================
# 1. ISOLATION FOREST
# =========================

@timed
def run_isolation_forest(X_train, X_test):
    model = IsolationForest(
        contamination="auto",
        n_estimators=200,
        random_state=42,
    )
    model.fit(X_train)
    test_scores = model.score_samples(X_test)
    return test_scores


def evaluate_threshold(test_scores, threshold, y_true):
    y_pred = (test_scores < threshold).astype(int)

    return {
        "AUC": roc_auc_score(y_true, -test_scores),
        "F1": f1_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "confusion": confusion_matrix(y_true, y_pred),
    }


# =========================
# 2. AUTO-ENCODEUR SIMPLE
# =========================

def build_autoencoder(input_dim):
    inputs = layers.Input(shape=(input_dim,))

    x = layers.Dense(
        64,
        activation="relu",
        kernel_regularizer=regularizers.l2(5e-4),
    )(inputs)
    x = layers.Dropout(0.3)(x)
    encoded = layers.Dense(
        32,
        activation="relu",
        kernel_regularizer=regularizers.l2(5e-4),
    )(x)

    x = layers.Dense(
        64,
        activation="relu",
        kernel_regularizer=regularizers.l2(5e-4),
    )(encoded)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(input_dim)(x)

    model = models.Model(inputs, outputs)
    model.compile(optimizer="adam", loss="mse")
    return model


@timed
def train_autoencoder(
    model,
    X_train,
    X_val,
    epochs=30,
    batch_size=64,
    callbacks=None,
):
    if callbacks is None:
        callbacks = []

    history = model.fit(
        X_train,
        X_train,
        validation_data=(X_val, X_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=0,
    )
    return history


# =========================
# 3. AUTO-ENCODEUR PROFOND
# =========================

from tensorflow.keras import layers, models, regularizers

def build_deep_autoencoder(input_dim):
    inputs = layers.Input(shape=(input_dim,))

    # On ajoute du bruit gaussien pour empêcher l'overfit
    x = layers.GaussianNoise(0.1)(inputs)

    # Encoder plus petit + forte régularisation
    x = layers.Dense(
        32,
        activation="relu",
        kernel_regularizer=regularizers.l2(1e-3),
    )(x)
    x = layers.Dropout(0.4)(x)
    encoded = layers.Dense(
        16,
        activation="relu",
        kernel_regularizer=regularizers.l2(1e-3),
    )(x)

    # Decoder symétrique
    x = layers.Dense(
        32,
        activation="relu",
        kernel_regularizer=regularizers.l2(1e-3),
    )(encoded)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(input_dim)(x)   # sortie linéaire

    model = models.Model(inputs, outputs)
    model.compile(optimizer="adam", loss="mse")
    return model
