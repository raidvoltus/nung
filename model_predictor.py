import os
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler

lightgbm_model = None
lstm_model = None
scaler = None
feature_selector = None

def load_models():
    global lightgbm_model, lstm_model, scaler, feature_selector

    if not os.path.exists("models/lightgbm_model.pkl") or not os.path.exists("models/lstm_model.h5"):
        raise FileNotFoundError("Model belum tersedia. Harap training model terlebih dahulu.")

    with open("models/lightgbm_model.pkl", "rb") as f:
        lightgbm_model = pickle.load(f)

    lstm_model = tf.keras.models.load_model("models/lstm_model.h5")

    with open("models/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    with open("models/feature_selector.pkl", "rb") as f:
        feature_selector = pickle.load(f)

def hybrid_predict(X):
    try:
        if lightgbm_model is None or lstm_model is None:
            load_models()

        X = pd.DataFrame(X)
        lightgbm_pred = lightgbm_model.predict(X)

        selected = feature_selector.get_support()
        X_selected = X.iloc[:, selected]
        X_scaled = scaler.transform(X_selected)
        X_reshaped = X_scaled.reshape(1, 1, -1)

        lstm_pred = lstm_model.predict(X_reshaped)[0][0]
        return (lightgbm_pred + lstm_pred) / 2
    except Exception as e:
        print(f"❌ Prediksi gagal: {e}")
        return None
