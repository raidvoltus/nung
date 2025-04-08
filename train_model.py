import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from lightgbm import LGBMClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import joblib

from data_fetcher import fetch_all_stock_data
from model_predictor import create_features, create_lstm_sequences

# --- Cek & Ambil Data Training ---
os.makedirs("data", exist_ok=True)
data_path = "data/training_data.csv"

if not os.path.exists(data_path):
    print("⚠️ File training_data.csv tidak ditemukan. Mengambil data historis...")
    df = fetch_all_stock_data(["BBRI.JK", "BBCA.JK", "TLKM.JK", "ASII.JK", "BMRI.JK"])
    df.to_csv(data_path, index=False)
    print("✅ Data training disimpan ke:", data_path)
else:
    print("✅ File training_data.csv ditemukan.")
    df = pd.read_csv(data_path)

# --- Buat Fitur & Target untuk LightGBM ---
df_features = create_features(df)
X = df_features.drop(columns=["Target"])
y = df_features["Target"]

# --- Training LightGBM ---
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
model_lgbm = LGBMClassifier()
model_lgbm.fit(X_train, y_train)

# Simpan model LightGBM
os.makedirs("models", exist_ok=True)
joblib.dump(model_lgbm, "models/lightgbm_model.pkl")
print("✅ Model LightGBM disimpan ke: models/lightgbm_model.pkl")

# --- Training LSTM ---
scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(X)

X_lstm, y_lstm = create_lstm_sequences(scaled_features, y.values)

X_train_lstm, X_val_lstm, y_train_lstm, y_val_lstm = train_test_split(X_lstm, y_lstm, test_size=0.2, random_state=42)

model_lstm = Sequential([
    LSTM(64, return_sequences=False, input_shape=(X_train_lstm.shape[1], X_train_lstm.shape[2])),
    Dense(1, activation='sigmoid')
])

model_lstm.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model_lstm.fit(X_train_lstm, y_train_lstm, epochs=50, batch_size=32, validation_data=(X_val_lstm, y_val_lstm), verbose=1)

# Simpan model LSTM
model_lstm.save("models/lstm_model.h5")
print("✅ Model LSTM disimpan ke: models/lstm_model.h5")
