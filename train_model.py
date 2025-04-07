import os
import pandas as pd
import numpy as np
import lightgbm as lgb
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import joblib
import logging

logging.basicConfig(level=logging.INFO)

# ==============================
# 1. Load dan siapkan data
# ==============================
df = pd.read_csv('data/historical_data.csv')

# Pastikan ada kolom harga close untuk target
if 'Close' not in df.columns:
    raise ValueError("Kolom 'Close' tidak ditemukan dalam data!")

# Sortir berdasarkan waktu
df = df.sort_values(by='Date')
df = df.dropna()

# ==============================
# 2. Buat target biner naik/turun
# ==============================
df['Close_Tomorrow'] = df['Close'].shift(-1)
df['Target'] = (df['Close_Tomorrow'] > df['Close']).astype(int)
df = df.dropna(subset=['Target'])  # drop baris terakhir karena target-nya NaN

# ==============================
# 3. Preprocessing fitur
# ==============================
# Drop kolom yang tidak dipakai
drop_cols = ['Date', 'Close_Tomorrow', 'Target']
feature_cols = [col for col in df.columns if col not in drop_cols]

X = df[feature_cols]
y = df['Target']

# Scaling
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Simpan scaler
os.makedirs("models", exist_ok=True)
joblib.dump(scaler, "models/scaler.pkl")
logging.info("✅ Scaler berhasil disimpan ke models/scaler.pkl")

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, shuffle=False)

# ==============================
# 4. Train LightGBM
# ==============================
lgb_model = lgb.LGBMClassifier()
lgb_model.fit(X_train, y_train)
joblib.dump(lgb_model, "models/lightgbm_model.pkl")
logging.info("✅ Model LightGBM berhasil disimpan.")

# ==============================
# 5. Train LSTM
# ==============================
X_train_lstm = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
X_test_lstm = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

lstm_model = Sequential()
lstm_model.add(LSTM(64, input_shape=(1, X_train.shape[1]), return_sequences=False))
lstm_model.add(Dense(1, activation='sigmoid'))

lstm_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
lstm_model.fit(X_train_lstm, y_train, epochs=55, batch_size=32, verbose=1)

lstm_model.save("models/lstm_model.h5")
logging.info("✅ Model LSTM berhasil disimpan.")
