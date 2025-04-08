import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from lightgbm import LGBMClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.callbacks import EarlyStopping
import joblib
from data_fetcher import fetch_all_stock_data
from model_predictor import create_features, create_lstm_sequences

# =========================
# Cek / Fetch Data
# =========================
symbols = ['BBRI', 'TLKM', 'BBCA', 'BMRI', 'ASII', 'ANTM', 'BRIS', 'SIDO', 'UNVR', 'MDKA']
df = fetch_all_stock_data(symbols)

# Save CSV untuk backup
os.makedirs("data", exist_ok=True)
df.to_csv("data/training_data.csv", index=False)

# =========================
# Preprocessing untuk LightGBM
# =========================
df_lgbm = create_features(df.copy())
X_lgbm = df_lgbm.drop(columns=['target'])
y_lgbm = df_lgbm['target']

X_train_lgbm, X_test_lgbm, y_train_lgbm, y_test_lgbm = train_test_split(
    X_lgbm, y_lgbm, test_size=0.2, random_state=42
)

model_lgbm = LGBMClassifier(n_estimators=100, learning_rate=0.05, random_state=42)
model_lgbm.fit(X_train_lgbm, y_train_lgbm)

# Simpan model
os.makedirs("models", exist_ok=True)
joblib.dump(model_lgbm, "models/lightgbm_model.pkl")

# =========================
# Preprocessing untuk LSTM
# =========================
X_lstm, y_lstm = create_lstm_sequences(df.copy(), sequence_length=20)
X_train_lstm, X_test_lstm, y_train_lstm, y_test_lstm = train_test_split(
    X_lstm, y_lstm, test_size=0.2, random_state=42
)

model_lstm = Sequential()
model_lstm.add(LSTM(units=50, return_sequences=True, input_shape=(X_train_lstm.shape[1], X_train_lstm.shape[2])))
model_lstm.add(LSTM(units=50))
model_lstm.add(Dense(1, activation='sigmoid'))

model_lstm.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model_lstm.fit(
    X_train_lstm, y_train_lstm,
    validation_data=(X_test_lstm, y_test_lstm),
    epochs=50,
    batch_size=32,
    callbacks=[EarlyStopping(patience=3)],
    verbose=1
)

# Simpan model
model_lstm.save("models/lstm_model.h5")

print("✅ Model LightGBM dan LSTM berhasil dilatih dan disimpan.")
