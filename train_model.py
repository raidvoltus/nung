import pandas as pd
import lightgbm as lgb
import numpy as np
from sklearn.model_selection import train_test_split
import joblib
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.callbacks import EarlyStopping
import os

# Load data
data_path = "data/historical_data.csv"
data = pd.read_csv(data_path)

# Pisahkan fitur dan target
X = data.drop(columns=["Target"])
X = X.select_dtypes(include=["number"])  # Hanya ambil kolom numerik
y = data["Target"]

# Bagi data untuk training & testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Train LightGBM
lgb_model = lgb.LGBMRegressor()
lgb_model.fit(X_train, y_train)

# Simpan model LightGBM
os.makedirs("models", exist_ok=True)
joblib.dump(lgb_model, "models/lightgbm_model.pkl")

# Persiapkan data untuk LSTM (harus 3D)
X_lstm = np.expand_dims(X.values, axis=1)  # [samples, timesteps, features]
y_lstm = y.values

X_train_lstm, X_test_lstm = X_lstm[:len(X_train)], X_lstm[len(X_train):]
y_train_lstm, y_test_lstm = y_lstm[:len(y_train)], y_lstm[len(y_train):]

# Buat dan latih model LSTM
lstm_model = Sequential()
lstm_model.add(LSTM(50, activation='relu', input_shape=(1, X.shape[1])))
lstm_model.add(Dense(1))
lstm_model.compile(optimizer='adam', loss='mse')

early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

lstm_model.fit(X_train_lstm, y_train_lstm, epochs=55, batch_size=16,
               validation_data=(X_test_lstm, y_test_lstm), callbacks=[early_stop], verbose=1)

# Simpan model LSTM
lstm_model.save("models/lstm_model.h5")
print("✅ Model LightGBM dan LSTM berhasil dilatih dan disimpan.")
