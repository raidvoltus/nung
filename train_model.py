import os
import pickle
import pandas as pd
import tensorflow as tf
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel

os.makedirs("models", exist_ok=True)

data = pd.read_csv("data/historical_data.csv")
X = data.drop(columns=["Target"])
y = data["Target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lgb_model = lgb.LGBMRegressor(objective="regression", metric="mae", learning_rate=0.01, n_estimators=1000)
lgb_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=50, verbose=False)

feature_selector = SelectFromModel(lgb_model, threshold="mean", prefit=True)

with open("models/lightgbm_model.pkl", "wb") as f:
    pickle.dump(lgb_model, f)
with open("models/feature_selector.pkl", "wb") as f:
    pickle.dump(feature_selector, f)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.iloc[:, feature_selector.get_support()])
X_test_scaled = scaler.transform(X_test.iloc[:, feature_selector.get_support()])
X_train_lstm = X_train_scaled.reshape(-1, 1, X_train_scaled.shape[1])
X_test_lstm = X_test_scaled.reshape(-1, 1, X_test_scaled.shape[1])

lstm_model = tf.keras.Sequential([
    tf.keras.layers.LSTM(50, return_sequences=True, input_shape=(1, X_train_scaled.shape[1])),
    tf.keras.layers.LSTM(50),
    tf.keras.layers.Dense(25, activation="relu"),
    tf.keras.layers.Dense(1)
])
lstm_model.compile(optimizer="adam", loss="mae")
lstm_model.fit(X_train_lstm, y_train, validation_data=(X_test_lstm, y_test), epochs=50, batch_size=16, verbose=1)
lstm_model.save("models/lstm_model.h5")

with open("models/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("✅ Training selesai.")