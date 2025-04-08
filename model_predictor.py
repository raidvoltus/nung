import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from lightgbm import LGBMClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import joblib
import os

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    df['Return'] = df['Close'].pct_change()
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    df.dropna(inplace=True)
    return df

def create_lstm_sequences(df, features, window_size=10):
    X, y = [], []
    for i in range(window_size, len(df)):
        X.append(df[features].iloc[i - window_size:i].values)
        y.append(df['Target'].iloc[i])
    return np.array(X), np.array(y)

def train_lightgbm(df, features):
    X = df[features]
    y = df["Target"]
    model = LGBMClassifier(n_estimators=100, learning_rate=0.1)
    model.fit(X, y)
    joblib.dump(model, "models/lightgbm_model.pkl")
    print("[+] LightGBM model saved.")
    return model

def train_lstm(df, features, window_size=10):
    scaler = MinMaxScaler()
    df_scaled = df.copy()
    df_scaled[features] = scaler.fit_transform(df[features])
    
    X, y = create_lstm_sequences(df_scaled, features, window_size)

    model = Sequential([
        LSTM(64, input_shape=(X.shape[1], X.shape[2]), return_sequences=False),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X, y, epochs=10, batch_size=32, validation_split=0.2,
              callbacks=[EarlyStopping(patience=3, restore_best_weights=True)],
              verbose=1)
    model.save("models/lstm_model.h5")
    joblib.dump(scaler, "models/lstm_scaler.pkl")
    print("[+] LSTM model saved.")
    return model

def predict_price_movement(lightgbm_model, lstm_model, df, features, window_size=10):
    last_data = df[features].iloc[-window_size:]
    
    # LightGBM
    proba_lgb = lightgbm_model.predict_proba([last_data.iloc[-1]])[0][1]
    
    # LSTM
    scaler = joblib.load("models/lstm_scaler.pkl")
    scaled_last = scaler.transform(last_data)
    lstm_input = np.expand_dims(scaled_last, axis=0)
    proba_lstm = lstm_model.predict(lstm_input, verbose=0)[0][0]
    
    final_proba = (proba_lgb + proba_lstm) / 2
    return final_proba
