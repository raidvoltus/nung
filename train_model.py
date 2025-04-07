import pandas as pd
import numpy as np
import os
import joblib
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.callbacks import EarlyStopping
from keras.models import load_model

logging.basicConfig(level=logging.INFO)

def create_target(df):
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)  # naik = 1, turun = 0
    return df.dropna()

def load_features_targets(df):
    feature_cols = [
        'SMA_5', 'SMA_10', 'EMA_5', 'EMA_10', 'RSI_14', 'MACD', 'MACD_signal',
        'BB_Middle', 'BB_Upper', 'BB_Lower', 'Stochastic_K', 'Stochastic_D',
        'ATR_14', 'Volume', 'Volume_MA20', 'Tenkan_sen', 'Kijun_sen',
        'Senkou_Span_A', 'Senkou_Span_B', 'Chikou_Span', 'SAR',
        'Fibo_0.236', 'Fibo_0.382', 'Fibo_0.5', 'Fibo_0.618', 'Fibo_0.786'
    ]
    df = df.dropna(subset=feature_cols + ['Target'])
    X = df[feature_cols]
    y = df['Target']
    return X, y

def train_lightgbm(X_train, y_train):
    model = LGBMClassifier(n_estimators=100, learning_rate=0.05, max_depth=6)
    model.fit(X_train, y_train)
    joblib.dump(model, "data/lgb_model.pkl")
    logging.info("✅ Model LightGBM berhasil dilatih dan disimpan.")
    return model

def train_lstm(X_train, y_train):
    X_train = np.reshape(X_train.values, (X_train.shape[0], 1, X_train.shape[1]))

    model = Sequential()
    model.add(LSTM(units=64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dropout(0.2))
    model.add(LSTM(units=32))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    early_stop = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
    model.fit(X_train, y_train, epochs=53, batch_size=32, verbose=0, callbacks=[early_stop])

    model.save("data/lstm_model.h5")
    logging.info("✅ Model LSTM berhasil dilatih dan disimpan.")
    return model

def train_models():
    if not os.path.exists('data/historical_data.csv'):
        logging.error("❌ File data/historical_data.csv tidak ditemukan.")
        return

    df = pd.read_csv('data/historical_data.csv')

    if df.empty:
        raise ValueError("❌ File CSV kosong. Tidak ada data untuk training.")

    if 'Datetime' in df.columns:
        df['Datetime'] = pd.to_datetime(df['Datetime'])
        df.set_index('Datetime', inplace=True)
    else:
        raise ValueError("❌ Kolom 'Datetime' tidak ditemukan di file CSV.")

    df = create_target(df)
    X, y = load_features_targets(df)

    # Standarisasi fitur
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, 'data/scaler.pkl')

    X_train, _, y_train, _ = train_test_split(X_scaled, y, test_size=0.2, shuffle=False)

    train_lightgbm(pd.DataFrame(X_train, columns=X.columns), y_train)
    train_lstm(pd.DataFrame(X_train, columns=X.columns), y_train)

if __name__ == '__main__':
    train_models()
