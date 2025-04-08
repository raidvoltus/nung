import pandas as pd
import numpy as np
import joblib
from keras.models import load_model

def load_models():
    try:
        lgb_model = joblib.load("data/lgb_model.pkl")
        lstm_model = load_model("data/lstm_model.h5")
        scaler = joblib.load("data/scaler.pkl")
    except Exception as e:
        raise RuntimeError(f"[ERROR] Gagal memuat model: {e}")
    return lgb_model, lstm_model, scaler

def prepare_features(df):
    feature_cols = [
        'SMA_5', 'SMA_10', 'EMA_5', 'EMA_10', 'RSI_14', 'MACD', 'MACD_signal',
        'BB_Middle', 'BB_Upper', 'BB_Lower', 'Stochastic_K', 'Stochastic_D',
        'ATR_14', 'Volume', 'Volume_MA20', 'Tenkan_sen', 'Kijun_sen',
        'Senkou_Span_A', 'Senkou_Span_B', 'Chikou_Span', 'SAR',
        'Fibo_0.236', 'Fibo_0.382', 'Fibo_0.5', 'Fibo_0.618', 'Fibo_0.786'
    ]
    df = df.dropna(subset=feature_cols)
    latest = df.iloc[-1:][feature_cols]
    return latest, df

def predict_signal(df):
    if len(df) < 60:
        return {"predict_up_prob": 0, "recommendation": "NO DATA", "send_signal": False}

    lgb_model, lstm_model, scaler = load_models()
    latest_features, full_df = prepare_features(df)

    X_scaled = scaler.transform(latest_features)

    lgb_pred = lgb_model.predict_proba(X_scaled)[0][1]
    lstm_input = np.reshape(X_scaled, (X_scaled.shape[0], 1, X_scaled.shape[1]))
    lstm_pred = lstm_model.predict(lstm_input, verbose=0)[0][0]

    final_prob = (lgb_pred + lstm_pred) / 2

    last_close = df['Close'].iloc[-1]
    atr = df['ATR_14'].iloc[-1]
    if atr < 1e-5: atr = 0.1

    res_level = df['High'].rolling(window=20).max().iloc[-1]
    sup_level = df['Low'].rolling(window=20).min().iloc[-1]

    take_profit = min(last_close + 2 * atr, res_level)
    stop_loss = max(last_close - atr, sup_level)

    rr_ratio = (take_profit - last_close) / (last_close - stop_loss + 1e-10)

    signal = {
        "predict_up_prob": round(final_prob, 4),
        "recommendation": "BUY" if final_prob >= 0.75 else "NO TRADE",
        "last_close": round(last_close, 2),
        "take_profit": round(take_profit, 2),
        "stop_loss": round(stop_loss, 2),
        "risk_reward": round(rr_ratio, 2),
        "send_signal": final_prob >= 0.75 and rr_ratio >= 1.5
    }

    return signal
