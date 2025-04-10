import os
import pandas as pd
import numpy as np
from datetime import datetime
from data_fetcher import fetch_all_stock_data
from model_predictor import create_features, create_lstm_sequences, hybrid_predict
from signal_handler import generate_signal
from train_model import *
import joblib
from tensorflow.keras.models import load_model

# =========================
# Parameter dan Symbol
# =========================
symbols = ['BBRI.JK', 'TLKM.JK', 'BBCA.JK', 'BMRI.JK', 'ASII.JK', 'ANTM.JK', 'BRIS.JK', 'SIDO.JK', 'UNVR.JK', 'MDKA.JK']

# =========================
# Cek model, training jika belum ada
# =========================
if not os.path.exists("models/lightgbm_model.pkl") or not os.path.exists("models/lstm_model.h5"):
    print("\n⚠️ Model belum ditemukan. Training sekarang...\n")
    os.system("python train_model.py")

# =========================
# Load model yang sudah ada
# =========================
model_lgbm = joblib.load("models/lightgbm_model.pkl")
model_lstm = load_model("models/lstm_model.h5")

# =========================
# Fetch Data Real-Time
# =========================
df = fetch_all_stock_data(symbols)
df.to_csv("data/realtime_data.csv", index=False)

# =========================
# Prediksi & Sinyal
# =========================
print("\n🚀 Memproses prediksi saham...\n")
results = []

for symbol in symbols:
    df_symbol = df[df['symbol'] == symbol].copy()
    if len(df_symbol) < 50:
        continue

    try:
        df_feat = create_features(df_symbol.copy())
        X_lgbm = df_feat.drop(columns=['target'])

        X_lstm_seq, _ = create_lstm_sequences(df_symbol.copy(), sequence_length=20)

        if len(X_lstm_seq) == 0:
            continue

        prob = hybrid_predict(X_lgbm, X_lstm_seq, model_lgbm, model_lstm)
        signal = generate_signal(df_symbol, prob)

        if signal:
            results.append(signal)

    except Exception as e:
        print(f"Error proses {symbol}: {e}")
        continue

# =========================
# Output Sinyal
# =========================
if results:
    print("\n📈 Sinyal Trading Ditemukan:\n")
    for s in results:
        print(f"{s['symbol']} | Buy: {s['buy_price']} | TP: {s['take_profit']} | SL: {s['stop_loss']} | Prob: {s['probability']:.2f} | RR: {s['risk_reward']:.2f}")
else:
    print("❌ Tidak ada sinyal trading valid ditemukan.")

# =========================
# Simpan log hasil
# =========================
log_file = "data/signal_log.csv"
os.makedirs("data", exist_ok=True)
log_df = pd.DataFrame(results)
log_df["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

if os.path.exists(log_file):
    log_df.to_csv(log_file, mode='a', header=False, index=False)
else:
    log_df.to_csv(log_file, index=False)
