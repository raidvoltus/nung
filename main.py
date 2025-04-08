import os
import time
import pandas as pd
import sqlite3
from datetime import datetime
from data_fetcher import fetch_data_with_indicators
from model_predictor import predict_signal
import requests

# === Konfigurasi Telegram ===
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# === Daftar saham ===
SAHAM_LIST = ['BBRI.JK', 'BBCA.JK', 'TLKM.JK', 'ASII.JK', 'BMRI.JK']

# === Kirim pesan Telegram ===
def kirim_telegram(message):
    if TELEGRAM_TOKEN and TELEGRAM_CHAT_ID:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "HTML"}
        requests.post(url, data=payload)
    else:
        print("[WARNING] Telegram belum dikonfigurasi!")

# === Simpan log ke SQLite ===
def simpan_log_sqlite(symbol, signal):
    conn = sqlite3.connect("data/sinyal_trading.db")
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS sinyal (
            timestamp TEXT, symbol TEXT, prob REAL, rekomendasi TEXT,
            close REAL, tp REAL, sl REAL, rr REAL
        )
    """)
    c.execute("""
        INSERT INTO sinyal VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        symbol,
        signal["predict_up_prob"],
        signal["recommendation"],
        signal["last_close"],
        signal["take_profit"],
        signal["stop_loss"],
        signal["risk_reward"]
    ))
    conn.commit()
    conn.close()

# === Simpan sinyal ke CSV ===
def simpan_backup_csv(symbol, signal):
    df = pd.DataFrame([{
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "symbol": symbol,
        "prob": signal["predict_up_prob"],
        "recommendation": signal["recommendation"],
        "last_close": signal["last_close"],
        "take_profit": signal["take_profit"],
        "stop_loss": signal["stop_loss"],
        "risk_reward": signal["risk_reward"]
    }])
    filename = f"data/sinyal_backup.csv"
    if os.path.exists(filename):
        df.to_csv(filename, mode='a', header=False, index=False)
    else:
        df.to_csv(filename, index=False)

# === Jalankan prediksi per saham ===
def jalankan_bot():
    for symbol in SAHAM_LIST:
        print(f"[INFO] Memproses: {symbol}")
        try:
            df = fetch_data_with_indicators(symbol)
            signal = predict_signal(df)

            print(f"Signal: {signal}")

            if signal["send_signal"]:
                pesan = (
                    f"<b>{symbol}</b>\n"
                    f"Rekomendasi: <b>{signal['recommendation']}</b>\n"
                    f"Probabilitas Naik: {signal['predict_up_prob']:.2f}\n"
                    f"Harga Terakhir: {signal['last_close']}\n"
                    f"Take Profit: {signal['take_profit']}\n"
                    f"Stop Loss: {signal['stop_loss']}\n"
                    f"Risk Reward: {signal['risk_reward']:.2f}"
                )
                kirim_telegram(pesan)

            simpan_log_sqlite(symbol, signal)
            simpan_backup_csv(symbol, signal)

            time.sleep(3)  # Hindari request terlalu cepat

        except Exception as e:
            print(f"[ERROR] Gagal memproses {symbol}: {e}")

if __name__ == "__main__":
    jalankan_bot()
