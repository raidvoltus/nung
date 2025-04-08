import sqlite3
import requests
import datetime

# Telegram Setup
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# Fungsi kirim pesan ke Telegram
def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "Markdown"
    }
    requests.post(url, data=payload)

# Logging SQLite
def log_signal(stock, signal_type, buy_price, take_profit, stop_loss, probability):
    conn = sqlite3.connect("data/signals_log.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS signals (
            date TEXT, stock TEXT, signal_type TEXT,
            buy_price REAL, take_profit REAL, stop_loss REAL, probability REAL
        )
    """)
    cursor.execute("""
        INSERT INTO signals (date, stock, signal_type, buy_price, take_profit, stop_loss, probability)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (datetime.datetime.now().isoformat(), stock, signal_type, buy_price, take_profit, stop_loss, probability))
    conn.commit()
    conn.close()

# Hitung TP/SL otomatis berdasarkan ATR dan RRR
def calculate_tp_sl(price, atr, support, resistance, rrr=2):
    take_profit = min(price + rrr * atr, resistance)
    stop_loss = max(price - atr, support)
    return round(take_profit, 2), round(stop_loss, 2)

# Fungsi kirim sinyal jika valid
def handle_signal(stock, current_price, atr, support, resistance, prediction, probability):
    if prediction == 1 and probability >= 0.8:
        take_profit, stop_loss = calculate_tp_sl(current_price, atr, support, resistance)

        rr_ratio = round((take_profit - current_price) / (current_price - stop_loss), 2)
        if rr_ratio < 1.5:
            print(f"[SKIP] {stock} RR ratio rendah ({rr_ratio})")
            return

        message = f"""
**Sinyal Beli AI - {stock}**
Harga Saat Ini: *Rp{current_price}*
Take Profit: *Rp{take_profit}*
Stop Loss: *Rp{stop_loss}*
Probabilitas: *{round(probability * 100, 2)}%*
Risk/Reward: *{rr_ratio}*

_Waspada! Gunakan manajemen risiko._
"""
        send_telegram_message(message)
        log_signal(stock, "BUY", current_price, take_profit, stop_loss, probability)
        print(f"[SINYAL TERKIRIM] {stock} - Prob: {round(probability, 2)} RR: {rr_ratio}")
