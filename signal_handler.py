import os
import logging
import requests
import time
from datetime import datetime
import pandas as pd

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")

if not TELEGRAM_TOKEN or not CHAT_ID:
    raise EnvironmentError("⚠️ TELEGRAM_TOKEN / CHAT_ID tidak ditemukan.")

logging.basicConfig(level=logging.INFO)

def send_telegram_message(message, max_retries=3):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    data = {"chat_id": CHAT_ID, "text": message, "parse_mode": "HTML"}
    for attempt in range(max_retries):
        try:
            response = requests.post(url, data=data)
            response.raise_for_status()
            return True
        except requests.exceptions.RequestException as e:
            logging.warning(f"Percobaan {attempt+1}: {e}")
            time.sleep(2 ** attempt)
    logging.error("❌ Gagal kirim Telegram.")
    return False

def log_predictions(ticker, actual, predicted):
    log_file = "data/predictions_log.csv"
    os.makedirs("data", exist_ok=True)

    df = pd.DataFrame([{
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "ticker": ticker,
        "actual": actual,
        "predicted": predicted
    }])
    df.to_csv(log_file, mode="a", index=False, header=not os.path.exists(log_file))