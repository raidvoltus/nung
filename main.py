import os
import logging
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from data_fetcher import StockDataFetcher
from model_predictor import hybrid_predict
from signal_handler import send_telegram_message

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

STOCK_LIST = ["BBCA.JK", "BBRI.JK", "BMRI.JK", "TLKM.JK", "ASII.JK"]
TAKE_PROFIT_PERCENTAGE = 1.03
STOP_LOSS_PERCENTAGE = 0.97

fetcher = StockDataFetcher()

def process_stock(ticker):
    try:
        df = fetcher.get_complete_data(ticker)
        if df is None or df.empty:
            return None

        latest_data = df.iloc[-1:].drop(columns=["Close"], errors="ignore")
        pred = hybrid_predict(latest_data)
        if pred is None:
            return None

        entry = round(pred, 2)
        tp = round(entry * TAKE_PROFIT_PERCENTAGE, 2)
        sl = round(entry * STOP_LOSS_PERCENTAGE, 2)

        return {"ticker": ticker, "entry_price": entry, "take_profit": tp, "stop_loss": sl}
    except Exception as e:
        logging.error(f"{ticker} gagal: {e}")
        return None

if __name__ == "__main__":
    logging.info("🚀 Mulai analisis saham...")
    with ProcessPoolExecutor(max_workers=5) as executor:
        results = list(executor.map(process_stock, STOCK_LIST))

    results = [r for r in results if r]
    if results:
        results.sort(key=lambda x: x["take_profit"], reverse=True)
        top_5 = results[:5]

        msg = "<b>📊 Top 5 Sinyal Trading Hari Ini:</b>\n"
        for r in top_5:
            msg += f"\n🔹 {r['ticker']}: Buy {r['entry_price']}, TP {r['take_profit']}, SL {r['stop_loss']}"
        send_telegram_message(msg)

        os.makedirs("data", exist_ok=True)
        pd.DataFrame(results).to_csv("data/trading_signals.csv", index=False)
    else:
        logging.warning("⚠️ Tidak ada sinyal valid.")