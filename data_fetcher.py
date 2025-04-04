import os
import pickle
import yfinance as yf
import pandas as pd
import ta
from datetime import datetime, timedelta
import logging
from fundamental_fetcher import get_fundamental_data

CACHE_FILE = "data/stock_cache.pkl"
CACHE_EXPIRY_DAYS = 7

os.makedirs("data", exist_ok=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class StockDataFetcher:
    def __init__(self):
        self.cache = self.load_cached_data()

    def load_cached_data(self):
        if os.path.exists(CACHE_FILE):
            try:
                with open(CACHE_FILE, "rb") as f:
                    data = pickle.load(f)
                    valid_data = {
                        ticker: (date, d)
                        for ticker, (date, d) in data.items()
                        if date > datetime.now() - timedelta(days=CACHE_EXPIRY_DAYS)
                    }
                    logging.info(f"📂 {len(valid_data)} saham dimuat dari cache.")
                    return valid_data
            except Exception as e:
                logging.error(f"❌ Gagal memuat cache: {e}")
        return {}

    def save_cached_data(self):
        try:
            with open(CACHE_FILE, "wb") as f:
                pickle.dump(self.cache, f)
            logging.info("✅ Cache disimpan.")
        except Exception as e:
            logging.error(f"❌ Gagal simpan cache: {e}")

    def get_stock_data(self, ticker):
        if ticker in self.cache:
            logging.info(f"🟢 {ticker} dari cache.")
            return self.cache[ticker][1]

        try:
            logging.info(f"📥 Unduh {ticker} dari Yahoo Finance...")
            stock = yf.Ticker(ticker)
            data = stock.history(period="5y")

            if data.empty or len(data) < 200:
                logging.warning(f"⚠️ Data {ticker} tidak mencukupi.")
                return None

            self.cache[ticker] = (datetime.now(), data)
            self.save_cached_data()
            return data
        except Exception as e:
            logging.error(f"❌ Gagal unduh {ticker}: {e}")
            return None

    def calculate_indicators(self, df):
        logging.info("📊 Hitung indikator teknikal...")
        df = df.copy()

        try:
            df["MA_20"] = df["Close"].rolling(20).mean()
            df["MA_50"] = df["Close"].rolling(50).mean()
            df["MA_200"] = df["Close"].rolling(200).mean()
            df["BB_High"] = ta.volatility.bollinger_hband(df["Close"], window=20)
            df["BB_Low"] = ta.volatility.bollinger_lband(df["Close"], window=20)
            df["Stoch_K"] = ta.momentum.stoch(df["High"], df["Low"], df["Close"], window=14)
            df["Stoch_D"] = df["Stoch_K"].rolling(3).mean()
            df["RSI_14"] = ta.momentum.rsi(df["Close"], window=14)
            df["ATR_14"] = ta.volatility.average_true_range(df["High"], df["Low"], df["Close"], window=14)
            df["ADX_14"] = ta.trend.adx(df["High"], df["Low"], df["Close"], window=14)
            df["VWAP"] = ta.volume.volume_weighted_average_price(df["High"], df["Low"], df["Close"], df["Volume"], window=14)
            df["Support"] = df["Low"].rolling(20).min()
            df["Resistance"] = df["High"].rolling(20).max()
            df["Volume_Mean"] = df["Volume"].rolling(20).mean()
            df["Volume_Change"] = df["Volume"].pct_change()

            df.dropna(inplace=True)
            return df
        except Exception as e:
            logging.error(f"❌ Error hitung indikator: {e}")
            return None

    def get_complete_data(self, ticker):
        df = self.get_stock_data(ticker)
        if df is None:
            return None

        df = self.calculate_indicators(df)
        if df is None:
            return None

        fundamentals = get_fundamental_data(ticker)
        if fundamentals:
            for k, v in fundamentals.items():
                df[k] = v

        return df