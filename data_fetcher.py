import yfinance as yf
import pandas as pd
import numpy as np
import os

class StockDataFetcher:
    def __init__(self, tickers, start_date="2023-01-01", interval="1h"):
        self.tickers = tickers
        self.start_date = start_date
        self.interval = interval

    def get_complete_data(self, ticker):
        print(f"🔄 Ambil data {ticker} dari Yahoo Finance...")
        df = yf.download(ticker, start=self.start_date, interval=self.interval, progress=False)
        if df.empty:
            print(f"⚠️ Data kosong untuk {ticker}.")
            return pd.DataFrame()
        df = df.dropna()
        df = calculate_indicators(df)
        df["Ticker"] = ticker
        return df

    def save_all_to_csv(self, filename="data/historical_data.csv"):
        os.makedirs("data", exist_ok=True)
        all_data = []
        for ticker in self.tickers:
            df = self.get_complete_data(ticker)
            if not df.empty:
                all_data.append(df)
        if all_data:
            final_df = pd.concat(all_data)
            final_df.to_csv(filename)
            print(f"✅ Data berhasil disimpan ke {filename}")
        else:
            print("❌ Tidak ada data yang bisa disimpan.")


def calculate_indicators(df):
    # SMA & EMA
    df["SMA_5"] = df["Close"].rolling(window=5).mean()
    df["EMA_12"] = df["Close"].ewm(span=12, adjust=False).mean()
    df["EMA_26"] = df["Close"].ewm(span=26, adjust=False).mean()

    # MACD
    df["MACD"] = df["EMA_12"] - df["EMA_26"]
    df["Signal_Line"] = df["MACD"].ewm(span=9, adjust=False).mean()

    # RSI
    delta = df["Close"].diff()
    gain = pd.Series(np.where(delta.values > 0, delta.values, 0).ravel(), index=df.index)
    loss = pd.Series(np.where(delta.values < 0, -delta.values, 0).ravel(), index=df.index)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    df["RSI"] = 100 - (100 / (1 + rs))

    # Bollinger Bands
    df["BB_Middle"] = df["Close"].rolling(window=20).mean()
    df["BB_Std"] = df["Close"].rolling(window=20).std()
    df["BB_Upper"] = df["BB_Middle"] + (2 * df["BB_Std"])
    df["BB_Lower"] = df["BB_Middle"] - (2 * df["BB_Std"])

    # Stochastic Oscillator
    low_14 = df["Low"].rolling(window=14).min()
    high_14 = df["High"].rolling(window=14).max()
    df["%K"] = 100 * ((df["Close"] - low_14) / (high_14 - low_14 + 1e-10))
    df["%D"] = df["%K"].rolling(window=3).mean()

    # ATR
    high_low = df["High"] - df["Low"]
    high_close = np.abs(df["High"] - df["Close"].shift())
    low_close = np.abs(df["Low"] - df["Close"].shift())
    tr = high_low.combine(high_close, np.maximum).combine(low_close, np.maximum)
    df["ATR"] = tr.rolling(window=14).mean()

    # Volume Moving Average
    df["Volume_MA20"] = df["Volume"].rolling(window=20).mean()

    # Parabolic SAR (sederhana)
    df["SAR"] = talib_sar(df)

    return df


def talib_sar(df):
    # Implementasi Parabolic SAR sederhana
    af = 0.02
    max_af = 0.2
    trend = True  # True = uptrend, False = downtrend
    ep = df["Low"].iloc[0] if trend else df["High"].iloc[0]
    sar = df["Close"].iloc[0]
    result = []

    for i in range(len(df)):
        prev_sar = sar
        if trend:
            sar = sar + af * (ep - sar)
            if df["Low"].iloc[i] < sar:
                trend = False
                sar = ep
                ep = df["High"].iloc[i]
                af = 0.02
        else:
            sar = sar + af * (ep - sar)
            if df["High"].iloc[i] > sar:
                trend = True
                sar = ep
                ep = df["Low"].iloc[i]
                af = 0.02

        if trend:
            if df["High"].iloc[i] > ep:
                ep = df["High"].iloc[i]
                af = min(af + 0.02, max_af)
        else:
            if df["Low"].iloc[i] < ep:
                ep = df["Low"].iloc[i]
                af = min(af + 0.02, max_af)

        result.append(sar)

    return pd.Series(result, index=df.index)
