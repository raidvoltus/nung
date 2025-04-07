import yfinance as yf
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

class StockDataFetcher:
    def __init__(self, tickers):
        self.tickers = tickers

    def get_complete_data(self, ticker):
        print(f"\n🔄 Ambil data {ticker} dari Yahoo Finance...")
        end = datetime.now()
        start = end - timedelta(days=180)  # 6 bulan terakhir
        df = yf.download(ticker, start=start, end=end, interval="1h", auto_adjust=True)

        if df.empty or len(df) < 50:
            print(f"⚠️ Data kosong untuk {ticker}.")
            return None

        df = df.dropna()
        df = calculate_indicators(df)
        return df

    def save_all_to_csv(self, output_path):
        all_data = []
        for ticker in self.tickers:
            df = self.get_complete_data(ticker)
            if df is not None:
                df["Ticker"] = ticker
                all_data.append(df)
        if all_data:
            combined_df = pd.concat(all_data)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            combined_df.to_csv(output_path)
            print(f"\n✅ Data historis berhasil disimpan ke {output_path}")
        else:
            print("❌ Tidak ada data yang berhasil diambil.")


def calculate_indicators(df):
    # Moving Averages
    df["SMA_20"] = df["Close"].rolling(window=20).mean()
    df["EMA_12"] = df["Close"].ewm(span=12, adjust=False).mean()
    df["EMA_26"] = df["Close"].ewm(span=26, adjust=False).mean()

    # RSI
    delta = df["Close"].diff()
    gain = pd.Series(np.where(delta > 0, delta, 0).ravel(), index=df.index)
    loss = pd.Series(np.where(delta < 0, -delta, 0).ravel(), index=df.index)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df["RSI"] = 100 - (100 / (1 + rs))

    # MACD
    df["MACD"] = df["EMA_12"] - df["EMA_26"]
    df["Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

    # Bollinger Bands
    df["BB_Middle"] = df["Close"].rolling(window=20).mean()
    bb_std = df["Close"].rolling(window=20).std().squeeze()
    df["BB_Upper"] = df["BB_Middle"] + 2 * bb_std
    df["BB_Lower"] = df["BB_Middle"] - 2 * bb_std

    # Stochastic Oscillator
    low_14 = df["Low"].rolling(window=14).min()
    high_14 = df["High"].rolling(window=14).max()
    df["%K"] = 100 * (df["Close"] - low_14) / (high_14 - low_14)
    df["%D"] = df["%K"].rolling(window=3).mean()

    # ATR
    df["H-L"] = df["High"] - df["Low"]
    df["H-PC"] = abs(df["High"] - df["Close"].shift(1))
    df["L-PC"] = abs(df["Low"] - df["Close"].shift(1))
    tr = df[["H-L", "H-PC", "L-PC"]].max(axis=1)
    df["ATR"] = tr.rolling(window=14).mean()

    # Volume Analysis
    df["Volume_MA_20"] = df["Volume"].rolling(window=20).mean()

    # Parabolic SAR
    df["SAR"] = talib_sar(df)

    df.drop(["H-L", "H-PC", "L-PC"], axis=1, inplace=True)
    return df


def talib_sar(df, af_start=0.02, af_increment=0.02, af_max=0.2):
    high = df["High"].values
    low = df["Low"].values
    close = df["Close"].values

    sar = np.zeros(len(df))
    trend = True  # True = uptrend, False = downtrend
    ep = low[0]
    af = af_start

    for i in range(1, len(df)):
        if trend:
            sar[i] = sar[i - 1] + af * (ep - sar[i - 1])
            if low[i] < sar[i]:
                trend = False
                sar[i] = ep
                ep = high[i]
                af = af_start
        else:
            sar[i] = sar[i - 1] + af * (ep - sar[i - 1])
            if high[i] > sar[i]:
                trend = True
                sar[i] = ep
                ep = low[i]
                af = af_start

        if trend:
            if high[i] > ep:
                ep = high[i]
                af = min(af + af_increment, af_max)
        else:
            if low[i] < ep:
                ep = low[i]
                af = min(af + af_increment, af_max)

    return sar
