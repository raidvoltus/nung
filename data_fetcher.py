import yfinance as yf
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

class StockDataFetcher:
    def __init__(self, period='6mo', interval='1h'):
        self.period = period
        self.interval = interval

    def fetch_data(self, ticker):
        print(f"🔄 Ambil data {ticker} dari Yahoo Finance...")
        df = yf.download(ticker, period=self.period, interval=self.interval, auto_adjust=True)
        df.dropna(inplace=True)
        return df

    def get_complete_data(self, ticker):
        df = self.fetch_data(ticker)
        df = calculate_indicators(df)
        return df

def calculate_indicators(df):
    # Moving Averages
    df['SMA_5'] = df['Close'].rolling(window=5).mean()
    df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()

    # RSI
    delta = df['Close'].diff()
    gain = pd.Series(np.where(delta > 0, delta, 0).flatten(), index=df.index)
    loss = pd.Series(np.where(delta < 0, -delta, 0).flatten(), index=df.index)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    df['RSI'] = 100 - (100 / (1 + rs))

    # MACD
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()

    # Bollinger Bands
    df['BB_Middle'] = df['Close'].rolling(window=20).mean()
    df['BB_Std'] = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Middle'] + (2 * df['BB_Std'])
    df['BB_Lower'] = df['BB_Middle'] - (2 * df['BB_Std'])

    # Stochastic Oscillator
    low_14 = df['Low'].rolling(window=14).min()
    high_14 = df['High'].rolling(window=14).max()
    df['%K'] = 100 * ((df['Close'] - low_14) / (high_14 - low_14 + 1e-10))
    df['%D'] = df['%K'].rolling(window=3).mean()

    # ATR (Average True Range)
    df['H-L'] = df['High'] - df['Low']
    df['H-PC'] = abs(df['High'] - df['Close'].shift(1))
    df['L-PC'] = abs(df['Low'] - df['Close'].shift(1))
    tr = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)
    df['ATR'] = tr.rolling(window=14).mean()

    # Volume Analysis
    df['Volume_MA20'] = df['Volume'].rolling(window=20).mean()
    df['Volume_Change'] = df['Volume'].pct_change()

    # Parabolic SAR
    df['SAR'] = talib_sar(df)

    # Support & Resistance (simple high/low pivot)
    df['Support'] = df['Low'].rolling(window=20).min()
    df['Resistance'] = df['High'].rolling(window=20).max()

    df.drop(['H-L', 'H-PC', 'L-PC'], axis=1, inplace=True)
    df.fillna(method='bfill', inplace=True)
    return df

def talib_sar(df, af=0.02, max_af=0.2):
    # Parabolic SAR (manual version)
    sar = df['Close'].copy()
    trend = True  # True for uptrend, False for downtrend
    ep = df['Low'][0] if trend else df['High'][0]
    sar[0] = df['Low'][0] if trend else df['High'][0]
    af_step = af

    for i in range(1, len(df)):
        prev_sar = sar[i - 1]
        if trend:
            sar[i] = prev_sar + af_step * (ep - prev_sar)
            if df['Low'][i] < sar[i]:
                trend = False
                sar[i] = ep
                ep = df['High'][i]
                af_step = af
            elif df['High'][i] > ep:
                ep = df['High'][i]
                af_step = min(af_step + af, max_af)
        else:
            sar[i] = prev_sar + af_step * (ep - prev_sar)
            if df['High'][i] > sar[i]:
                trend = True
                sar[i] = ep
                ep = df['Low'][i]
                af_step = af
            elif df['Low'][i] < ep:
                ep = df['Low'][i]
                af_step = min(af_step + af, max_af)

    return sar
