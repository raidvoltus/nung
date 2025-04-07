import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class StockDataFetcher:
    def __init__(self, start_date=None, end_date=None, interval='1h'):
        self.start_date = start_date or (datetime.now() - timedelta(days=180)).strftime('%Y-%m-%d')
        self.end_date = end_date or datetime.now().strftime('%Y-%m-%d')
        self.interval = interval

    def get_complete_data(self, ticker):
        print(f"🔄 Ambil data {ticker} dari Yahoo Finance...")
        df = yf.download(ticker, start=self.start_date, end=self.end_date, interval=self.interval)
        if df.empty:
            print(f"⚠️ Data {ticker} kosong.")
            return pd.DataFrame()
        df = df.dropna()
        df = calculate_indicators(df)
        return df

def calculate_indicators(df):
    # SMA & EMA
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()

    # RSI
    delta = df['Close'].diff()
    gain = pd.Series(np.where(delta > 0, delta, 0), index=df.index)
    loss = pd.Series(np.where(delta < 0, -delta, 0), index=df.index)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # MACD
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

    # Bollinger Bands
    df['BB_Middle'] = df['Close'].rolling(window=20).mean()
    df['BB_Std'] = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Middle'] + 2 * df['BB_Std']
    df['BB_Lower'] = df['BB_Middle'] - 2 * df['BB_Std']

    # Stochastic Oscillator
    low_14 = df['Low'].rolling(window=14).min()
    high_14 = df['High'].rolling(window=14).max()
    df['%K'] = 100 * ((df['Close'] - low_14) / (high_14 - low_14))
    df['%D'] = df['%K'].rolling(window=3).mean()

    # ATR
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['ATR'] = tr.rolling(window=14).mean()

    # Parabolic SAR
    df['SAR'] = talib_sar(df)

    # Volume Analysis
    df['Volume_MA_20'] = df['Volume'].rolling(window=20).mean()

    return df

def talib_sar(df, af=0.02, max_af=0.2):
    sar = pd.Series(index=df.index, dtype='float64')
    trend = True

    if 'Low' not in df.columns or 'High' not in df.columns:
        print("❌ Kolom 'Low' atau 'High' tidak ditemukan dalam data.")
        return sar

    ep = df['Low'].iloc[0] if trend else df['High'].iloc[0]
    sar.iloc[0] = df['Low'].iloc[0] if trend else df['High'].iloc[0]
    af_step = af

    for i in range(1, len(df)):
        prev_sar = sar.iloc[i - 1]
        if trend:
            sar.iloc[i] = prev_sar + af_step * (ep - prev_sar)
            if df['Low'].iloc[i] < sar.iloc[i]:
                trend = False
                sar.iloc[i] = ep
                ep = df['High'].iloc[i]
                af_step = af
            elif df['High'].iloc[i] > ep:
                ep = df['High'].iloc[i]
                af_step = min(af_step + af, max_af)
        else:
            sar.iloc[i] = prev_sar + af_step * (ep - prev_sar)
            if df['High'].iloc[i] > sar.iloc[i]:
                trend = True
                sar.iloc[i] = ep
                ep = df['Low'].iloc[i]
                af_step = af
            elif df['Low'].iloc[i] < ep:
                ep = df['Low'].iloc[i]
                af_step = min(af_step + af, max_af)

    return sar
