import pandas as pd
import numpy as np
import os
import yfinance as yf
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)

class StockDataFetcher:
    def __init__(self):
        self.cache_dir = "data/cache/"
        os.makedirs(self.cache_dir, exist_ok=True)

    def fetch(self, symbol, period="6mo", interval="1h"):
        cache_path = f"{self.cache_dir}{symbol}_{interval}.csv"

        if os.path.exists(cache_path):
            df = pd.read_csv(cache_path, parse_dates=True, index_col=0)
            print(f"🟢 {symbol} dari cache.")
        else:
            print(f"🔄 Ambil data {symbol} dari Yahoo Finance...")
            df = yf.download(symbol, period=period, interval=interval)
            df.to_csv(cache_path)

        df = df.dropna()
        return df
        
    def get_complete_data(self, symbol):
        df = self.fetch(symbol)
        df = calculate_indicators(df)
        df['Symbol'] = symbol
        return df
        
def download_data(symbol, period='6mo', interval='1h'):
    filename = f'data/{symbol}_data.csv'
    if os.path.exists(filename):
        logging.info(f"🟢 {symbol} dari cache.")
        return pd.read_csv(filename, parse_dates=['Datetime'], index_col='Datetime')

    df = yf.download(symbol, period=period, interval=interval)
    df = df.rename_axis('Datetime').reset_index()
    df.to_csv(filename, index=False)
    logging.info(f"✅ Data {symbol} berhasil diunduh dan disimpan.")
    return df.set_index('Datetime')

def calculate_indicators(df):
    # SMA & EMA
    df['SMA_5'] = df['Close'].rolling(window=5).mean()
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['EMA_5'] = df['Close'].ewm(span=5, adjust=False).mean()
    df['EMA_10'] = df['Close'].ewm(span=10, adjust=False).mean()

    # RSI
    delta = df['Close'].diff()
    gain = pd.Series(np.where(delta > 0, delta, 0), index=df.index)
    loss = pd.Series(np.where(delta < 0, -delta, 0), index=df.index)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    df['RSI_14'] = 100 - (100 / (1 + rs))

    # MACD
    ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema_12 - ema_26
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

    # Bollinger Bands
    df['BB_Middle'] = df['Close'].rolling(window=20).mean()
    df['BB_Std'] = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Middle'] + (2 * df['BB_Std'])
    df['BB_Lower'] = df['BB_Middle'] - (2 * df['BB_Std'])

    # Stochastic Oscillator
    low14 = df['Low'].rolling(window=14).min()
    high14 = df['High'].rolling(window=14).max()
    df['Stochastic_K'] = 100 * ((df['Close'] - low14) / (high14 - low14 + 1e-10))
    df['Stochastic_D'] = df['Stochastic_K'].rolling(window=3).mean()

    # ATR
    df['H-L'] = df['High'] - df['Low']
    df['H-PC'] = abs(df['High'] - df['Close'].shift(1))
    df['L-PC'] = abs(df['Low'] - df['Close'].shift(1))
    df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)
    df['ATR_14'] = df['TR'].rolling(window=14).mean()

    # Volume Analysis
    df['Volume_MA20'] = df['Volume'].rolling(window=20).mean()
    df['Volume_Breakout'] = df['Volume'] > df['Volume_MA20']

    # Parabolic SAR (manual)
    df['SAR'] = np.nan
    af = 0.02
    max_af = 0.2
    ep = df['Low'].iloc[0]
    sar = df['High'].iloc[0]
    long = True
    for i in range(2, len(df)):
        prev_sar = sar
        if long:
            ep = max(ep, df['High'].iloc[i])
            sar = prev_sar + af * (ep - prev_sar)
            if df['Low'].iloc[i] < sar:
                long = False
                sar = ep
                ep = df['Low'].iloc[i]
                af = 0.02
        else:
            ep = min(ep, df['Low'].iloc[i])
            sar = prev_sar + af * (ep - prev_sar)
            if df['High'].iloc[i] > sar:
                long = True
                sar = ep
                ep = df['High'].iloc[i]
                af = 0.02
        df.at[df.index[i], 'SAR'] = sar

    # Fibonacci
    max_price = df['High'].max()
    min_price = df['Low'].min()
    diff = max_price - min_price
    df['Fibo_0.236'] = max_price - 0.236 * diff
    df['Fibo_0.382'] = max_price - 0.382 * diff
    df['Fibo_0.5'] = max_price - 0.5 * diff
    df['Fibo_0.618'] = max_price - 0.618 * diff
    df['Fibo_0.786'] = max_price - 0.786 * diff

    # Ichimoku Cloud
    df['Tenkan_sen'] = (df['High'].rolling(window=9).max() + df['Low'].rolling(window=9).min()) / 2
    df['Kijun_sen'] = (df['High'].rolling(window=26).max() + df['Low'].rolling(window=26).min()) / 2
    df['Senkou_Span_A'] = ((df['Tenkan_sen'] + df['Kijun_sen']) / 2).shift(26)
    df['Senkou_Span_B'] = ((df['High'].rolling(window=52).max() + df['Low'].rolling(window=52).min()) / 2).shift(26)
    df['Chikou_Span'] = df['Close'].shift(-26)

    # Bersihkan kolom bantu
    df.drop(columns=['H-L', 'H-PC', 'L-PC', 'TR'], inplace=True)

    return df

def fetch_all_data(symbols):
    all_data = []
    for symbol in symbols:
        try:
            df = download_data(symbol)
            df = calculate_indicators(df)
            df['Symbol'] = symbol
            all_data.append(df)
        except Exception as e:
            logging.warning(f"❌ Gagal memproses {symbol}: {e}")

    if all_data:
        combined = pd.concat(all_data)
        combined.to_csv('data/historical_data.csv')
        logging.info("✅ Semua data berhasil digabung dan disimpan.")
        return combined
    else:
        logging.warning("⚠️ Tidak ada data yang berhasil diambil.")
        return pd.DataFrame()
