import pandas as pd
import numpy as np
import os
import yfinance as yf
import logging

logging.basicConfig(level=logging.INFO)

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
    df = df.copy()
    
    # Moving Averages
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['EMA_10'] = df['Close'].ewm(span=10, adjust=False).mean()
    
    # RSI
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    # Bollinger Bands
    sma20 = df['Close'].rolling(window=20).mean()
    std20 = df['Close'].rolling(window=20).std()
    df['BB_upper'] = sma20 + (2 * std20)
    df['BB_lower'] = sma20 - (2 * std20)
    
    # Stochastic Oscillator
    low14 = df['Low'].rolling(window=14).min()
    high14 = df['High'].rolling(window=14).max()
    df['%K'] = 100 * ((df['Close'] - low14) / (high14 - low14 + 1e-10))
    df['%D'] = df['%K'].rolling(window=3).mean()
    
    # ATR
    df['H-L'] = df['High'] - df['Low']
    df['H-PC'] = abs(df['High'] - df['Close'].shift(1))
    df['L-PC'] = abs(df['Low'] - df['Close'].shift(1))
    df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)
    df['ATR'] = df['TR'].rolling(window=14).mean()
    
    # Parabolic SAR
    df['SAR'] = df['Close'].copy()
    af = 0.02
    max_af = 0.2
    trend_up = True
    ep = df['Low'][0] if trend_up else df['High'][0]

    for i in range(1, len(df)):
        if trend_up:
            df.at[df.index[i], 'SAR'] = df['SAR'][i - 1] + af * (ep - df['SAR'][i - 1])
            if df['Low'][i] < df['SAR'][i]:
                trend_up = False
                ep = df['High'][i]
                af = 0.02
                df.at[df.index[i], 'SAR'] = ep
            else:
                if df['High'][i] > ep:
                    ep = df['High'][i]
                    af = min(af + 0.02, max_af)
        else:
            df.at[df.index[i], 'SAR'] = df['SAR'][i - 1] + af * (ep - df['SAR'][i - 1])
            if df['High'][i] > df['SAR'][i]:
                trend_up = True
                ep = df['Low'][i]
                af = 0.02
                df.at[df.index[i], 'SAR'] = ep
            else:
                if df['Low'][i] < ep:
                    ep = df['Low'][i]
                    af = min(af + 0.02, max_af)
    
    # Volume Analysis
    df['Volume_Change'] = df['Volume'].pct_change()
    df['Vol_MA_10'] = df['Volume'].rolling(window=10).mean()
    
    # Support / Resistance (sederhana: swing high/low)
    df['Support'] = df['Low'].rolling(window=20).min()
    df['Resistance'] = df['High'].rolling(window=20).max()

    # Target Biner (Naik / Turun)
    df['Target'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)

    df = df.dropna()
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
