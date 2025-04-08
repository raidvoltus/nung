import yfinance as yf
import pandas as pd
import numpy as np

def fetch_stock_data(symbol: str, period: str = "6mo", interval: str = "1h") -> pd.DataFrame:
    df = yf.download(symbol, period=period, interval=interval)
    df.dropna(inplace=True)
    df["Symbol"] = symbol
    df.reset_index(inplace=True)

    # Tambahkan indikator teknikal untuk day trading
    df["SMA_10"] = df["Close"].rolling(window=10).mean()
    df["EMA_10"] = df["Close"].ewm(span=10, adjust=False).mean()
    df["RSI"] = compute_rsi(df["Close"])
    df["MACD"], df["MACD_signal"] = compute_macd(df["Close"])
    df["Upper_BB"], df["Lower_BB"] = compute_bollinger_bands(df["Close"])
    df["%K"], df["%D"] = compute_stochastic(df["High"], df["Low"], df["Close"])
    df["ATR"] = compute_atr(df["High"], df["Low"], df["Close"])
    df["SAR"] = compute_parabolic_sar(df)
    df["Volume_Change"] = df["Volume"].pct_change()

    df.dropna(inplace=True)
    return df

def fetch_all_stock_data(symbols: list) -> pd.DataFrame:
    all_data = []
    for symbol in symbols:
        print(f"[INFO] Fetching data for {symbol}")
        try:
            data = fetch_stock_data(symbol)
            all_data.append(data)
        except Exception as e:
            print(f"[ERROR] Gagal fetch {symbol}: {e}")
    return pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()

# ====== FUNGSI INDIKATOR ======
def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=period).mean()
    avg_loss = pd.Series(loss).rolling(window=period).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    return 100 - (100 / (1 + rs))

def compute_macd(series: pd.Series):
    exp1 = series.ewm(span=12, adjust=False).mean()
    exp2 = series.ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=9, adjust=False).mean()
    return macd, signal

def compute_bollinger_bands(series: pd.Series, window: int = 20, num_std: int = 2):
    sma = series.rolling(window=window).mean()
    std = series.rolling(window=window).std()
    upper = sma + (std * num_std)
    lower = sma - (std * num_std)
    return upper, lower

def compute_stochastic(high: pd.Series, low: pd.Series, close: pd.Series, k_period=14, d_period=3):
    low_min = low.rolling(window=k_period).min()
    high_max = high.rolling(window=k_period).max()
    k = 100 * ((close - low_min) / (high_max - low_min + 1e-10))
    d = k.rolling(window=d_period).mean()
    return k, d

def compute_atr(high, low, close, period=14):
    high_low = high - low
    high_close = np.abs(high - close.shift())
    low_close = np.abs(low - close.shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    return atr

def compute_parabolic_sar(df, af=0.02, max_af=0.2):
    high = df['High'].values
    low = df['Low'].values
    close = df['Close'].values

    sar = close.copy()
    trend = 1
    ep = low[0] if trend == 1 else high[0]
    af_val = af

    for i in range(2, len(close)):
        sar[i] = sar[i-1] + af_val * (ep - sar[i-1])
        reverse = False

        if trend == 1:
            if low[i] < sar[i]:
                trend = -1
                reverse = True
                sar[i] = ep
                ep = high[i]
                af_val = af
        else:
            if high[i] > sar[i]:
                trend = 1
                reverse = True
                sar[i] = ep
                ep = low[i]
                af_val = af

        if not reverse:
            if trend == 1:
                if high[i] > ep:
                    ep = high[i]
                    af_val = min(af_val + af, max_af)
            else:
                if low[i] < ep:
                    ep = low[i]
                    af_val = min(af_val + af, max_af)

    return pd.Series(sar, index=df.index)
