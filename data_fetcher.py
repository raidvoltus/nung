import yfinance as yf
import pandas as pd
import numpy as np
import datetime as dt

class StockDataFetcher:
    def __init__(self, tickers):
        self.tickers = tickers

    def get_complete_data(self, ticker):
        print(f"\n🔄 Ambil data {ticker} dari Yahoo Finance...")
        end_date = dt.datetime.now()
        start_date = end_date - dt.timedelta(days=365)  # 1 tahun untuk 1H

        df = yf.download(ticker, start=start_date, end=end_date, interval='1h', progress=False)

        if df.empty or 'Close' not in df.columns:
            print(f"⚠️ Data kosong untuk {ticker}.")
            return pd.DataFrame()

        df.dropna(inplace=True)
        df = calculate_indicators(df)
        return df

    def save_all_to_csv(self, path):
        combined_df = []
        for ticker in self.tickers:
            df = self.get_complete_data(ticker)
            if not df.empty:
                df['Ticker'] = ticker
                combined_df.append(df)
        
        if combined_df:
            final_df = pd.concat(combined_df)
            final_df.to_csv(path)
            print(f"\n✅ Data disimpan di {path}")
        else:
            print("\n⚠️ Tidak ada data valid yang berhasil disimpan.")

# ======================= INDICATOR FUNCTIONS ========================= #

def calculate_indicators(df):
    df["SMA_20"] = df["Close"].rolling(window=20).mean()
    df["EMA_12"] = df["Close"].ewm(span=12, adjust=False).mean()
    df["EMA_26"] = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = df["EMA_12"] - df["EMA_26"]
    df["Signal_Line"] = df["MACD"].ewm(span=9, adjust=False).mean()

    delta = df["Close"].diff()
    gain = pd.Series(np.where(delta > 0, delta, 0), index=df.index)
    loss = pd.Series(np.where(delta < 0, -delta, 0), index=df.index)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df["RSI"] = 100 - (100 / (1 + rs))

    df["Upper_BB"] = df["Close"].rolling(window=20).mean() + (df["Close"].rolling(window=20).std() * 2)
    df["Lower_BB"] = df["Close"].rolling(window=20).mean() - (df["Close"].rolling(window=20).std() * 2)

    low14 = df["Low"].rolling(window=14).min()
    high14 = df["High"].rolling(window=14).max()
    df["%K"] = 100 * ((df["Close"] - low14) / (high14 - low14))
    df["%D"] = df["%K"].rolling(window=3).mean()

    df["ATR"] = atr(df, 14)
    df["SAR"] = talib_sar(df)

    df["Volume_Change"] = df["Volume"].pct_change()

    return df

def atr(df, period=14):
    high_low = df["High"] - df["Low"]
    high_close = np.abs(df["High"] - df["Close"].shift())
    low_close = np.abs(df["Low"] - df["Close"].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    return true_range.rolling(window=period).mean()

def talib_sar(df):
    if df.empty or 'High' not in df.columns or 'Low' not in df.columns:
        return pd.Series(index=df.index, data=np.nan)

    sar = []
    trend = True  # Uptrend = True
    ep = df['Low'].iloc[0] if trend else df['High'].iloc[0]
    af = 0.02
    max_af = 0.2

    for i in range(len(df)):
        if i == 0:
            sar.append(df['Low'].iloc[0])
            continue

        prev_sar = sar[-1]
        if trend:
            new_sar = prev_sar + af * (ep - prev_sar)
            if df["Low"].iloc[i] < new_sar:
                trend = False
                sar.append(ep)
                ep = df["High"].iloc[i]
                af = 0.02
            else:
                if df["High"].iloc[i] > ep:
                    ep = df["High"].iloc[i]
                    af = min(af + 0.02, max_af)
                sar.append(new_sar)
        else:
            new_sar = prev_sar + af * (ep - prev_sar)
            if df["High"].iloc[i] > new_sar:
                trend = True
                sar.append(ep)
                ep = df["Low"].iloc[i]
                af = 0.02
            else:
                if df["Low"].iloc[i] < ep:
                    ep = df["Low"].iloc[i]
                    af = min(af + 0.02, max_af)
                sar.append(new_sar)

    return pd.Series(sar, index=df.index)
