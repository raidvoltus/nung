import os
import pandas as pd
from data_fetcher import StockDataFetcher

STOCK_LIST = ["BBCA.JK", "BBRI.JK", "BMRI.JK", "TLKM.JK", "ASII.JK"]
OUTPUT_FILE = "data/historical_data.csv"

os.makedirs("data", exist_ok=True)
fetcher = StockDataFetcher()

all_data = []

for ticker in STOCK_LIST:
    df = fetcher.get_complete_data(ticker)
    if df is None or df.empty:
        continue

    df["Ticker"] = ticker
    df["Target"] = df["Close"].shift(-1)  # Target: harga penutupan besok
    df.dropna(inplace=True)
    all_data.append(df)

if all_data:
    final_df = pd.concat(all_data, ignore_index=True)
    final_df.to_csv(OUTPUT_FILE, index=False)
    print(f"✅ Data historis disimpan ke {OUTPUT_FILE}")
else:
    print("⚠️ Tidak ada data yang berhasil diambil.")
