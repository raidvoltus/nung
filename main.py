import os
import pandas as pd
import numpy as np
from datetime import datetime
from data_fetcher import fetch_all_stock_data
from model_predictor import load_models, create_features, create_lstm_sequences
from signal_handler import handle_signal

# Load model
lightgbm_model, lstm_model, scaler = load_models()

# Jalankan pipeline utama
def main():
    print("Menjalankan bot trading AI...")

    df_all = fetch_all_stock_data()
    if df_all.empty:
        print("Data kosong. Gagal fetch data.")
        return

    tickers = df_all['ticker'].unique()

    for ticker in tickers:
        df = df_all[df_all['ticker'] == ticker].copy()
        if len(df) < 60:
            continue

        # Fitur teknikal
        df = create_features(df)
        df.dropna(inplace=True)

        if len(df) < 30:
            continue

        # Prediksi LightGBM
        last_row = df.iloc[[-1]]
        X_last = last_row.drop(columns=["date", "ticker", "close", "support", "resistance"])
        y_pred_proba = lightgbm_model.predict_proba(X_last)[0][1]
        prediction = int(y_pred_proba >= 0.5)

        # Prediksi LSTM
        sequence = create_lstm_sequences(df)
        lstm_pred = lstm_model.predict(sequence, verbose=0)[0][0]
        combined_proba = round((y_pred_proba + lstm_pred) / 2, 3)

        current_price = last_row["close"].values[0]
        atr = last_row["atr"].values[0]
        support = last_row["support"].values[0]
        resistance = last_row["resistance"].values[0]

        # Kirim sinyal hanya jika valid
        handle_signal(
            stock=ticker,
            current_price=current_price,
            atr=atr,
            support=support,
            resistance=resistance,
            prediction=prediction,
            probability=combined_proba
        )

if __name__ == "__main__":
    main()
