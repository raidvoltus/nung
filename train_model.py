import os
import pandas as pd
from data_fetcher import fetch_all_stock_data
from model_predictor import create_features, train_lightgbm, train_lstm

# Pastikan folder models ada
os.makedirs("models", exist_ok=True)

# Cek apakah model sudah ada
lightgbm_path = "models/lightgbm_model.pkl"
lstm_path = "models/lstm_model.h5"

if not os.path.exists(lightgbm_path) or not os.path.exists(lstm_path):
    print("\n⚠️ Model belum ditemukan. Training sekarang...\n")

    # Ambil data
    df = fetch_all_stock_data()

    # Buat fitur dan target
    df = create_features(df)

    # Simpan data training untuk dokumentasi
    df.to_csv("data/training_data.csv", index=False)

    # Tentukan fitur yang digunakan
    features = [
        'Open', 'High', 'Low', 'Close', 'Volume',
        'SMA_5', 'EMA_5', 'RSI_14', 'MACD', 'MACD_Signal',
        'Upper_BB', 'Lower_BB', 'Stoch_K', 'Stoch_D',
        'ATR_14', 'OBV', 'Fib_0', 'Fib_0.5', 'Fib_1',
        'Parabolic_SAR', 'Ichimoku_Cloud'
    ]
    
    # Latih model
    train_lightgbm(df, features)
    train_lstm(df, features)
    
    print("\n✅ Training selesai. Model tersimpan di folder 'models/'\n")

else:
    print("\n✅ Model sudah tersedia. Lewati training...\n")
