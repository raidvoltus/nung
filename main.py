import os
import pandas as pd
import numpy as np
import time
from datetime import datetime
from data_fetcher import fetch_data_with_indicators
from model_predictor import predict_signal
from signal_handler import send_signal_to_telegram
from train_model import retrain_model

print("\n🚀 Mulai Bot Trading AI untuk Day Trading\n")

# Ambil daftar saham (dapat diupdate sesuai kebutuhan)
tickers = [
    'ACES.JK', 'ADMR.JK', 'ADRO.JK', 'AKRA.JK', 'AMMN.JK', 'AMRT.JK', 'ANTM.JK', 'ARTO.JK',
    'ASII.JK', 'AUTO.JK', 'AVIA.JK', 'BBCA.JK', 'BBNI.JK', 'BBRI.JK', 'BBTN.JK', 'BBYB.JK',
    'BDKR.JK', 'BFIN.JK', 'BMRI.JK', 'BMTR.JK', 'BNGA.JK', 'BRIS.JK', 'BRMS.JK', 'BRPT.JK',
    'BSDE.JK', 'BTPS.JK', 'CMRY.JK', 'CPIN.JK', 'CTRA.JK', 'DEWA.JK', 'DSNG.JK', 'ELSA.JK',
    'EMTK.JK', 'ENRG.JK', 'ERAA.JK', 'ESSA.JK', 'EXCL.JK', 'FILM.JK', 'GGRM.JK', 'GJTL.JK',
    'GOTO.JK', 'HEAL.JK', 'HMSP.JK', 'HRUM.JK', 'ICBP.JK', 'INCO.JK', 'INDF.JK', 'INDY.JK',
    'INKP.JK', 'INTP.JK', 'ISAT.JK', 'ITMG.JK', 'JPFA.JK', 'JSMR.JK', 'KLBF.JK', 'MDKA.JK',
    'MEDC.JK', 'MIKA.JK', 'MNCN.JK', 'MTEL.JK', 'MYOR.JK', 'NCKL.JK', 'PGAS.JK', 'PNLF.JK',
    'PTBA.JK', 'PTPP.JK', 'PWON.JK', 'ROTI.JK', 'SAME.JK', 'SCMA.JK', 'SIDO.JK', 'SILO.JK',
    'SMGR.JK'
]

# Loop untuk setiap saham
for ticker in tickers:
    print(f"\n🔍 Memproses {ticker}...")

    try:
        df = fetch_data_with_indicators(ticker)

        if df is None or df.empty or len(df) < 50:
            print(f"⚠️  Data tidak cukup untuk {ticker}")
            continue

        # Pastikan semua kolom yang digunakan memiliki shape 1D
        for col in df.columns:
            if isinstance(df[col].iloc[-1], (np.ndarray, list)):
                df[col] = df[col].apply(lambda x: x[0] if isinstance(x, (np.ndarray, list)) else x)

        signal, prob, take_profit, stop_loss, rr_ratio = predict_signal(df)

        if signal == 'buy' and prob >= 0.8:
            message = f"""
📈 *Sinyal Beli Terdeteksi!*  
Saham: *{ticker}*  
Probabilitas: *{prob*100:.2f}%*  
Take Profit: *{take_profit:.2f}*  
Stop Loss: *{stop_loss:.2f}*  
Risk/Reward: *{rr_ratio:.2f}*
            """
            kirim_sinyal_telegram(message.strip())
        else:
            print(f"ℹ️  Tidak ada sinyal valid untuk {ticker} (Probabilitas: {prob:.2f})")

        time.sleep(1)

    except Exception as e:
        print(f"❌ Error saat proses {ticker}: {e}")

# Training ulang model setelah seluruh proses
print("\n🧠 Retraining Model...")
try:
    retrain_model()
    print("✅ Model berhasil diretrain!")
except Exception as e:
    print(f"❌ Error saat retrain model: {e}")
