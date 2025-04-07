import os
import pandas as pd
from train_model import train_lightgbm_model
from model_predictor import predict_signal
from signal_handler import send_signal_to_telegram
from data_fetcher import download_data, calculate_indicators

# Daftar saham yang akan dianalisis
STOCK_LIST = [
    "ACES.JK", "ADMR.JK", "ADRO.JK", "AKRA.JK", "AMMN.JK", "AMRT.JK", "ANTM.JK",
    "ARTO.JK", "ASII.JK", "AUTO.JK", "AVIA.JK", "BBCA.JK", "BBNI.JK", "BBRI.JK",
    "BBTN.JK", "BBYB.JK", "BDKR.JK", "BFIN.JK", "BMRI.JK", "BMTR.JK", "BNGA.JK",
    "BRIS.JK", "BRMS.JK", "BRPT.JK", "BSDE.JK", "BTPS.JK", "CMRY.JK", "CPIN.JK",
    "CTRA.JK", "DEWA.JK", "DSNG.JK", "ELSA.JK", "EMTK.JK", "ENRG.JK", "ERAA.JK",
    "ESSA.JK", "EXCL.JK", "FILM.JK", "GGRM.JK", "GJTL.JK", "GOTO.JK", "HEAL.JK",
    "HMSP.JK", "HRUM.JK", "ICBP.JK", "INCO.JK", "INDF.JK", "INDY.JK", "INKP.JK",
    "INTP.JK", "ISAT.JK", "ITMG.JK", "JPFA.JK", "JSMR.JK", "KLBF.JK", "MDKA.JK",
    "MEDC.JK", "MIKA.JK", "MNCN.JK", "MTEL.JK", "MYOR.JK", "NCKL.JK", "PGAS.JK",
    "PNLF.JK", "PTBA.JK", "PTPP.JK", "PWON.JK", "ROTI.JK", "SAME.JK", "SCMA.JK",
    "SIDO.JK", "SILO.JK", "SMGR.JK", "SMRA.JK", "TBIG.JK", "TINS.JK", "TKIM.JK",
    "TLKM.JK", "TOWR.JK", "TPIA.JK", "UNTR.JK", "UNVR.JK", "WIKA.JK", "WSKT.JK",
    "WTON.JK"
] # Contoh daftar saham


def run_day_trading_bot():
    print("🚀 Mulai Bot Trading AI untuk Day Trading")

    for symbol in STOCK_LIST:
        try:
            print(f"\n🔍 Memproses {symbol}...")
            
            # Ambil data historis + indikator
            df = download_data(symbol)
            df = calculate_indicators(df)

            if df is None or df.empty:
                print(f"⚠️ Data kosong untuk {symbol}")
                continue

            # Simpan backup data
            df.to_csv(f"data/{symbol}_data.csv", index=True)

            # Train model & prediksi sinyal
            model = train_lightgbm_model(df)
            signal = predict_signal(df, model)

            # Kirim sinyal ke Telegram
            send_signal_to_telegram(symbol, signal)

        except Exception as e:
            print(f"❌ Error saat proses {symbol}: {e}")

if __name__ == "__main__":
    run_day_trading_bot()
