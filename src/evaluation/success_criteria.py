from config.settings import MIN_MAE_IMPROVEMENT, MIN_SHAP_SENTIMENT

def check_success(improvement_pct, shap_sentiment):
    print("\n" + "="*60)
    print("PENILAIAN KRITERIA KESEHASILAN PROYEK")
    print("="*60)
    
    mae_ok = improvement_pct >= MIN_MAE_IMPROVEMENT
    shap_ok = shap_sentiment > MIN_SHAP_SENTIMENT
    
    if mae_ok and shap_ok:
        print("PROYEK DINYATAKAN SUKSES!")
        print("Model Hybrid LSTM + Sentimen SIAP DEPLOYMENT")
    else:
        print("Belum memenuhi semua kriteria.")
        if not mae_ok:
            print(f"→ MAE hanya turun {improvement_pct:.2f}% (< {MIN_MAE_IMPROVEMENT}%)")
        if not shap_ok:
            print(f"→ Kontribusi sentimen lemah (SHAP = {shap_sentiment:.6f})")
    print("="*60)