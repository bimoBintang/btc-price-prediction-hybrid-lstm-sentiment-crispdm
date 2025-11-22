"""
Fungsi:
1. Plot Time Series: Harga Close BTC
2. Plot Net Sentiment Score Harian
3. Scatter: Sentimen vs Return(t+1) + Korelasi
4. Uji Stasioneritas (ADF Test)
5. Simpan plot ke results/
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
import os
from datetime import datetime

# --- KONFIGURASI ---
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 10)
plt.rcParams['font.size'] = 12
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

def run_eda(data: pd.DataFrame):
    """
    Jalankan EDA lengkap sesuai laporan CRISP-DM.
    
    Args:
        data (pd.DataFrame): Dataset gabungan (harga + sentimen) dengan index Date
    """
    print("\n" + "="*60)
    print("EXPLORATORY DATA ANALYSIS (EDA) - CRISP-DM: Data Understanding")
    print("="*60)
    
    # 1. Time Series: Harga Close BTC
    plt.figure()
    plt.subplot(2, 2, 1)
    plt.plot(data['Close'], label='Harga Close BTC', color='blue', linewidth=1.5)
    plt.title('Time Series Harga Close BTC (USD)')
    plt.xlabel('Tanggal')
    plt.ylabel('Harga (USD)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 2. Net Sentiment Score Harian
    plt.subplot(2, 2, 2)
    plt.plot(data['Net_Sentiment_Score'], color='orange', linewidth=1.5)
    plt.axhline(0, color='red', linestyle='--', linewidth=1, label='Netral')
    plt.title('Net Sentiment Score Harian (Twitter + Reddit)')
    plt.xlabel('Tanggal')
    plt.ylabel('Skor Sentimen (-1 hingga +1)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 3. Korelasi: Sentimen vs Return(t+1)
    returns = data['Close'].pct_change().shift(-1)  # Return hari berikutnya
    correlation = returns.corr(data['Net_Sentiment_Score'])
    
    plt.subplot(2, 2, 3)
    plt.scatter(data['Net_Sentiment_Score'], returns, alpha=0.6, color='green', s=20)
    plt.title(f'Sentimen vs Return Harga (t+1)\nKorelasi: {correlation:.4f}')
    plt.xlabel('Net Sentiment Score')
    plt.ylabel('Return Harga (t+1)')
    plt.axhline(0, color='black', linewidth=0.8)
    plt.axvline(0, color='black', linewidth=0.8)
    plt.grid(True, alpha=0.3)

    # 4. Histogram Volume Diskusi
    plt.subplot(2, 2, 4)
    plt.hist(data['Discussion_Volume'], bins=50, color='purple', alpha=0.7, edgecolor='black')
    plt.title('Distribusi Volume Diskusi Harian')
    plt.xlabel('Jumlah Post/Tweet + Komentar')
    plt.ylabel('Frekuensi')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(RESULTS_DIR, "eda_comprehensive.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot EDA disimpan: {plot_path}")

    # --- UJI STASIONERITAS (ADF TEST) ---
    print("\n" + "-"*50)
    print("UJI STASIONERITAS: Augmented Dickey-Fuller (ADF) Test")
    print("-"*50)
    
    adf_result = adfuller(data['Close'].dropna())
    print(f"ADF Statistic     : {adf_result[0]:.6f}")
    print(f"p-value           : {adf_result[1]:.6f}")
    print(f"Lags Used         : {adf_result[2]}")
    print(f"Observations      : {adf_result[3]}")
    
    if adf_result[1] < 0.05:
        print("KESIMPULAN: Data STASIONER (p-value < 0.05) → Tidak perlu differencing", {adf_result[1] < 0.05})
    else:
        print("KESIMPULAN: Data TIDAK STASIONER (p-value >= 0.05) → Perlu differencing", {adf_result[1] >= 0.05})

    # --- STATISTIK DESKRIPTIF ---
    print("\n" + "-"*50)
    print("STATISTIK DESKRIPTIF")
    print("-"*50)
    stats = {
        'Periode Data': f"{data.index.min().date()} hingga {data.index.max().date()}",
        'Jumlah Hari': len(data),
        'Rata-rata Harga Close': f"${data['Close'].mean():,.2f}",
        'Std Dev Harga': f"${data['Close'].std():,.2f}",
        'Rata-rata Net Sentiment': f"{data['Net_Sentiment_Score'].mean():.4f}",
        'Std Dev Sentiment': f"{data['Net_Sentiment_Score'].std():.4f}",
        'Korelasi Sentimen vs Return(t+1)': f"{correlation:.4f}",
        'Total Volume Diskusi': f"{data['Discussion_Volume'].sum():,}"
    }
    
    for k, v in stats.items():
        print(f"{k:35}: {v}")

    # Simpan laporan teks
    report_path = os.path.join(RESULTS_DIR, "eda_report.txt")
    with open(report_path, 'w') as f:
        f.write("LAPORAN EXPLORATORY DATA ANALYSIS (EDA)\n")
        f.write(f"Dibuat pada: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Author: Bimo Bintang Siswanto (2370231011)\n")
        f.write("="*60 + "\n\n")
        f.write("STATISTIK DESKRIPTIF\n")
        f.write("-"*40 + "\n")
        for k, v in stats.items():
            f.write(f"{k:35}: {v}\n")
        f.write("\nUJI ADF\n")
        f.write("-"*40 + "\n")
        f.write(f"ADF Statistic: {adf_result[0]:.6f}\n")
        f.write(f"p-value: {adf_result[1]:.6f}\n")
        f.write(f"Kesimpulan: {'STASIONER' if adf_result[1] < 0.05 else 'TIDAK STASIONER'}\n")
    
    print(f"\nLaporan teks disimpan: {report_path}")
    print("\nEDA SELESAI. Siap lanjut ke Data Preparation.")
    print("="*60)