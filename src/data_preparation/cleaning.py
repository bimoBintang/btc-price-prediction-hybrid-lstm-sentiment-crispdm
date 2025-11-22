# src/data_preparation/cleaning.py
import pandas as pd
import numpy as np

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    print("\nDATA PREPARATION - Cleaning & Preprocessing")
    initial = len(df)
    
    # 1. Hapus duplikat
    df = df[~df.index.duplicated(keep='first')]
    
    # 2. Interpolasi missing values (sentimen bisa kosong)
    df['Net_Sentiment_Score'] = df['Net_Sentiment_Score'].interpolate(method='linear')
    df['Discussion_Volume'] = df['Discussion_Volume'].fillna(0)
    
    # 3. Hapus baris dengan NaN di harga
    df = df.dropna(subset=['Close'])
    
    cleaned = len(df)
    print(f"Cleaning selesai: {initial} → {cleaned} hari (-{initial-cleaned})")
    return df