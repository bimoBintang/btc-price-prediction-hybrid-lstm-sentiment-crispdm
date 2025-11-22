# src/data_preparation/feature_engineering.py
import pandas as pd
import numpy as np

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    print("Feature Engineering: Menambahkan indikator teknikal...")
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    ema12 = df['Close'].ewm(span=12).mean()
    ema26 = df['Close'].ewm(span=26).mean()
    df['MACD'] = ema12 - ema26
    df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
    
    # Moving Averages
    df['MA7'] = df['Close'].rolling(7).mean()
    df['MA21'] = df['Close'].rolling(21).mean()
    
    # Lagged Features
    for lag in [1, 3, 7]:
        df[f'Close_lag_{lag}'] = df['Close'].shift(lag)
        df[f'Volume_lag_{lag}'] = df['Volume'].shift(lag)
    
    # Volume Diskusi (sentimen)
    df['Discussion_Volume'] = df['Discussion_Volume'].fillna(0)
    
    return df