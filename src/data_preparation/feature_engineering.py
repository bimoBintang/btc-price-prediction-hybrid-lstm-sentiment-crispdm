# src/data_preparation/feature_engineering.py
import pandas as pd
import numpy as np

from ta.trend import MACD, EMAIndicator, SMAIndicator
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands


class TechnicalIndicator:

    @staticmethod
    def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        print("Feature Engineering: Menambahkan indikator teknikal...")
        
        # RSI
        rsi = RSIIndicator(close=df['Close'], window=14)
        df['RSI'] = rsi.rsi()
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        macd = MACD(close=df['Close'])
        df['MACD'] = macd.macd()
        df['MACD_Signal'] =macd.macd_signal()
        ema12 = df['Close'].ewm(span=12).mean()
        ema26 = df['Close'].ewm(span=26).mean()
        df['MACD'] = ema12 - ema26
        df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()

          # Bollinger Bands
        bollinger = BollingerBands(close=df['Close'])
        df['BB_High'] = bollinger.bollinger_hband()
        df['BB_Low'] = bollinger.bollinger_lband()
        
        # Moving Averages
        df['SMA_20'] = SMAIndicator(close=df['Close'], window=20).sma_indicator()
        df['SMA_50'] = SMAIndicator(close=df['Close'], window=50).sma_indicator()
        df['EMA_12'] = EMAIndicator(close=df['Close'], window=12).ema_indicator()
        
        # Lagged Features
        for lag in [1, 3, 7]:
            df[f'Close_lag_{lag}'] = df['Close'].shift(lag)
            df[f'Volume_lag_{lag}'] = df['Volume'].shift(lag)
        
        # Volume Diskusi (sentimen)
        df['Discussion_Volume'] = df['Discussion_Volume'].fillna(0)
        
        return df