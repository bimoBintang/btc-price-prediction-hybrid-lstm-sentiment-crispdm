"""
Data Processor for cleaning, preprocessing, and feature engineering.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple
import os
import sys

# Add parent path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


class DataProcessor:
    """
    Data processing pipeline for BTC price prediction.
    
    Features:
    - Data cleaning and validation
    - Missing value handling
    - Feature engineering (technical indicators, lagged features)
    - Sentiment score processing
    - Data normalization
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the data processor.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        
        # Default parameters
        self.sma_windows = self.config.get('sma_windows', [7, 20, 50, 200])
        self.ema_windows = self.config.get('ema_windows', [12, 26])
        self.rsi_window = self.config.get('rsi_window', 14)
        self.lag_periods = self.config.get('lag_periods', [1, 3, 7, 14])
    
    # ==================== CLEANING ====================
    
    def clean_price_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and validate price data.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            Cleaned DataFrame
        """
        if df.empty:
            return df
        
        df = df.copy()
        
        # Ensure required columns
        required_cols = ['open', 'high', 'low', 'close']
        for col in required_cols:
            if col not in df.columns:
                print(f"[DataProcessor] Warning: Missing column '{col}'")
                return pd.DataFrame()
        
        # Remove duplicates
        if not isinstance(df.index, pd.DatetimeIndex):
            if 'timestamp' in df.columns:
                df = df.set_index('timestamp')
            elif 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                df = df.set_index('date')
        
        df = df[~df.index.duplicated(keep='first')]
        
        # Sort by date
        df = df.sort_index()
        
        # Fix OHLC inconsistencies
        df['high'] = df[['open', 'high', 'low', 'close']].max(axis=1)
        df['low'] = df[['open', 'high', 'low', 'close']].min(axis=1)
        
        # Handle missing values
        df = self._handle_missing_values(df)
        
        # Remove outliers (optional - 3 std from mean)
        for col in ['close', 'volume']:
            if col in df.columns:
                df = self._remove_outliers(df, col, n_std=3)
        
        return df
    
    def clean_sentiment_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean sentiment data.
        
        Args:
            df: DataFrame with sentiment data
            
        Returns:
            Cleaned DataFrame
        """
        if df.empty:
            return df
        
        df = df.copy()
        
        # Ensure Date column is datetime
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.set_index('Date')
        
        # Remove duplicates
        df = df[~df.index.duplicated(keep='first')]
        
        # Sort by date
        df = df.sort_index()
        
        # Clip sentiment scores to valid range [-1, 1]
        if 'Net_Sentiment_Score' in df.columns:
            df['Net_Sentiment_Score'] = df['Net_Sentiment_Score'].clip(-1, 1)
        
        # Fill missing sentiment with neutral (0)
        sentiment_cols = [c for c in df.columns if 'sentiment' in c.lower() or 'score' in c.lower()]
        for col in sentiment_cols:
            df[col] = df[col].fillna(0)
        
        return df
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values using forward fill then backward fill."""
        # Forward fill first
        df = df.ffill()
        # Backward fill for any remaining NaNs at the start
        df = df.bfill()
        return df
    
    def _remove_outliers(self, df: pd.DataFrame, column: str, n_std: float = 3) -> pd.DataFrame:
        """Remove outliers beyond n standard deviations."""
        if column not in df.columns:
            return df
        
        mean = df[column].mean()
        std = df[column].std()
        
        lower = mean - n_std * std
        upper = mean + n_std * std
        
        return df[(df[column] >= lower) & (df[column] <= upper)]
    
    # ==================== FEATURE ENGINEERING ====================
    
    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add technical indicators to price data.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with technical indicators
        """
        if df.empty or 'close' not in df.columns:
            return df
        
        df = df.copy()
        
        # Simple Moving Averages
        for window in self.sma_windows:
            df[f'sma_{window}'] = df['close'].rolling(window=window).mean()
        
        # Exponential Moving Averages
        for window in self.ema_windows:
            df[f'ema_{window}'] = df['close'].ewm(span=window, adjust=False).mean()
        
        # RSI
        df['rsi_14'] = self._calculate_rsi(df['close'], self.rsi_window)
        
        # MACD
        ema_12 = df['close'].ewm(span=12, adjust=False).mean()
        ema_26 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # Bollinger Bands
        df['bollinger_middle'] = df['close'].rolling(window=20).mean()
        bb_std = df['close'].rolling(window=20).std()
        df['bollinger_upper'] = df['bollinger_middle'] + (bb_std * 2)
        df['bollinger_lower'] = df['bollinger_middle'] - (bb_std * 2)
        df['bollinger_width'] = (df['bollinger_upper'] - df['bollinger_lower']) / df['bollinger_middle']
        
        # Average True Range (ATR)
        df['atr_14'] = self._calculate_atr(df, 14)
        
        # Price momentum
        df['momentum_7'] = df['close'].pct_change(periods=7)
        df['momentum_14'] = df['close'].pct_change(periods=14)
        
        # Volatility
        df['volatility_7'] = df['close'].rolling(window=7).std() / df['close'].rolling(window=7).mean()
        df['volatility_14'] = df['close'].rolling(window=14).std() / df['close'].rolling(window=14).mean()
        
        # Price position relative to range
        df['price_position'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-8)
        
        return df
    
    def add_lagged_features(self, df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Add lagged features for time series prediction.
        
        Args:
            df: DataFrame
            columns: Columns to create lags for (default: ['close', 'volume'])
            
        Returns:
            DataFrame with lagged features
        """
        if df.empty:
            return df
        
        df = df.copy()
        
        if columns is None:
            columns = ['close']
            if 'volume' in df.columns:
                columns.append('volume')
        
        for col in columns:
            if col not in df.columns:
                continue
            for lag in self.lag_periods:
                df[f'{col}_lag_{lag}'] = df[col].shift(lag)
                # Also add percentage change from lagged value
                df[f'{col}_pct_change_{lag}'] = (df[col] - df[f'{col}_lag_{lag}']) / (df[f'{col}_lag_{lag}'] + 1e-8)
        
        return df
    
    def add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add time-based features.
        
        Args:
            df: DataFrame with DatetimeIndex
            
        Returns:
            DataFrame with time features
        """
        if df.empty:
            return df
        
        df = df.copy()
        
        if not isinstance(df.index, pd.DatetimeIndex):
            return df
        
        # Day of week (0 = Monday, 6 = Sunday)
        df['day_of_week'] = df.index.dayofweek
        
        # Is weekend
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        
        # Month
        df['month'] = df.index.month
        
        # Day of month
        df['day_of_month'] = df.index.day
        
        # Quarter
        df['quarter'] = df.index.quarter
        
        # Week of year
        df['week_of_year'] = df.index.isocalendar().week.astype(int)
        
        return df
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate Relative Strength Index."""
        delta = prices.diff()
        
        gain = delta.where(delta > 0, 0)
        loss = (-delta).where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=window).mean()
        avg_loss = loss.rolling(window=window).mean()
        
        rs = avg_gain / (avg_loss + 1e-8)
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _calculate_atr(self, df: pd.DataFrame, window: int = 14) -> pd.Series:
        """Calculate Average True Range."""
        high = df['high']
        low = df['low']
        close = df['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.rolling(window=window).mean()
        
        return atr
    
    # ==================== MERGE & COMBINE ====================
    
    def merge_price_and_sentiment(
        self, 
        price_df: pd.DataFrame, 
        sentiment_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Merge price data with sentiment data.
        
        Args:
            price_df: DataFrame with price data
            sentiment_df: DataFrame with sentiment data
            
        Returns:
            Merged DataFrame
        """
        if price_df.empty:
            return price_df
        
        price_df = price_df.copy()
        
        # Ensure both have date index
        if not isinstance(price_df.index, pd.DatetimeIndex):
            if 'date' in price_df.columns:
                price_df['date'] = pd.to_datetime(price_df['date'])
                price_df = price_df.set_index('date')
        
        if sentiment_df.empty:
            # Add placeholder sentiment columns
            price_df['net_sentiment_score'] = 0.0
            price_df['discussion_volume'] = 0
            price_df['engagement'] = 0
            return price_df
        
        sentiment_df = sentiment_df.copy()
        if not isinstance(sentiment_df.index, pd.DatetimeIndex):
            if 'Date' in sentiment_df.columns:
                sentiment_df['Date'] = pd.to_datetime(sentiment_df['Date'])
                sentiment_df = sentiment_df.set_index('Date')
        
        # Normalize index to date only (remove time component)
        price_df.index = pd.to_datetime(price_df.index.date)
        sentiment_df.index = pd.to_datetime(sentiment_df.index.date)
        
        # Merge on date
        merged = price_df.join(sentiment_df, how='left')
        
        # Fill missing sentiment with neutral values
        if 'Net_Sentiment_Score' in merged.columns:
            merged['Net_Sentiment_Score'] = merged['Net_Sentiment_Score'].fillna(0)
        if 'Discussion_Volume' in merged.columns:
            merged['Discussion_Volume'] = merged['Discussion_Volume'].fillna(0)
        if 'Engagement' in merged.columns:
            merged['Engagement'] = merged['Engagement'].fillna(0)
        
        return merged
    
    # ==================== TARGET VARIABLE ====================
    
    def add_target_variable(
        self, 
        df: pd.DataFrame, 
        target_col: str = 'close',
        horizon: int = 1
    ) -> pd.DataFrame:
        """
        Add target variable for prediction.
        
        Args:
            df: DataFrame
            target_col: Column to predict
            horizon: Prediction horizon (days ahead)
            
        Returns:
            DataFrame with target variable
        """
        if df.empty or target_col not in df.columns:
            return df
        
        df = df.copy()
        
        # Next day close price
        df['target_price'] = df[target_col].shift(-horizon)
        
        # Price change (raw)
        df['target_change'] = df['target_price'] - df[target_col]
        
        # Price change (percentage)
        df['target_pct_change'] = (df['target_change'] / df[target_col]) * 100
        
        # Direction (binary: 1 = up, 0 = down)
        df['target_direction'] = (df['target_change'] > 0).astype(int)
        
        return df
    
    # ==================== FULL PIPELINE ====================
    
    def process_full_pipeline(
        self,
        price_data: pd.DataFrame,
        sentiment_data: Optional[pd.DataFrame] = None,
        add_targets: bool = True
    ) -> pd.DataFrame:
        """
        Run the full data processing pipeline.
        
        Args:
            price_data: Raw price data
            sentiment_data: Raw sentiment data
            add_targets: Whether to add target variables
            
        Returns:
            Fully processed DataFrame ready for modeling
        """
        print("[DataProcessor] Starting full processing pipeline...")
        
        # Step 1: Clean price data
        print("[DataProcessor] Cleaning price data...")
        df = self.clean_price_data(price_data)
        
        if df.empty:
            print("[DataProcessor] Error: No data after cleaning")
            return df
        
        # Step 2: Add technical indicators
        print("[DataProcessor] Adding technical indicators...")
        df = self.add_technical_indicators(df)
        
        # Step 3: Add lagged features
        print("[DataProcessor] Adding lagged features...")
        df = self.add_lagged_features(df)
        
        # Step 4: Add time features
        print("[DataProcessor] Adding time features...")
        df = self.add_time_features(df)
        
        # Step 5: Merge with sentiment data
        if sentiment_data is not None:
            print("[DataProcessor] Merging with sentiment data...")
            sentiment_data = self.clean_sentiment_data(sentiment_data)
            df = self.merge_price_and_sentiment(df, sentiment_data)
        
        # Step 6: Add target variable
        if add_targets:
            print("[DataProcessor] Adding target variables...")
            df = self.add_target_variable(df)
        
        # Step 7: Handle any remaining NaNs
        print("[DataProcessor] Handling missing values...")
        df = df.dropna()
        
        print(f"[DataProcessor] Pipeline complete. Final shape: {df.shape}")
        
        return df


if __name__ == "__main__":
    # Test the processor
    import numpy as np
    
    # Create sample data
    dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
    sample_data = pd.DataFrame({
        'open': np.random.uniform(40000, 45000, 100),
        'high': np.random.uniform(41000, 46000, 100),
        'low': np.random.uniform(39000, 44000, 100),
        'close': np.random.uniform(40000, 45000, 100),
        'volume': np.random.uniform(1e9, 5e9, 100)
    }, index=dates)
    
    processor = DataProcessor()
    processed = processor.process_full_pipeline(sample_data)
    
    print("\nProcessed columns:")
    print(processed.columns.tolist())
    print(f"\nShape: {processed.shape}")
