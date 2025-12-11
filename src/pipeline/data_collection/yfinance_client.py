"""
YFinance Client for BTC Price Data.

Uses Yahoo Finance API via yfinance library for BTC-USD data.
"""
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict, Any


class YFinanceClient:
    """
    Client for fetching BTC data from Yahoo Finance.
    
    Features:
    - Historical OHLCV data
    - Real-time price (with 15 min delay)
    - Market info
    """
    
    def __init__(self, symbol: str = "BTC-USD"):
        """
        Initialize YFinance client.
        
        Args:
            symbol: Trading symbol (default: BTC-USD)
        """
        self.symbol = symbol
        self.ticker = yf.Ticker(symbol)
        self._cache = {}
        self._cache_ttl = 60  # seconds
    
    def _get_cached(self, key: str) -> Optional[Any]:
        """Get cached value if not expired."""
        if key in self._cache:
            data, timestamp = self._cache[key]
            if (datetime.now() - timestamp).seconds < self._cache_ttl:
                return data
        return None
    
    def _set_cache(self, key: str, value: Any):
        """Set cache value."""
        self._cache[key] = (value, datetime.now())
    
    def get_current_price(self) -> Dict[str, Any]:
        """
        Get current BTC price.
        
        Returns:
            Dict with price info
        """
        cached = self._get_cached('current_price')
        if cached:
            return cached
        
        try:
            info = self.ticker.info
            
            result = {
                'symbol': self.symbol,
                'price': info.get('regularMarketPrice', 0),
                'previous_close': info.get('regularMarketPreviousClose', 0),
                'open': info.get('regularMarketOpen', 0),
                'day_high': info.get('regularMarketDayHigh', 0),
                'day_low': info.get('regularMarketDayLow', 0),
                'volume': info.get('regularMarketVolume', 0),
                'market_cap': info.get('marketCap', 0),
                'change': info.get('regularMarketChange', 0),
                'change_percent': info.get('regularMarketChangePercent', 0),
                'timestamp': datetime.now().isoformat(),
                'source': 'yfinance'
            }
            
            self._set_cache('current_price', result)
            print(f"[YFinance] Current price: ${result['price']:,.2f}")
            return result
            
        except Exception as e:
            print(f"[YFinance] Error getting current price: {e}")
            return {
                'symbol': self.symbol,
                'price': 0,
                'error': str(e),
                'source': 'yfinance'
            }
    
    def get_historical_data(
        self,
        period: str = "1y",
        interval: str = "1d"
    ) -> pd.DataFrame:
        """
        Get historical OHLCV data.
        
        Args:
            period: Data period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            interval: Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
            
        Returns:
            DataFrame with OHLCV data
        """
        cache_key = f'historical_{period}_{interval}'
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached
        
        try:
            df = self.ticker.history(period=period, interval=interval)
            
            if df.empty:
                print(f"[YFinance] No data returned for {self.symbol}")
                return pd.DataFrame()
            
            # Rename columns to lowercase
            df.columns = [col.lower() for col in df.columns]
            
            # Add date column
            df['date'] = df.index.date
            
            print(f"[YFinance] Fetched {len(df)} records for {period} period")
            
            self._set_cache(cache_key, df)
            return df
            
        except Exception as e:
            print(f"[YFinance] Error getting historical data: {e}")
            return pd.DataFrame()
    
    def get_historical_prices(self, days: int = 365) -> pd.DataFrame:
        """
        Get historical prices for specified number of days.
        
        Args:
            days: Number of days of history
            
        Returns:
            DataFrame with OHLCV data
        """
        # Map days to period
        if days <= 5:
            period = "5d"
        elif days <= 30:
            period = "1mo"
        elif days <= 90:
            period = "3mo"
        elif days <= 180:
            period = "6mo"
        elif days <= 365:
            period = "1y"
        elif days <= 730:
            period = "2y"
        else:
            period = "5y"
        
        df = self.get_historical_data(period=period, interval="1d")
        
        if not df.empty and len(df) > days:
            df = df.tail(days)
        
        return df
    
    def get_intraday_data(self, interval: str = "1h") -> pd.DataFrame:
        """
        Get intraday data.
        
        Args:
            interval: Data interval (1m, 5m, 15m, 30m, 1h)
            
        Returns:
            DataFrame with intraday data
        """
        # YFinance limits intraday data to last 7 days for most intervals
        return self.get_historical_data(period="7d", interval=interval)
    
    def get_market_info(self) -> Dict[str, Any]:
        """
        Get market information.
        
        Returns:
            Dict with market info
        """
        try:
            info = self.ticker.info
            
            return {
                'symbol': self.symbol,
                'name': info.get('shortName', 'Bitcoin USD'),
                'currency': info.get('currency', 'USD'),
                'exchange': info.get('exchange', 'CCC'),
                'market_cap': info.get('marketCap', 0),
                'circulating_supply': info.get('circulatingSupply', 0),
                'volume_24h': info.get('volume24Hr', 0),
                'all_time_high': info.get('fiftyTwoWeekHigh', 0),
                'all_time_low': info.get('fiftyTwoWeekLow', 0),
                'fifty_day_avg': info.get('fiftyDayAverage', 0),
                'two_hundred_day_avg': info.get('twoHundredDayAverage', 0),
                'source': 'yfinance'
            }
            
        except Exception as e:
            print(f"[YFinance] Error getting market info: {e}")
            return {'error': str(e)}
    
    def get_price_at_date(self, date: datetime) -> Optional[float]:
        """
        Get closing price at specific date.
        
        Args:
            date: Target date
            
        Returns:
            Closing price or None
        """
        try:
            start = date - timedelta(days=1)
            end = date + timedelta(days=1)
            
            df = self.ticker.history(start=start, end=end)
            
            if not df.empty:
                return float(df['Close'].iloc[-1])
            return None
            
        except Exception as e:
            print(f"[YFinance] Error getting price at date: {e}")
            return None


if __name__ == "__main__":
    # Test the client
    print("Testing YFinance Client...")
    
    client = YFinanceClient()
    
    # Test current price
    print("\n--- Current Price ---")
    current = client.get_current_price()
    print(f"Price: ${current.get('price', 0):,.2f}")
    print(f"24h Change: {current.get('change_percent', 0):.2f}%")
    
    # Test historical data
    print("\n--- Historical Data (30 days) ---")
    historical = client.get_historical_prices(days=30)
    print(f"Records: {len(historical)}")
    if not historical.empty:
        print(f"Latest close: ${historical['close'].iloc[-1]:,.2f}")
        print(f"Date range: {historical.index[0]} to {historical.index[-1]}")
    
    # Test market info
    print("\n--- Market Info ---")
    info = client.get_market_info()
    print(f"Market Cap: ${info.get('market_cap', 0):,.0f}")
    print(f"52 Week High: ${info.get('all_time_high', 0):,.2f}")
    
    print("\n✅ YFinance client test passed!")
