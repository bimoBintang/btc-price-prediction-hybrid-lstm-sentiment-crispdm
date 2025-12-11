"""
CoinGecko API Client for fetching Bitcoin price and market data.
"""
import os
import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
import time


class CoinGeckoClient:
    """
    Client for CoinGecko API to fetch cryptocurrency market data.
    
    Features:
    - Fetch current BTC price
    - Fetch historical OHLCV data
    - Fetch market cap and volume
    - Rate limiting support
    """
    
    BASE_URL = "https://api.coingecko.com/api/v3"
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize CoinGecko client.
        
        Args:
            api_key: Optional API key for Pro tier (higher rate limits)
        """
        self.api_key = api_key or os.getenv('COINGECKO_API_KEY')
        self.session = requests.Session()
        self.last_request_time = 0
        self.rate_limit_delay = 1.5  # seconds between requests (free tier: ~50/min)
        
        if self.api_key:
            self.session.headers.update({'x-cg-pro-api-key': self.api_key})
            self.rate_limit_delay = 0.1  # Pro tier has higher limits
    
    def _rate_limit(self):
        """Apply rate limiting between requests."""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - elapsed)
        self.last_request_time = time.time()
    
    def _make_request(self, endpoint: str, params: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Make a request to CoinGecko API with rate limiting.
        
        Args:
            endpoint: API endpoint
            params: Query parameters
            
        Returns:
            JSON response as dictionary
        """
        self._rate_limit()
        url = f"{self.BASE_URL}/{endpoint}"
        
        try:
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"[CoinGecko] API Error: {e}")
            return {}
    
    def get_current_price(self, coin_id: str = 'bitcoin', vs_currency: str = 'usd') -> Dict[str, Any]:
        """
        Get current price for a cryptocurrency.
        
        Args:
            coin_id: CoinGecko coin ID (default: 'bitcoin')
            vs_currency: Target currency (default: 'usd')
            
        Returns:
            Dict with price info
        """
        params = {
            'ids': coin_id,
            'vs_currencies': vs_currency,
            'include_24hr_change': 'true',
            'include_24hr_vol': 'true',
            'include_market_cap': 'true'
        }
        
        data = self._make_request('simple/price', params)
        
        if coin_id in data:
            return {
                'coin_id': coin_id,
                'price': data[coin_id].get(vs_currency, 0),
                'market_cap': data[coin_id].get(f'{vs_currency}_market_cap', 0),
                'volume_24h': data[coin_id].get(f'{vs_currency}_24h_vol', 0),
                'change_24h': data[coin_id].get(f'{vs_currency}_24h_change', 0),
                'timestamp': datetime.now()
            }
        return {}
    
    def get_market_data(self, coin_id: str = 'bitcoin') -> Dict[str, Any]:
        """
        Get detailed market data for a cryptocurrency.
        
        Args:
            coin_id: CoinGecko coin ID
            
        Returns:
            Dict with detailed market information
        """
        data = self._make_request(f'coins/{coin_id}', {
            'localization': 'false',
            'tickers': 'false',
            'community_data': 'false',
            'developer_data': 'false'
        })
        
        if not data:
            return {}
        
        market_data = data.get('market_data', {})
        
        return {
            'coin_id': coin_id,
            'name': data.get('name'),
            'symbol': data.get('symbol', '').upper(),
            'current_price': market_data.get('current_price', {}).get('usd', 0),
            'market_cap': market_data.get('market_cap', {}).get('usd', 0),
            'market_cap_rank': market_data.get('market_cap_rank', 0),
            'total_volume': market_data.get('total_volume', {}).get('usd', 0),
            'high_24h': market_data.get('high_24h', {}).get('usd', 0),
            'low_24h': market_data.get('low_24h', {}).get('usd', 0),
            'price_change_24h': market_data.get('price_change_24h', 0),
            'price_change_percentage_24h': market_data.get('price_change_percentage_24h', 0),
            'circulating_supply': market_data.get('circulating_supply', 0),
            'total_supply': market_data.get('total_supply', 0),
            'ath': market_data.get('ath', {}).get('usd', 0),
            'ath_date': market_data.get('ath_date', {}).get('usd'),
            'timestamp': datetime.now()
        }
    
    def get_historical_prices(
        self, 
        coin_id: str = 'bitcoin', 
        vs_currency: str = 'usd',
        days: int = 365
    ) -> pd.DataFrame:
        """
        Get historical OHLCV data.
        
        Args:
            coin_id: CoinGecko coin ID
            vs_currency: Target currency
            days: Number of days of history (max: 365 for hourly, unlimited for daily)
            
        Returns:
            DataFrame with OHLCV data
        """
        params = {
            'vs_currency': vs_currency,
            'days': days,
            'interval': 'daily' if days > 90 else 'hourly'
        }
        
        data = self._make_request(f'coins/{coin_id}/ohlc', params)
        
        if not data:
            return pd.DataFrame()
        
        df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df['date'] = df['timestamp'].dt.date
        df = df.set_index('timestamp')
        
        return df
    
    def get_market_chart(
        self,
        coin_id: str = 'bitcoin',
        vs_currency: str = 'usd',
        days: int = 30
    ) -> pd.DataFrame:
        """
        Get market chart data (prices, market caps, volumes).
        
        Args:
            coin_id: CoinGecko coin ID
            vs_currency: Target currency  
            days: Number of days
            
        Returns:
            DataFrame with price, market_cap, and volume
        """
        params = {
            'vs_currency': vs_currency,
            'days': days
        }
        
        data = self._make_request(f'coins/{coin_id}/market_chart', params)
        
        if not data:
            return pd.DataFrame()
        
        # Process prices
        prices_df = pd.DataFrame(data.get('prices', []), columns=['timestamp', 'price'])
        market_caps_df = pd.DataFrame(data.get('market_caps', []), columns=['timestamp', 'market_cap'])
        volumes_df = pd.DataFrame(data.get('total_volumes', []), columns=['timestamp', 'volume'])
        
        # Merge all data
        df = prices_df.merge(market_caps_df, on='timestamp').merge(volumes_df, on='timestamp')
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df['date'] = df['timestamp'].dt.date
        df = df.set_index('timestamp')
        
        return df
    
    def get_trading_volume(
        self, 
        coin_id: str = 'bitcoin',
        days: int = 30
    ) -> pd.DataFrame:
        """
        Get trading volume history.
        
        Args:
            coin_id: CoinGecko coin ID
            days: Number of days
            
        Returns:
            DataFrame with volume data
        """
        chart_data = self.get_market_chart(coin_id=coin_id, days=days)
        
        if chart_data.empty:
            return pd.DataFrame()
        
        return chart_data[['volume', 'date']].copy()


# Convenience function for quick usage
def fetch_btc_data(days: int = 365) -> Dict[str, pd.DataFrame]:
    """
    Fetch all BTC data from CoinGecko.
    
    Args:
        days: Number of days of historical data
        
    Returns:
        Dict with 'current', 'historical', and 'volume' DataFrames
    """
    client = CoinGeckoClient()
    
    return {
        'current': client.get_current_price(),
        'market_data': client.get_market_data(),
        'historical': client.get_historical_prices(days=days),
        'chart': client.get_market_chart(days=min(days, 365)),
    }


if __name__ == "__main__":
    # Test the client
    client = CoinGeckoClient()
    
    print("=== Current Price ===")
    current = client.get_current_price()
    print(f"BTC Price: ${current.get('price', 0):,.2f}")
    
    print("\n=== Market Data ===")
    market = client.get_market_data()
    print(f"Market Cap: ${market.get('market_cap', 0):,.0f}")
    print(f"24h Volume: ${market.get('total_volume', 0):,.0f}")
    
    print("\n=== Historical Data (7 days) ===")
    hist = client.get_historical_prices(days=7)
    print(hist.head())
