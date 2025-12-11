import os
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
import asyncio

# Internal imports
from .coingecko_client import CoinGeckoClient
from .yfinance_client import YFinanceClient

# Import existing scraper (adjust path as needed)
# import sys
# sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from data_understanding.scrape_sentiment import DataScrapingCollector, get_sentiment_data


class DataCollector:
    """
    Unified data collector that aggregates data from multiple sources:
    - Twitter (via twscrape)
    - Reddit (via snscrape)  
    - CoinGecko (price and market data)
    - YFinance (Yahoo Finance BTC-USD data)
    - News API
    
    This is the main entry point for the data collection layer of the pipeline.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the data collector.
        
        Args:
            config: Configuration dictionary with API keys and settings
        """
        self.config = config or self._load_config_from_env()
        
        # Initialize clients
        self.coingecko_client = CoinGeckoClient(
            api_key=self.config.get('coingecko_api_key')
        )
        
        # Initialize YFinance client
        self.yfinance_client = YFinanceClient(symbol="BTC-USD")
        
        self.sentiment_collector = DataScrapingCollector(self.config)
        
        # Data cache
        self._cache = {}
        self._cache_expiry = {}
        self.cache_duration = timedelta(minutes=15)
    
    def _load_config_from_env(self) -> Dict[str, Any]:
        """Load configuration from environment variables."""
        return {
            # Twitter
            'twitter_username': os.getenv('TWITTER_USERNAME'),
            'twitter_password': os.getenv('TWITTER_PASSWORD'),
            'twitter_email': os.getenv('TWITTER_EMAIL'),
            # Reddit
            'reddit_client_id': os.getenv('REDDIT_CLIENT_ID'),
            'reddit_client_secret': os.getenv('REDDIT_CLIENT_SECRET'),
            'reddit_user_agent': os.getenv('REDDIT_USER_AGENT'),
            # Instagram
            'instagram_access_token': os.getenv('INSTAGRAM_ACCESS_TOKEN'),
            'instagram_business_account_id': os.getenv('INSTAGRAM_BUSINESS_ACCOUNT_ID'),
            # Facebook
            'facebook_access_token': os.getenv('FACEBOOK_ACCESS_TOKEN'),
            'facebook_page_id': os.getenv('FACEBOOK_PAGE_ID'),
            # APIs
            'coingecko_api_key': os.getenv('COINGECKO_API_KEY'),
            'newsapi_key': os.getenv('NEWS_API_KEY'),
        }
    
    def _is_cache_valid(self, key: str) -> bool:
        """Check if cached data is still valid."""
        if key not in self._cache or key not in self._cache_expiry:
            return False
        return datetime.now() < self._cache_expiry[key]
    
    def _set_cache(self, key: str, data: Any):
        """Set data in cache with expiry."""
        self._cache[key] = data
        self._cache_expiry[key] = datetime.now() + self.cache_duration
    
    def collect_price_data(
        self, 
        coin_id: str = 'bitcoin',
        days: int = 365
    ) -> Dict[str, Any]:
        """
        Collect price and market data from CoinGecko.
        
        Args:
            coin_id: Cryptocurrency ID
            days: Number of days of historical data
            
        Returns:
            Dict containing current price, market data, and historical data
        """
        cache_key = f"price_{coin_id}_{days}"
        
        if self._is_cache_valid(cache_key):
            print(f"[DataCollector] Using cached price data for {coin_id}")
            return self._cache[cache_key]
        
        print(f"[DataCollector] Fetching price data for {coin_id}...")
        
        data = {
            'current': self.coingecko_client.get_current_price(coin_id),
            'market_data': self.coingecko_client.get_market_data(coin_id),
            'historical_ohlcv': self.coingecko_client.get_historical_prices(coin_id, days=days),
            'market_chart': self.coingecko_client.get_market_chart(coin_id, days=min(days, 365)),
            'collected_at': datetime.now()
        }
        
        self._set_cache(cache_key, data)
        return data
    
    def collect_sentiment_data(
        self,
        start_date: datetime,
        end_date: datetime,
        query: str = 'bitcoin',
        subreddit: str = 'cryptocurrency'
    ) -> pd.DataFrame:
        """
        Collect sentiment data from social media platforms.
        
        Args:
            start_date: Start date for data collection
            end_date: End date for data collection
            query: Search query
            subreddit: Reddit subreddit to scrape
            
        Returns:
            DataFrame with aggregated sentiment data
        """
        cache_key = f"sentiment_{start_date.date()}_{end_date.date()}_{query}"
        
        if self._is_cache_valid(cache_key):
            print(f"[DataCollector] Using cached sentiment data")
            return self._cache[cache_key]
        
        print(f"[DataCollector] Fetching sentiment data from {start_date.date()} to {end_date.date()}...")
        
        config = {
            **self.config,
            'start_date': start_date,
            'end_date': end_date,
            'query': query,
            'subreddit': subreddit,
            'hashtag': 'bitcoin',
            'days': (end_date - start_date).days
        }
        
        sentiment_df = get_sentiment_data(config)
        
        self._set_cache(cache_key, sentiment_df)
        return sentiment_df
    
    def collect_reddit_data(
        self,
        start_date: datetime,
        end_date: datetime,
        subreddits: Optional[List[str]] = None,
        query: str = 'bitcoin',
        limit: int = 500
    ) -> pd.DataFrame:
        """
        Collect data specifically from Reddit.
        
        Args:
            start_date: Start date
            end_date: End date
            subreddits: List of subreddits to scrape
            query: Search query
            limit: Max posts per subreddit
            
        Returns:
            DataFrame with Reddit posts and sentiment
        """
        if subreddits is None:
            subreddits = ['Bitcoin', 'CryptoCurrency', 'btc', 'BitcoinMarkets']
        
        all_data = []
        
        for subreddit in subreddits:
            try:
                df = self.sentiment_collector._scrape_reddit_sentiment(
                    start_date=start_date,
                    end_date=end_date,
                    subreddit=subreddit,
                    query=query,
                    limit=limit
                )
                if not df.empty:
                    df['subreddit'] = subreddit
                    all_data.append(df)
            except Exception as e:
                print(f"[DataCollector] Error scraping r/{subreddit}: {e}")
        
        if not all_data:
            return pd.DataFrame()
        
        return pd.concat(all_data, ignore_index=True)
    
    async def collect_twitter_data(
        self,
        start_date: datetime,
        end_date: datetime,
        query: str = 'bitcoin OR btc',
        limit: int = 1000
    ) -> pd.DataFrame:
        """
        Collect data from Twitter (async).
        
        Args:
            start_date: Start date
            end_date: End date
            query: Search query
            limit: Max tweets to collect
            
        Returns:
            DataFrame with tweets and sentiment
        """
        try:
            df = await self.sentiment_collector._scrape_twitter_sentiment(
                start_date=start_date,
                end_date=end_date,
                query=query,
                limit=limit
            )
            return df
        except Exception as e:
            print(f"[DataCollector] Error scraping Twitter: {e}")
            return pd.DataFrame()
    
    def collect_news_data(
        self,
        start_date: datetime,
        end_date: datetime,
        query: str = 'bitcoin',
        limit: int = 100
    ) -> pd.DataFrame:
        """
        Collect news articles.
        
        Args:
            start_date: Start date
            end_date: End date
            query: Search query
            limit: Max articles
            
        Returns:
            DataFrame with news articles and sentiment
        """
        try:
            df = self.sentiment_collector._scrape_news_sentiment(
                start_date=start_date,
                end_date=end_date,
                query=query,
                limit=limit
            )
            return df
        except Exception as e:
            print(f"[DataCollector] Error scraping news: {e}")
            return pd.DataFrame()
    
    def collect_yfinance_data(self, days: int = 365) -> Dict[str, Any]:
        """
        Collect price data from Yahoo Finance.
        
        Args:
            days: Number of days of historical data
            
        Returns:
            Dict with YFinance data
        """
        cache_key = f"yfinance_{days}"
        
        if self._is_cache_valid(cache_key):
            print("[DataCollector] Using cached YFinance data")
            return self._cache[cache_key]
        
        print(f"[DataCollector] Fetching YFinance data...")
        
        data = {
            'current': self.yfinance_client.get_current_price(),
            'historical': self.yfinance_client.get_historical_prices(days=days),
            'market_info': self.yfinance_client.get_market_info(),
            'source': 'yfinance',
            'collected_at': datetime.now()
        }
        
        self._set_cache(cache_key, data)
        return data
    
    def collect_all(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        days: int = 30,
        use_yfinance: bool = True
    ) -> Dict[str, Any]:
        """
        Collect all data from all sources.
        
        Args:
            start_date: Start date (default: days ago)
            end_date: End date (default: now)
            days: Number of days if dates not specified
            use_yfinance: Whether to fetch data from YFinance
            
        Returns:
            Dict with all collected data
        """
        if end_date is None:
            end_date = datetime.now()
        if start_date is None:
            start_date = end_date - timedelta(days=days)
        
        print(f"[DataCollector] Collecting all data from {start_date.date()} to {end_date.date()}")
        
        # Collect price data from CoinGecko
        price_data = self.collect_price_data(days=days)
        
        # Collect YFinance data (optional)
        yfinance_data = None
        if use_yfinance:
            yfinance_data = self.collect_yfinance_data(days=days)
        
        # Collect sentiment data
        sentiment_data = self.collect_sentiment_data(
            start_date=start_date,
            end_date=end_date
        )
        
        # Collect Reddit data
        reddit_data = self.collect_reddit_data(
            start_date=start_date,
            end_date=end_date
        )
        
        # Collect news data
        news_data = self.collect_news_data(
            start_date=start_date,
            end_date=end_date
        )
        
        return {
            'price': price_data,
            'yfinance': yfinance_data,
            'sentiment': sentiment_data,
            'reddit': reddit_data,
            'news': news_data,
            'collection_metadata': {
                'start_date': start_date,
                'end_date': end_date,
                'collected_at': datetime.now()
            }
        }


# Convenience function
def run_collection(days: int = 30) -> Dict[str, Any]:
    """
    Run full data collection.
    
    Args:
        days: Number of days of data to collect
        
    Returns:
        Dict with all collected data
    """
    collector = DataCollector()
    return collector.collect_all(days=days)


if __name__ == "__main__":
    # Test the collector
    collector = DataCollector()
    
    print("=== Testing Price Data Collection ===")
    price_data = collector.collect_price_data(days=7)
    print(f"Current BTC Price: ${price_data['current'].get('price', 0):,.2f}")
    print(f"Historical data points: {len(price_data['historical_ohlcv'])}")
    
    print("\n=== Testing YFinance Data Collection ===")
    yf_data = collector.collect_yfinance_data(days=30)
    print(f"YFinance Current Price: ${yf_data['current'].get('price', 0):,.2f}")
    print(f"YFinance Historical data points: {len(yf_data['historical'])}")

