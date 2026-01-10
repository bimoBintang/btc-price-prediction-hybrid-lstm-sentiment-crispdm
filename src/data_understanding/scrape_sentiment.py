import pandas as pd
from datetime import datetime
import json
from nltk.sentiment.vader import SentimentIntensityAnalyzer 
import nltk
import os
from typing import Dict, Any
import requests

# Try to import twscrape
try:
    from twscrape import API, gather
    TWSCRAPE_AVAILABLE = True
except ImportError:
    TWSCRAPE_AVAILABLE = False
    print("[Warning] twscrape tidak terinstall → Twitter akan pakai fallback snscrape (mungkin gagal)")

# Try to import snscrape for Reddit
# Note: snscrape has compatibility issues with Python 3.12+
try:
    import snscrape.modules.reddit as snreddit
    SNSCRAPE_AVAILABLE = True
except (ImportError, AttributeError, Exception) as e:
    SNSCRAPE_AVAILABLE = False
    snreddit = None  # Define as None to avoid NameError
    print(f"[Warning] snscrape tidak kompatibel dengan Python ini: {e}")

nltk.download('vader_lexicon')

REDDIT_CLIENT_ID = os.getenv('REDDIT_CLIENT_ID', 'value')
REDDIT_CLIENT_SECRET = os.getenv('REDDIT_CLIENT_SECRET', 'value')
REDDIT_USER_AGENT = os.getenv('REDDIT_USER_AGENT', 'value')

INSTAGRAM_CLIENT_ID = os.getenv('INSTAGRAM_CLIENT_ID', 'value')
INSTAGRAM_CLIENT_SECRET = os.getenv('INSTAGRAM_CLIENT_SECRET', 'value')
INSTAGRAM_USER_AGENT = os.getenv('INSTAGRAM_USER_AGENT', 'value')

sid = SentimentIntensityAnalyzer()


def aggregate_daily(items_list, platform_name):
    """Agregasi data harian dengan sentiment analysis"""
    if not items_list:
        print(f"[{platform_name}] Tidak ada data ditemukan.")
        return pd.DataFrame()
    
    df = pd.DataFrame(items_list)
    print(f"[{platform_name}] Total scraped: {len(df)}")
    
    # Agregasi harian
    daily = df.groupby('Date').agg({
        'Text': lambda x: ' '.join(x),
        'Engagement': 'sum'
    }).reset_index()
    
    daily['Net_Sentiment_Score'] = daily['Text'].apply(analyze_sentiment)
    daily['Discussion_Volume'] = df.groupby('Date').size().values
    daily['Platform'] = platform_name
    
    return daily[['Date', 'Platform', 'Net_Sentiment_Score', 'Discussion_Volume', 'Engagement']]


def analyze_sentiment(text):
    scores = sid.polarity_scores(text)
    return scores['compound']


def get_sentiment_data(config: Dict[str, Any]) -> pd.DataFrame:
    """
    Get sentiment data from cache or scrape from multiple platforms.
    """
    json_file = 'data/raw/btc_sentiment_daily.json'
    
    # Try to load from cache first
    if os.path.exists(json_file):
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
                df = pd.DataFrame(data)
                if 'Date' in df.columns:
                    df['Date'] = pd.to_datetime(df['Date'])
                    df = df.set_index('Date')
                return df
        except Exception as e:
            print(f"Error loading cache: {e}")
    
    # Scrape from multiple platforms
    collector = DataScrapingCollector(config)
    
    twitter_df = collector._scrape_twitter_sentiment(
        start_date=config.get('start_date'), 
        end_date=config.get('end_date'), 
        query=config.get('query', 'bitcoin OR btc')
    )
    reddit_df = collector._scrape_reddit_sentiment(
        start_date=config.get('start_date'), 
        end_date=config.get('end_date'), 
        query=config.get('query', 'bitcoin'),
        subreddit=config.get('subreddit', 'cryptocurrency')
    )
    news_df = collector._scrape_news_sentiment(
        start_date=config.get('start_date'), 
        end_date=config.get('end_date'), 
        query=config.get('query', 'bitcoin'), 
        days=str(config.get('days', 7))
    )
    instagram_df = collector._scrape_instagram_sentiment(
        start_date=config.get('start_date'), 
        end_date=config.get('end_date'),
        hashtag=config.get('hashtag', 'bitcoin')
    )
    facebook_df = collector._scrape_facebook_sentiment(
        start_date=config.get('start_date'), 
        end_date=config.get('end_date'), 
        query=config.get('query', 'bitcoin')
    )

    # Gabungan semua dataframes
    df_list = [df for df in [twitter_df, reddit_df, news_df, instagram_df, facebook_df] if df is not None and not df.empty]

    if not df_list:
        print("Tidak ada data yang berhasil di-scrape!")
        return pd.DataFrame()
    
    combined = pd.concat(df_list, ignore_index=True)
    
    if 'Date' in combined.columns:
        combined['Date'] = pd.to_datetime(combined['Date'])
    
    daily = combined.groupby('Date').agg({
        'Net_Sentiment_Score': 'mean',
        'Discussion_Volume': 'sum',
        'Engagement': 'sum'
    }).reset_index()

    # Simpan ke JSON dengan format yang proper
    daily_json = daily.to_dict(orient='records')
    for record in daily_json:
        if isinstance(record['Date'], pd.Timestamp):
            record['Date'] = record['Date'].strftime('%Y-%m-%d')
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(json_file), exist_ok=True)
    
    with open(json_file, 'w') as f:
        json.dump(daily_json, f, indent=2)
    
    return daily


class DataScrapingCollector:
    """Collector class for scraping sentiment data from multiple platforms."""

    def __init__(self, config: Dict[str, Any]):
        # Twitter
        self.twitter_username = config.get('twitter_username')
        self.twitter_password = config.get('twitter_password')
        self.twitter_email = config.get('twitter_email')
        # Instagram
        self.instagram_access_token = config.get('instagram_access_token')
        self.instagram_business_account_id = config.get('instagram_business_account_id')
        # Facebook
        self.facebook_access_token = config.get('facebook_access_token')
        self.facebook_page_id = config.get('facebook_page_id')
        # News API
        self.newsapi_key = config.get('newsapi_key', os.getenv('NEWS_API_KEY', ''), default=str(os.getenv('NEWS_API_KEY', '')))

        # Target
        self.crypto_subs = [
            'Bitcoin',
            'CryptoCurrency',
            'btc',
            'BitcoinMarkets',
            'CryptoMarkets',
            'wallstreetbets'
        ]
        
        # Bitcoin keywords
        self.btc_keywords = ['bitcoin', 'btc', 'BTC', 'Bitcoin']

    async def _scrape_twitter_sentiment(self, start_date: datetime, end_date: datetime, query: str = 'bitcoin OR btc', limit: int = 1000) -> pd.DataFrame:
        """Scrape Twitter sentiment data using twscrape."""
        items_list = []

        if not TWSCRAPE_AVAILABLE:
            print("[Twitter] twscrape tidak tersedia")
            return pd.DataFrame()

        if not self.twitter_username:
            print("[Twitter] Username tidak tersedia")
            return pd.DataFrame()

        try:
            api = API()
            await api.pool.add_account(self.twitter_username, self.twitter_password, self.twitter_email, self.twitter_username)
            await api.pool.login_all()
            
            search_query = f"{query} since:{start_date.date()} until:{end_date.date()} lang:en -is:retweet"
            tweets = await gather(api.search(self=self.crypto_subs,q=search_query, limit=limit))

            for i, tweet in enumerate(tweets):
                if i >= limit:
                    break
                
                # Filter berdasarkan tanggal
                msg_date = pd.to_datetime(tweet.date).date() if tweet.date else datetime.now().date()
                if not (start_date.date() <= msg_date <= end_date.date()):
                    continue

                items_list.append({
                    'Date': tweet.date.date(),
                    'Text': tweet.rawContent or tweet.text or '',
                    'Engagement': (tweet.likeCount or 0) + (tweet.retweetCount or 0),
                    'Platform': 'Twitter'
                })
        except Exception as e:
            print(f"[Twitter] twscrape gagal: {e}")
            return pd.DataFrame()

        return aggregate_daily(items_list, 'Twitter')

    def _scrape_reddit_sentiment(self, start_date: datetime, end_date: datetime, subreddit: str = 'cryptocurrency', query: str = 'bitcoin', limit: int = 500) -> pd.DataFrame:
        """Scrape Reddit sentiment data using snscrape."""
        if not SNSCRAPE_AVAILABLE:
            print("[Reddit] snscrape tidak tersedia")
            return pd.DataFrame()

        posts_list = []
        try:
            if subreddit:
                scraper = snreddit.RedditSubredditScraper(subreddit=subreddit, mode='posts', self=self.crypto_subs)
            else:
                start_str = start_date.strftime('%Y-%m-%d')
                end_str = end_date.strftime('%Y-%m-%d')
                full_query = f'{query} since:{start_str} until:{end_str} lang:en'
                scraper = snreddit.RedditSearchScraper(full_query)

            for i, post in enumerate(scraper.get_items()):
                if i >= limit * 2:
                    break
                if post.created is None:
                    continue
                    
                post_date = pd.to_datetime(post.created, unit='s').date()
                if not (start_date.date() <= post_date <= end_date.date()):
                    continue
                    
                # Filter berdasarkan query
                text = (post.title or '') + ' ' + (post.selftext or '')
                if query.lower() not in text.lower():
                    continue

                posts_list.append({
                    'Date': post_date,
                    'Text': text,
                    'Engagement': (post.score or 0) + (post.numComments or 0),
                    'Platform': 'Reddit'
                }) 

                if (i + 1) % 100 == 0:
                    print(f"[Reddit] Scraped {i + 1} posts...")
        except Exception as e:
            print(f"[Reddit] Error: {e}")
            print("[Reddit] Note: Reddit scraping via Pushshift mungkin terbatas")
            return pd.DataFrame()
            
        return aggregate_daily(posts_list, 'Reddit')

    def _scrape_instagram_sentiment(self, start_date: datetime, end_date: datetime, hashtag: str = 'bitcoin', limit: int = 100) -> pd.DataFrame:
        """Scrape Instagram sentiment data using Official API."""
        if not self.instagram_access_token or not self.instagram_business_account_id:
            print("[Instagram Official API] Missing credentials")
            return pd.DataFrame()
        
        try:
            base_url = os.getenv('META_API_URL', 'https://graph.facebook.com/')
            url = f"{base_url}v18.0/{self.instagram_business_account_id}/media"

            posts_list = []
            params = {
                'access_token': self.instagram_access_token,
                'fields': 'id,caption,like_count,comments_count,timestamp,media_type,media_url',
                'limit': limit
            }

            response = requests.get(url=url, params=params)
            response.raise_for_status()

            data = response.json()

            if 'data' not in data:
                print("[Instagram Official API] No data returned")
                return pd.DataFrame()

            for i, post in enumerate(data['data']):
                if i >= limit:
                    break

                post_date = pd.to_datetime(post.get('timestamp')).date() if post.get('timestamp') else datetime.now().date()
                
                # Filter tanggal
                if not (start_date.date() <= post_date <= end_date.date()):
                    continue

                # Filter berdasarkan hashtag
                text = post.get('caption', '') or ''
                if hashtag.lower() not in text.lower():
                    continue

                posts_list.append({
                    'Date': post_date,
                    'Text': text,
                    'Engagement': post.get('like_count', 0) + post.get('comments_count', 0),
                    'Platform': 'Instagram'
                })
                
                # Debug: print atribut di post pertama
                if i == 0:
                    print(f"[Instagram] First post keys: {list(post.keys())}")

                if (i + 1) % 50 == 0:
                    print(f"[Instagram] Scraped {i + 1} posts...")
        except Exception as e:
            print(f"[Instagram] Error: {e}")
            print("[Instagram] Tip: Instagram scraping memerlukan login/cookies")
            return pd.DataFrame()
        
        return aggregate_daily(posts_list, 'Instagram')
    
    def _scrape_facebook_sentiment(self, start_date: datetime, end_date: datetime, query: str = 'bitcoin', limit: int = 500) -> pd.DataFrame:
        """Scrape Facebook sentiment data using Official API."""
        if not self.facebook_access_token or not self.facebook_page_id:
            print("[Facebook Official API] Missing credentials")
            return pd.DataFrame()
        
        try:
            base_url = os.getenv('META_API_URL', 'https://graph.facebook.com/')
            url = f"{base_url}v18.0/{self.facebook_page_id}/posts"

            params = {
                'access_token': self.facebook_access_token,
                'fields': 'message,created_time,likes.summary(true),comments.summary(true),shares',
                'limit': limit
            }

            response = requests.get(url=url, params=params)
            response.raise_for_status()

            data = response.json()

            if 'data' not in data:
                print("[Facebook Official API] No data returned")
                return pd.DataFrame()

            posts_list = []
            for i, post in enumerate(data['data']):
                if i >= limit:
                    break

                # Filter berdasarkan query
                text = post.get('message', '') or ''
                if query.lower() not in text.lower():
                    continue

                # Filter tanggal
                post_date = pd.to_datetime(post.get('created_time')).date() if post.get('created_time') else datetime.now().date()
                if not (start_date.date() <= post_date <= end_date.date()):
                    continue
                
                likes = post.get('likes', {}).get('summary', {}).get('total_count', 0)
                comments = post.get('comments', {}).get('summary', {}).get('total_count', 0)
                shares = post.get('shares', {}).get('count', 0)

                posts_list.append({
                    'Date': post_date,
                    'Text': text,
                    'Engagement': likes + comments + shares,
                    'Platform': 'Facebook'
                })
                
                if (i + 1) % 50 == 0:
                    print(f"[Facebook] Scraped {i + 1} posts...")
        
        except Exception as e:
            print(f"[Facebook] Error: {e}")
            print("[Facebook] Note: Facebook scraping sangat terbatas tanpa authentication")
            return pd.DataFrame()
        
        return aggregate_daily(posts_list, 'Facebook')
    
    def _scrape_news_sentiment(self, start_date: datetime, end_date: datetime, query: str = "bitcoin", language: str = 'en', limit: int = 100, days: str = '7') -> pd.DataFrame:
        """Scrape news articles sentiment using News API."""
        if not self.newsapi_key:
            print("[News API] API key tidak tersedia")
            return pd.DataFrame()

        try:
            url = self.newsapi_key
            from_date = start_date.strftime('%Y-%m-%d')
            to_date = end_date.strftime('%Y-%m-%d')

            params = {
                'q': query,
                'from': from_date,
                'to': to_date,
                'language': language,
                'sortBy': 'publishedAt',
                'pageSize': min(limit, 100),  # News API max is 100
                'apiKey': self.newsapi_key
            }

            response = requests.get(url=url, params=params)
            response.raise_for_status()

            data = response.json()

            articles_list = []
            if data.get('status') == 'ok':
                for i, article in enumerate(data.get('articles', [])):
                    if i >= limit:
                        break

                    # Filter berdasarkan query
                    title = article.get('title', '') or ''
                    description = article.get('description', '') or ''
                    content = article.get('content', '') or ''
                    text = f"{title} {description} {content}"
                    
                    if query.lower() not in text.lower():
                        continue

                    # Filter tanggal
                    published_at = article.get('publishedAt')
                    if published_at:
                        article_date = pd.to_datetime(published_at).date()
                    else:
                        article_date = datetime.now().date()
                        
                    if not (start_date.date() <= article_date <= end_date.date()):
                        continue
                        
                    articles_list.append({
                        'Date': article_date,
                        'Text': text,
                        'Engagement': 1,  # News doesn't have engagement metrics
                        'Platform': 'News'
                    })

                    if (i + 1) % 50 == 0:
                        print(f"[News] Scraped {i + 1} articles...")
            else:
                print(f"[News API] Status: {data.get('status')}, Message: {data.get('message', 'Unknown error')}")
                return pd.DataFrame()

        except Exception as e:
            print(f"[News] Error: {e}")
            print("[News] Note: News API memerlukan API key yang valid")
            return pd.DataFrame()
        
        return aggregate_daily(articles_list, 'News')