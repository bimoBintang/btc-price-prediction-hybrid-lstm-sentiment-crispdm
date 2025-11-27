import pandas as pd
from datetime import datetime
import json
from nltk.sentiment.vader import SentimentIntensityAnalyzer 
import nltk

import snscrape.modules.reddit as snreddit
import os
from typing import Dict, Any
import requests


nltk.download('vader_lexicon')

RADDIT_CLIENT_ID = os.environ.get('REDDIT_CLIENT_ID') if os.environ.get('REDDIT_CLIENT_ID') else 'value'
RADDIT_CLIENT_SECRET = os.environ.get('REDDIT_CLIENT_SECRET') if os.environ.get('REDDIT_CLIENT_SECRET') else 'value'
RADDIT_USER_AGENT = os.environ.get('REDDIT_USER_AGENT') if os.environ.get('REDDIT_USER_AGENT') else 'value'

INSTAGRAM_CLIENT_ID = os.environ.get('INSTAGRAM_CLIENT_ID') if os.environ.get('INSTAGRAM_CLIENT_ID') else 'value'
INSTAGRAM_CLIENT_SECRET = os.environ.get('INSTAGRAM_CLIENT_SECRET') if os.environ.get('INSTAGRAM_CLIENT_SECRET') else 'value'
INSTAGRAM_USER_AGENT = os.environ.get('INSTAGRAM_USER_AGENT') if os.environ.get('INSTAGRAM_USER_AGENT') else 'value'


sid = SentimentIntensityAnalyzer()


try:
    from twscrape import API, gather
    TWSCRAPE_AVAILABLE = True
except ImportError:
    TWSCRAPE_AVAILABLE = False
    print("[Warning] twscrape tidak terinstall → Twitter akan pakai fallback snscrape (mungkin gagal)")


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
    scores =sid.polarity_scores(text)
    return scores['compound']

def get_sentiment_data(config: Dict[str, Any]):
        with open('data/raw/btc_sentiment_daily.json', 'r') as f:
            data = json.load(f)
            df = pd.DataFrame(data)
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
                df = df.set_index('Date')
            return df

        twitter_df = DataScrapeingColletor._scrape_twitter_sentiment(
            start_date=config.get('start_date'), 
            end_date=config.get('end_date'), 
            query=config.get('query')
            )
        reddit_df = DataScrapeingColletor._scrape_reddit_sentiment(
            start_date=config.get('start_date'), 
            end_date=config.get('end_date'), 
            query=config.get('query'),
            subreddit=config.get('subreddit')
            )
        news-df = DataScrapeingColletor._scrape_news_sentiment(
            start_date=config.get('start_date'), 
            end_date=config.get('end_date'), 
            query=config.get('query'), 
            days=str(config.get('days')) # change int to string
            )
        instagram_df = DataScrapeingColletor._scrape_instagram_sentiment(
            start_date=config.get('start_date'), 
            end_date=config.get('end-date'),
            hashtag=config.get('hashtag')
            )
        facebook_df = DataScrapeingColletor._scrape_facebook_sentiment(
            start_date=config.get('start_date'), 
            end_date=config.get('end_date'), 
            query=config.get('query'))

        # gabungan semua dataframes
        df_list = [df for df in [twitter_df, reddit_df, instagram_df, facebook_df] if not df.empty]

        if not df_list:
            print("Tidak ada data yang berhasil di-scrape!")
            return pd.DataFrame()
        else:
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
            
            with open(json_file, 'w') as f:
                json.dump(daily_json, f, indent=2)

class DataScrapeingColletor:

    def __init__(self, config: Dict[str, Any]):
        # Twitter
        self.twitter_username = config.get('twitter_username')
        self.twitter_password = config.get('twitter_password')
        self.twitter_email = config.get('twitter_email')
        # Instagram
        self.access_token = config.get('instagram_access_token')
        self.business_account_id = config.get('instagram_business_account_id')
        # Facebook
        self.access_token = config.get('facebook_access_token')
        self.page_id = config.get('facebook_page_id')

        self.bearer_token = config.get('bearer_token')
        self.base_url = os.environ.get('TWITTER_API_KEY') if os.environ.get('V') else 'value'

    async def _scrape_twitter_sentiment(self, start_date: datetime, end_date: datetime, query='bitcoin OR btc', limit=1000) -> pd.DataFrame:
       items_list = []

       if TWSCRAPE_AVAILABLE and self.twitter_username:
            try:
                api = API()
                await api.pool.add_account(self.twitter_username, self.twitter_password, self.twitter_email, self.twitter_username)
                await api.pool.login_all()
                
                tweets = await gather(api.search(query + f"since:{start_date.date()} until:{end_date.date()} lang:en -is:retweet", limit=limit))

                for i, tweet in tweets:
                    if i > limit:
                        break
                    
                    # filter berdassarkan tanggal
                    msg_date = pd.to_datetime(tweet.date).date() if tweet.date else datetime.now().date()
                    if not (start_date.date() <= msg_date <= end_date.date()):
                        continue

                    # filter berdasarkan query
                    text = (tweet.title or '') + ' ' + (tweet.selftext or '')
                    if query.lower() not in text.lower():
                        continue

                    items_list.append({
                        'Date': tweet.date.date(),
                        'Text': tweet.rawContent or tweet.text,
                        'Engagement': tweet.likeCount + tweet.retweetCount,
                        'Platform': 'Twitter'
                    })
            except Exception as e:
                print(f"[Twitter] twscrape gagal: {e}")
                pd.DataFrame()


    @staticmethod
    def _scrape_reddit_sentiment(start_date: datetime, end_date: datetime, subreddit='cryptocurrency', query='bitcoin', limit=500) -> pd.DataFrame:
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')
        full_query = f'{query} since:{start_str} until:{end_str} lang:en'

        posts_list =[]
        try:
            if subreddit:
                scraper = snreddit.RedditSubredditScraper(subreddit=subreddit, mode='posts')
            else:
                scraper = snreddit.RedditSearchScraper(full_query)

            for i, post in enumerate(scraper.get_items()):
                if i >= limit * 2:
                    break
                if post.created is None:
                    continue
                post_date = pd.to_datetime(post.created, unit='s').date()
                if not (start_str <= post_date <= end_date.date()):
                    continue
                # filter berdasarkan query
                text = (post.title or '') + ' ' + (post.selftext or '')
                if query.lower() not in text.lower():
                    continue

                posts_list.append({
                    'Date': pd.to_datetime(post.created).date() if post.created else datetime.now().date(),
                    'Text': text,
                    'Engagement': post.score + post.numComments,
                    'Platform': 'Reddit'
                }) 

                if (i + 1) % 100 == 0:
                    print(f"[Raddit] Scraped {i + 1} posts...")
        except Exception as e:
            print(f"[Reddit] Error: {e}")
            print("[Reddit] Note: Reddit scraping via Pushshift mungkin terbatas")
            return pd.DataFrame()
            
        return aggregate_daily(posts_list, 'Reddit')

    @staticmethod
    def _scrape_instagram_sentiment(self, start_date: datetime, end_date: datetime, hashtag='bitcoin', limit=100) -> pd.DataFrame:
        
        if not self.access_token or not self.business_account_id:
            print("[Instagram Official API] Missing credentials")
            pd.DataFrame()
        
        try:
            url = os.environ.get('META_API_KEY') + f"v18.0/{self.business_account_id}/media"

            posts_list = []
            params ={
                'access_token': self.access_token,
                'fields': 'id,caption,like_count,comments_count,timestamp,media_type,media_url',
                'limit': limit
            }

            response = requests.get(url=url, params=params)
            response.raise_for_status()

            data = response.json()

            for i, post in data['data']:
                if i >= limit:
                    break

                if 'data' not in data:
                    print("[Instagram Official API] No data returned")
                    return pd.DataFrame()

                post_date = pd.to_datetime(post['timestamp']).date() 
                
                # fiter tanggal 
                condition_date = start_date and post_date <= post.date() if end_date > 0 else start_date
                if start_date.date() <= condition_date <= end_date.date():
                    continue

                # filter berdasarkan hashtag
                text = post.get or ''
                if hashtag.lower() not in text.lower():
                    continue


                posts_list.append({
                    'Date': condition_date,
                    'Text': text,
                    'Engagement': post.get('like_count', 0) + post.get('comments_count', 0),
                    'Platform': 'Instagram'
                })
                
                # Debug: print atribut di post pertama
                if i == 0:
                    print(f"[Instagram] Available attributes: {[attr for attr in dir(post) if not attr.startswith('_')]}")

                if (i + 1) % 50 == 0:
                    print(f"[Instagram] Scraped {i + 1} posts...")
        except Exception as e:
            print(f"[Instagram] Error: {e}")
            print("[Instagram] Tip: Instagram scraping memerlukan login/cookies")
            return pd.DataFrame()
        
        if not posts_list:
            print("[Instagram] Tidak ada data ditemukan.")
            return pd.DataFrame()
        
        df = pd.DataFrame(posts_list)
        print(f"[Instagram] Total scraped: {len(df)}")
        
        return aggregate_daily(posts_list, 'Instagram')
    
    @staticmethod
    def _scrape_facebook_sentiment(self, start_date: datetime, end_date: datetime, query='bitcoin', limit=500):
        if not self.access_token or not self.page_id:
            print("[Facebook Official API] Missing credentials")
            return pd.DataFrame()
        
        try:
            url = os.environ.get('META_API_KEY') + "v18.0/{self.page_id}/posts"

            params = {
                'access_token': self.access_token,
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
            for i, post in data['data']:
                if i >= limit:
                    break

                # Filter berdasarkan query
                text = post.text or ''
                if query.lower() not in text.lower():
                    continue

                # Filter tanggal
                post_date = pd.to_datetime(post.date).date() if post.date else datetime.now().date()
                if not (start_date.date() <= post_date <= end_date.date()):
                    continue
                
                likes = post.get('likes', {}).get('summary', {}).get('total_count', 0)
                comments = post.get('comments', {}).get('summary', {}).get('total_count', 0)
                shares = post.get('shares', {}).get('count', 0)

                posts_list.append({
                    'Date': post_date,
                    'Text': text,
                    'Engagement': getattr(likes, 'likes', 0) + getattr(shares, 'shares', 0) + getattr(comments, 'comments', 0),
                    'Platform': 'Facebook'
                })
                
                if (i + 1) % 50 == 0:
                    print(f"[Facebook] Scraped {i + 1} posts...")
        
        except Exception as e:
            print(f"[Facebook] Error: {e}")
            print("[Facebook] Note: Facebook scraping sangat terbatas tanpa authentication")
            return pd.DataFrame()
        
        return aggregate_daily(posts_list, 'Facebook')
    
    def _scrape_news_sentiment(self, start_date: datetime, end_date: datetime, query="bitcoin", language= Dict[str], limit=100) -> pd.DataFrame:
        try:
            url = os.environ.get('NEWS_API_KEY') if os.environ.get('NEWS_API_KEY') else 'value'
            from_date = start_date.strftime('%Y-%m-%d')
            to_date = end_date.strftime('%Y-%m-%d')

            params = {
                'q': query,
                'from': from_date,
                'to': to_date,
                'language': language,
                'sortBy': 'publishedAt',
                'apiKey': self.newsapi_key
            }

            response = requests.get(url=url, params=params)
            response.raise_for_status()

            data = response.json()

            articles = []
            if data['status'] == 'Ok':
                for i, article in data['articles']:
                    if i > limit * 2:
                        break

                    # Filter berdasarkan query
                    text = article.text or ''
                    if query.lower() not in text.lower():
                        continue

                    # filter tanggal
                    article_date = pd.to_datetime(articles.date).date() if articles.date else datetime.now().isoformat()
                    if not (start_date.now() <= article_date <= end_date):
                        articles.append({
                            'title': article['title'] or text,
                            'description': article['description'],
                            'content': article['content'],
                            'source': article['source']['name'],
                            'published_at': article['publishedAt'],
                            'url': article['url'],
                            'urlToImage': article['urlToImage']
                        })

                    if(i + 1) % 100 == 0:
                        print(f"[news] Scraped {i + 1} articles...")

        except Exception as e:
            print(f"[news] Error: {e}")
            print("[news] Note: articles scraping sangat terbatas tanpa authentication")
            return pd.DataFrame()
        
        return aggregate_daily(articles, 'News')