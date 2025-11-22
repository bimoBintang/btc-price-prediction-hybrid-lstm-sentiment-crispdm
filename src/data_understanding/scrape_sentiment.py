import pandas as pd
from datetime import datetime
import json
from nltk.sentiment.vader import SentimentIntensityAnalyzer 
import nltk

import snscrape.modules.reddit as snreddit
import snscrape.modules.twitter as sntwitter
import snscrape.modules.facebook as snfacebook
import snscrape.modules.instagram as sninstagram
import snscrape.modules.telegram as sntelegram
import os


nltk.download('vader_lexicon')

RADDIT_CLIENT_ID = os.environ.get('REDDIT_CLIENT_ID') if os.environ.get('REDDIT_CLIENT_ID') else 'value'
RADDIT_CLIENT_SECRET = os.environ.get('REDDIT_CLIENT_SECRET') if os.environ.get('REDDIT_CLIENT_SECRET') else 'value'
RADDIT_USER_AGENT = os.environ.get('REDDIT_USER_AGENT') if os.environ.get('REDDIT_USER_AGENT') else 'value'

INSTAGRAM_CLIENT_ID = os.environ.get('INSTAGRAM_CLIENT_ID') if os.environ.get('INSTAGRAM_CLIENT_ID') else 'value'
INSTAGRAM_CLIENT_SECRET = os.environ.get('INSTAGRAM_CLIENT_SECRET') if os.environ.get('INSTAGRAM_CLIENT_SECRET') else 'value'
INSTAGRAM_USER_AGENT = os.environ.get('INSTAGRAM_USER_AGENT') if os.environ.get('INSTAGRAM_USER_AGENT') else 'value'


sid = SentimentIntensityAnalyzer()


def get_sentiment_data(start_date, end_date, query):
    json_file = 'data/raw/btc_sentiment_daily.json'
    os.makedirs(os.path.dirname(json_file), exist_ok=True)

    if os.path.exists(json_file):
        print(f"Memuat sentimen dari {json_file}")
        with open(json_file, 'r') as f:
            data = json.load(f)
            df = pd.DataFrame(data)
            if 'Date' in df.columns:
                 df.index = pd.to_datetime(df.index)
        return df

    print("Scraping sentimen dari Twitter & Reddit...")
    twitter_df = scrape_twitter_sentiment(start_date, end_date)
    reddit_df = scrape_reddit_sentiment(start_date, end_date)
    instagram_df = scrape_instagram_sentiment(start_date, end_date)
    facebook_df = scrape_facebook_sentiment(start_date, end_date, query=query)
    telegram_df = scrape_telegram_sentiment(start_date, end_date, query=query)

    # Combine all dataframes
    df_list = [df for df in [twitter_df, reddit_df, instagram_df, telegram_df, facebook_df] if not df.empty]
    
    if not df_list:
        print("Tidak ada data yang berhasil di-scrape!")
        return pd.DataFrame()

    combined = pd.concat(df_list, ignore_index=True)
    
    daily = combined.groupby ('Date').agg({
        'Net_Sentiment_Score': 'mean',
        'Discussion_Volume': 'sum',
        'Engagement': 'sum'
    }).reset_index()

    # Simpan ke JSON dengan format yang proper
    daily_json = daily.to_dict(orient='records')
    for record in daily_json:
        record['Date'] = record['Date'].strftime('%Y-%m-%d')
    
    with open(json_file, 'w') as f:
        json.dump(daily_json, f, indent=2)

    return daily

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

def scrape_twitter_sentiment(start_date, end_date, query='bitcoin OR btc', max_items=1000):
    start_str = start_date.strftime('%Y-%m-%d')
    end_str = end_date.strftime('%Y-%m-%d')
    full_query = f'{query} since:{start_str} until:{end_str} lang:en'
    
    print(f"[Twitter] Query: {full_query}")
    
    items_list = []
    try:
        for i, tweet in enumerate(sntwitter.TwitterSearchScraper(full_query).get_items()):
            if i >= max_items:
                break
            items_list.append({
                'Date': pd.to_datetime(tweet.date).date(),
                'Text': tweet.rawContent,
                'Engagement': tweet.likeCount + tweet.retweetCount,
                'Platform': 'Twitter'
            })
            
            if (i + 1) % 100 == 0:
                print(f"[Twitter] Scraped {i + 1} tweets...")
    
    except Exception as e:
        print(f"[Twitter] Error: {e}")
        return pd.DataFrame()
    
    return aggregate_daily(items_list, 'Twitter')


def scrape_reddit_sentiment(start_date, end_date, subreddit_name='cryptocurrency', query='bitcoin', limit=500):
    start_str = start_date.strftime('%Y-%m-%d')
    end_str = end_date.strftime('%Y-%m-%d')
    full_query = f'{query} since:{start_str} until:{end_str} lang:en'

    posts_list =[]
    try:
        if subreddit_name:
            scraper = snreddit.RedditSubredditScraper(subreddit=subreddit_name)
        else:
            scraper = snreddit.RedditSearchScraper(full_query)

        for i, post in enumerate(scraper.get_items()):
            if i >= limit:
                break

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
            if (i + 1) % 50 == 0:
                print(f"[Raddit] Scraped {i + 1} posts...")
    except Exception as e:
        print(f"[Reddit] Error: {e}")
        print("[Reddit] Note: Reddit scraping via Pushshift mungkin terbatas")
        return pd.DataFrame()
        
    return aggregate_daily(posts_list, 'Reddit')


def scrape_instagram_sentiment(start_date, end_date, hastag_name = None, username = None, location = None, limit=500):
    start_str = start_date.strftime('%Y-%m-%d')
    end_str = end_date.strftime('%Y-%m-%d')
    full_query = f'{hastag_name} since:{start_str} until:{end_str} lang:en'
    
    posts_list = []
    try:
        scraper = sninstagram.InstagramHashtagScraper(hashtag=hastag_name) or \
            sninstagram.InstagramHashtagScraper(username=username) or \
            sninstagram.InstagramLocationScraper(locationId=location) or \
            sninstagram.InstagramPost.content(full_query)

        for i, post in enumerate(scraper.get_items()):
            if i >= limit:
                break

            # Cek atribut yang tersedia
            text = ''
            if hasattr(post, 'content'):
                text = post.content or ''
            elif hasattr(post, 'caption'):
                text = post.caption or ''
            elif hasattr(post, 'description'):
                text = post.description or ''

            # Ambil engagement metrics
            likes = getattr(post, 'likes', 0) or getattr(post, 'likesCount', 0) or getattr(post, 'likeCount', 0) or 0
            comments = getattr(post, 'comments', 0) or getattr(post, 'commentsCount', 0) or getattr(post, 'commentCount', 0) or 0

            posts_list.append({
                'Date': pd.to_datetime(post.date).date() if hasattr(post, 'date') and post.date else datetime.now().date(),
                'Text': text,
                'Engagement': likes + comments,
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


def scrape_telegram_sentiment(start_date, end_date, query='bitcoin', channel_name=None, limit=500):
    start_str = start_date.strftime('%Y-%m-%d')
    end_str = end_date.strftime('%Y-%m-%d')
    full_query = f'{query} since:{start_str} until:{end_str} lang:en'
    
    if not channel_name:
        print("[Telegram] Error: Must provide channel_name")
        return pd.DataFrame()
    
    channels_list = []

    try:
        scraper = sntelegram.TelegramChannelScraper(channel_name) or \
            sntelegram.TelegramPost.content(full_query) or \
            sninstagram.InstagramPost.content(full_query)

        for i, msg in enumerate(scraper.get_items()):
            if i >= limit:
                break

            # Filter berdasarkan query
            text = msg.content or ''
            if query.lower() not in text.lower():
                continue

            # Filter tanggal
            msg_date = pd.to_datetime(msg.date).date() if msg.date else datetime.now().date()
            if not (start_date.date() <= msg_date <= end_date.date()):
                continue

            channels_list.append({
                'Date': msg_date,
                'Text': text,
                'Engagement': getattr(msg, 'views', 0),
                'Platform': 'Telegram'
            })

            if (i + 1) % 100 == 0:
                print(f"[Telegram] Scraped {i + 1} messages...")
    
    except Exception as e:
        print(f"[Telegram] Error: {e}")
        return pd.DataFrame()
    
    return aggregate_daily(channels_list, 'Telegram')


def scrape_facebook_sentiment(start_date, end_date, query='bitcoin', username=None, group=None, posts=None, community=None, limit=500):
    start_str = start_date.strftime('%Y-%m-%d')
    end_str = end_date.strftime('%Y-%m-%d')
    full_query = f'{query} since:{start_str} until:{end_str} lang:en'
    
    posts_list = []
    try:
        if group:
            scraper = snfacebook.FacebookGroupScraper(group=group)
        elif username:
            scraper = snfacebook.FacebookUserScraper(username)
        elif community:
            scraper = snfacebook.FacebookCommunityScraper(community)
        else:
            # Default: gunakan search
            scraper = snfacebook.FacebookSearchScraper(query)

        for i, post in enumerate(scraper.get_items()):
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

            posts_list.append({
                'Date': post_date,
                'Text': text,
                'Engagement': getattr(post, 'likes', 0) + getattr(post, 'shares', 0),
                'Platform': 'Facebook'
            })
            
            if (i + 1) % 50 == 0:
                print(f"[Facebook] Scraped {i + 1} posts...")
    
    except Exception as e:
        print(f"[Facebook] Error: {e}")
        print("[Facebook] Note: Facebook scraping sangat terbatas tanpa authentication")
        return pd.DataFrame()
    
    return aggregate_daily(posts_list, 'Facebook')
