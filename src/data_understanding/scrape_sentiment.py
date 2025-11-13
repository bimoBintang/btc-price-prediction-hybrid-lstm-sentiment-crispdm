import pandas as pd
from datetime import datetime, timedelta
import json
from nltk.sentiment.vader import SentimentIntensityAnalyzer 
import nltk

import snscrape.modules.reddit as snreddit
import snscrape.modules.twitter as sntwitter
import snscrape.modules.facebook as snfacebook
import snscrape.modules.instagram as sninstagram
import snscrape.modules.telegram as sntelegram
import snscrape.modules.mastodon as snmastodon
import praw
import os


nltk.download('vader_lexicon')

RADDIT_CLIENT_ID = os.environ.get('RADDIT_CLIENT_ID')
RADDIT_CLIENT_SECRET = os.environ.get('RADDIT_CLIENT_SECRET')
RADDIT_USER_AGENT = os.environ.get('RADDIT_USER_AGENT')

INSTAGRAM_CLIENT_ID = os.environ.get('INSTAGRAM_CLIENT_ID')
INSTAGRAM_CLIENT_SECRET = os.environ.get('INSTAGRAM_CLIENT_SECRET')
INSTAGRAM_USER_AGENT = os.environ.get('INSTAGRAM_USER_AGENT')


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
    secores =sid.polarity_scores(text)
    return secores['compound']

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


def scrape_raddit_sentiment(sub_raddit_name, query='bitcoin', limit=500):
    
    posts_list =[]
    try:
        scraper = snreddit.RedditSubredditScraper(name=sub_raddit_name)
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


def scrape_instagram_sentiment(hastag_name, username, location, limit=500):
    posts_list = []

    try:
        scraper = sninstagram.InstagramHashtagScraper(hashtag=hastag_name) | sninstagram.InstagramHashtagScraper(username=username) | sninstagram.InstagramLocationScraper(locationId=location)

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


def scrape_telegram_sentiment(channel_name, limit=500):
    channels_list = []
    try:
        scraper = sntelegram.TelegramChannelScraper(channel_name)

        for i, msg in enumerate(scraper.get_items()):
            if i >= limit:
                break

            channels_list.append({
                'Date': pd.to_datetime(msg.date).date() if msg.date else datetime.now().date(),
                'Text': msg.content or '',
                'Engagement': getattr(msg, 'views', 0),
                'Platform': 'Telegram'
            })

            if (i + 1) % 100 == 0:
                print(f"[Telegram] Scraped {i + 1} messages...")
    
    except Exception as e:
        print(f"[Telegram] Error: {e}")
        return pd.DataFrame()
    
    return aggregate_daily(channels_list, 'Telegram')


def scrape_facebook_sentiment(usernname, group, posts, community, limit=500):
    posts_list = []
    try:
        scraper = snfacebook.FacebookPost(posts) | snfacebook.FacebookGroupScraper(group=group) | snfacebook.FacebookUserScraper(usernname) | snfacebook.FacebookCommunityScraper(community)

        for i, post in enumerate(scraper.get_items()):
            if i >= limit:
                break

            posts_list.append({
                'Date': pd.to_datetime(post.date).date() if post.date else datetime.now().date(),
                'Text': post.text or '',
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
