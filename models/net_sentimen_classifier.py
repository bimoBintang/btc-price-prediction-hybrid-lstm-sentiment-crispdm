import pandas as pd
from datetime import datetime, timedelta
import json
import os
from nltk.sentiment.vader import SentimentIntensityAnalyzer  # Lebih baik untuk sosial media
import nltk
import snscrape.modules.twitter as sntwitter
import praw
from dotenv import load_dotenv

nltk.download('vader_lexicon')



load_dotenv()

REDDIT_CLIENT_ID = os.environ.get('REDDIT_CLIENT_ID')     # Dari https://www.reddit.com/prefs/apps
REDDIT_CLIENT_SECRET = os.environ.get('REDDIT_CLIENT_SECRET')  
REDDIT_USER_AGENT = os.environ.get('REDDIT_USER_AGENT')  

def main():
    analyze_sentiment()
    scrape_twitter_sentiment()
    scrape_raddit_sentiment()
    get_combined_sentiment()

# --- FUNGSI UNTUK SENTIMENT ANALYSIS ---
sid = SentimentIntensityAnalyzer()

def analyze_sentiment(text):
    secores =sid.polarity_scores(text)
    return secores['compound']


def scrape_twitter_sentiment(start_date, end_date, query='bitcoin OR btc OR BTC', max_tweet=1000):

    start_str = start_date.strftime('%Y-%m-%d')
    end_str = end_date.strftime('%Y-%m-%d')
    full_query = f'{query} since:{start_str} until:{end_str} lang:en'

    tweets_list = []
    for i, tweet in enumerate(sntwitter.TwitterSearchScraper(full_query).get_items()):
        if i >= max_tweet:
            break
        tweets_list.append({
            'Date': pd.to_datetime(tweet.date).date(),
            'Text': tweet.rawContent,
            'Likes': tweet.likeCount,
            'Retweets': tweet.retweetCount
        })

    twitter_df = pd.DataFrame(tweets_list)
    if twitter_df.empty:
        print("Tidak ada tweet ditemukan. Coba query lain atau periksa date range.")
        return pd.DataFrame()
    
    # Agregasi harian
    twitter_daily = twitter_df.groupby('Date').agg({
        'Text': lambda x: ' '.join(x),  # Gabung teks untuk analisis
        'Likes': 'sum',
        'Retweets': 'sum'
    }).reset_index()
    
    twitter_daily['Net_Sentiment_Score'] = twitter_daily['Text'].apply(analyze_sentiment)
    twitter_daily['Discussion_Volume'] = twitter_df.groupby('Date').size().values  # Jumlah tweet/hari
    
    return twitter_daily[['Date', 'Net_Sentiment_Score', 'Discussion_Volume']]


def scrape_raddit_sentiment(subRaddit_name='bitcon', start_date=None, end_date=None, limit=500):
    
    raddit = praw.Reddit(
        client_id=REDDIT_CLIENT_ID,
        client_secret=REDDIT_CLIENT_SECRET,
        user_agent=REDDIT_USER_AGENT
    )

    subRaddit = raddit.subreddit(subRaddit_name)
    posts_list = []

    for post in subRaddit.top(limit=limit, time_filter='month'):
        posts_list.append({
            'Date': pd.to_datetime(post.created_utc, unit='s').date(),
            'Text': f"{post.title} {post.selftext}",  # Gabung title + body
            'Score': post.score,
            'Comments': post.num_comments
        })

    reddit_df = pd.DataFrame(posts_list)
    if reddit_df.empty:
        print("Tidak ada post Reddit ditemukan.")
        return pd.DataFrame()
    
    # Filter date range jika diberikan
    if start_date and end_date:
        mask = (reddit_df['Date'] >= start_date.date()) & (reddit_df['Date'] <= end_date.date())
        reddit_df = reddit_df[mask]
    
    # Agregasi harian
    reddit_daily = reddit_df.groupby('Date').agg({
        'Text': lambda x: ' '.join(x),
        'Score': 'sum',
        'Comments': 'sum'
    }).reset_index()
    
    reddit_daily['Net_Sentiment_Score'] = reddit_daily['Text'].apply(analyze_sentiment)
    reddit_daily['Discussion_Volume'] = reddit_df.groupby('Date').size().values  # Jumlah post/hari
    
    return reddit_daily[['Date', 'Net_Sentiment_Score', 'Discussion_Volume']]


def scrape_instagram_sentiment(hastag_name, start_date, end_date, limit=500):
    instagram = 

def get_combined_sentiment(start_date, end_date):
    # Gabung sentimen dari Twitter & Reddit, rata-rata harian
    twitter_send = scrape_twitter_sentiment(start_date=start_date, end_date=end_date)
    raddit_send = scrape_raddit_sentiment(start_date=start_date, end_date=end_date)

    # gabungan berdasarkan date
    combined = pd.concat([twitter_send, raddit_send], ignore_index=True)
    combined_daily = combined.groupby('Date').agg({
        'Net_Sentiment_Score': 'mean', # nilai rata rata
        'Discussion_Volume': 'sum'
    }).reset_index()

    # Konversi Date ke pd.DatetimeIndex untuk join dengan BTC data
    combined_daily['Date'] = pd.to_datetime(combined_daily['Date'])
    combined_daily = combined_daily.set_index('Date')
    
    return combined_daily


if __name__ == "__main__":
    # Date range: 30 hari terakhir
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    print("=== SCRAPING SENTIMEN HARIAN (Twitter + Reddit) ===")
    sentiment_df = get_combined_sentiment(start_date, end_date)
    
    if not sentiment_df.empty:
        print("\nHasil Sentimen Harian:")
        print(sentiment_df.head(10))  # Tampilkan 10 hari pertama
        
        # Simpan ke JSON (seperti YFinance)
        json_file = 'btc_sentiment_daily.json'
        sentiment_dict = sentiment_df.to_dict()
        with open(json_file, 'w') as f:
            json.dump(sentiment_dict, f, indent=2, default=str)
        print(f"\nData sentimen disimpan ke: {json_file}")
        
        # Statistik
        print(f"\nStatistik Keseluruhan:")
        print(f"- Rata-rata Net Sentiment: {sentiment_df['Net_Sentiment_Score'].mean():.3f}")
        print(f"- Total Volume Diskusi: {sentiment_df['Discussion_Volume'].sum():,}")
    else:
        print("Gagal scraping. Periksa credentials/date/query.")