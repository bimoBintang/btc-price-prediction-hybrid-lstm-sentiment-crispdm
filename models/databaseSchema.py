from sqlalchemy import (
    create_engine, Column, Integer, String, Text, DECIMAL, 
    Boolean, TIMESTAMP, Date, BigInteger, ForeignKey, 
    CheckConstraint, UniqueConstraint, Index, ARRAY, func
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker
from sqlalchemy.dialects.postgresql import UUID
from datetime import datetime
import os

Base = declarative_base()


class DatabaseManager:
    def __init__(self, connection_string=None):
        if connection_string is None:
            # Gunakan environment variables jika tidak ada connection string
            connection_string = os.getenv('DATABASE_URL')
        
        self.engine = create_engine(
            connection_string,
            echo=False,  # Set True untuk debug SQL queries
            pool_size=10,
            max_overflow=20,
            pool_pre_ping=True  # Verify connections before using
        )
        
        self.SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=self.engine
        )
    
    def create_all_tables(self):
        """Create all tables in database"""
        Base.metadata.create_all(bind=self.engine)
        print("✅ All tables created successfully!")
    
    def drop_all_tables(self):
        """Drop all tables (use with caution!)"""
        Base.metadata.drop_all(bind=self.engine)
        print("⚠️  All tables dropped!")
    
    def get_session(self):
        """Get database session"""
        return self.SessionLocal()
    
    def seed_initial_data(self):
        """Insert initial data into database"""
        session = self.get_session()
        
        try:
            # Check if data already exists
            if session.query(DataSource).count() > 0:
                print("⚠️  Initial data already exists, skipping seed...")
                return
            
            # Insert Data Sources
            sources = [
                DataSource(source_name='yfinance', source_url='https://finance.yahoo.com', rate_limit_per_minute=2000),
                DataSource(source_name='coinbase', source_url='https://api.coinbase.com', rate_limit_per_minute=10),
                DataSource(source_name='coingecko', source_url='https://api.coingecko.com', rate_limit_per_minute=50)
            ]
            session.add_all(sources)
            
            # Insert Social Media Platforms
            platforms = [
                SocialMediaPlatform(platform_name='Twitter', platform_url='https://twitter.com'),
                SocialMediaPlatform(platform_name='Reddit', platform_url='https://reddit.com'),
                SocialMediaPlatform(platform_name='Telegram', platform_url='https://telegram.org')
            ]
            session.add_all(platforms)
            
            # Insert Cryptocurrencies
            cryptos = [
                Cryptocurrency(symbol='BTC', name='Bitcoin', coingecko_id='bitcoin', market_cap_rank=1),
                Cryptocurrency(symbol='ETH', name='Ethereum', coingecko_id='ethereum', market_cap_rank=2),
                Cryptocurrency(symbol='BNB', name='Binance Coin', coingecko_id='binancecoin', market_cap_rank=3)
            ]
            session.add_all(cryptos)
            
            session.commit()
            print("✅ Initial data seeded successfully!")
            
        except Exception as e:
            session.rollback()
            print(f"❌ Error seeding data: {e}")
        finally:
            session.close()


class DataSource(Base):
    __tablename__ = 'data_source'

    source_id = Column(Integer, primary_key=True, autoincrement=True)
    source_name = Column(String(50), nullable=False, unique=True)
    source_url = Column(Text)
    api_version = Column(String(20))
    is_active = Column(Boolean, default=True)
    rate_limit_per_minute = Column(Integer)
    created_at = Column(TIMESTAMP, default=datetime.utcnow)
    updated_at = Column(TIMESTAMP, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    crypto_prices = relationship("CryptoPrice", back_populates="source")
    
    def __repr__(self):
        return f"<DataSource(name='{self.source_name}', active={self.is_active})>"


class Cryptocurrency(Base):
    __tablename__ = 'cryptocurrencies'
    
    crypto_id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(10), nullable=False, unique=True)
    name = Column(String(100), nullable=False)
    coingecko_id = Column(String(50))
    coinbase_id = Column(String(50))
    market_cap_rank = Column(Integer)
    is_active = Column(Boolean, default=True)
    created_at = Column(TIMESTAMP, default=datetime.utcnow)
    updated_at = Column(TIMESTAMP, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    crypto_prices = relationship("CryptoPrice", back_populates="cryptocurrency")
    technical_indicators = relationship("TechnicalIndicator", back_populates="cryptocurrency")
    sentiment_aggregations = relationship("DailySentimentAggregation", back_populates="cryptocurrency")
    training_data = relationship("ModelTrainingData", back_populates="cryptocurrency")
    predictions = relationship("ModelPrediction", back_populates="cryptocurrency")
    
    def __repr__(self):
        return f"<Cryptocurrency(symbol='{self.symbol}', name='{self.name}')>"
    

class CryptoPrice(Base):
    __tablename__ = 'crypto_prices'
    
    price_id = Column(BigInteger, primary_key=True, autoincrement=True)
    crypto_id = Column(Integer, ForeignKey('cryptocurrencies.crypto_id'), nullable=False)
    source_id = Column(Integer, ForeignKey('data_sources.source_id'), nullable=False)
    
    # OHLCV Data
    timestamp = Column(TIMESTAMP, nullable=False)
    date_only = Column(Date, nullable=False)
    open_price = Column(DECIMAL(20, 8), nullable=False)
    high_price = Column(DECIMAL(20, 8), nullable=False)
    low_price = Column(DECIMAL(20, 8), nullable=False)
    close_price = Column(DECIMAL(20, 8), nullable=False)
    volume = Column(DECIMAL(30, 8), nullable=False)
    
    # Additional Market Data
    market_cap = Column(DECIMAL(30, 2))
    circulating_supply = Column(DECIMAL(30, 8))
    
    # Metadata
    timeframe = Column(String(10), default='1d')
    currency = Column(String(10), default='USD')
    created_at = Column(TIMESTAMP, default=datetime.utcnow)
    
    # Relationships
    cryptocurrency = relationship("Cryptocurrency", back_populates="crypto_prices")
    source = relationship("DataSource", back_populates="crypto_prices")
    
    # Constraints
    __table_args__ = (
        UniqueConstraint('crypto_id', 'source_id', 'timestamp', 'timeframe', 
                        name='uq_crypto_price_record'),
        CheckConstraint('high_price >= low_price', name='check_high_low'),
        CheckConstraint('high_price >= open_price', name='check_high_open'),
        CheckConstraint('high_price >= close_price', name='check_high_close'),
        CheckConstraint('low_price <= open_price', name='check_low_open'),
        CheckConstraint('low_price <= close_price', name='check_low_close'),
        Index('idx_crypto_prices_crypto_date', 'crypto_id', 'date_only'),
        Index('idx_crypto_prices_timestamp', 'timestamp'),
        Index('idx_crypto_prices_source', 'source_id'),
    )
    
    def __repr__(self):
        return f"<CryptoPrice(crypto_id={self.crypto_id}, date={self.date_only}, close={self.close_price})>"
    

class TechnicalIndicator(Base):
    __tablename__ = 'technical_indicators'
    
    indicator_id = Column(BigInteger, primary_key=True, autoincrement=True)
    crypto_id = Column(Integer, ForeignKey('cryptocurrencies.crypto_id'), nullable=False)
    date_only = Column(Date, nullable=False)
    
    # Moving Averages
    sma_7 = Column(DECIMAL(20, 8))
    sma_20 = Column(DECIMAL(20, 8))
    sma_50 = Column(DECIMAL(20, 8))
    sma_200 = Column(DECIMAL(20, 8))
    ema_12 = Column(DECIMAL(20, 8))
    ema_26 = Column(DECIMAL(20, 8))
    
    # Momentum Indicators
    rsi_14 = Column(DECIMAL(10, 4))
    macd = Column(DECIMAL(20, 8))
    macd_signal = Column(DECIMAL(20, 8))
    macd_histogram = Column(DECIMAL(20, 8))
    
    # Volatility Indicators
    bollinger_upper = Column(DECIMAL(20, 8))
    bollinger_middle = Column(DECIMAL(20, 8))
    bollinger_lower = Column(DECIMAL(20, 8))
    atr_14 = Column(DECIMAL(20, 8))
    
    calculated_at = Column(TIMESTAMP, default=datetime.utcnow)
    
    # Relationships
    cryptocurrency = relationship("Cryptocurrency", back_populates="technical_indicators")
    
    __table_args__ = (
        UniqueConstraint('crypto_id', 'date_only', name='uq_tech_indicator'),
        Index('idx_tech_indicators_crypto_date', 'crypto_id', 'date_only'),
    )
    
    def __repr__(self):
        return f"<TechnicalIndicator(crypto_id={self.crypto_id}, date={self.date_only}, rsi={self.rsi_14})>"


class SocialMediaPlatform(Base):
    __tablename__ = 'social_media_platforms'
    
    platform_id = Column(Integer, primary_key=True, autoincrement=True)
    platform_name = Column(String(50), nullable=False, unique=True)
    platform_url = Column(Text)
    api_endpoint = Column(Text)
    is_active = Column(Boolean, default=True)
    created_at = Column(TIMESTAMP, default=datetime.utcnow)
    
    # Relationships
    posts = relationship("SocialMediaPost", back_populates="platform")
    sentiment_aggregations = relationship("DailySentimentAggregation", back_populates="platform")
    
    def __repr__(self):
        return f"<SocialMediaPlatform(name='{self.platform_name}')>"
    

class SocialMediaPost(Base):
    __tablename__ = 'social_media_posts'
    
    post_id = Column(BigInteger, primary_key=True, autoincrement=True)
    platform_id = Column(Integer, ForeignKey('social_media_platforms.platform_id'), nullable=False)
    crypto_id = Column(Integer, ForeignKey('cryptocurrencies.crypto_id'))
    
    # Post Identifiers
    external_post_id = Column(String(255), nullable=False)
    author_id = Column(String(255))
    author_username = Column(String(255))
    
    # Content
    post_text = Column(Text, nullable=False)
    post_language = Column(String(10), default='en')
    
    # Metadata
    posted_at = Column(TIMESTAMP, nullable=False)
    collected_at = Column(TIMESTAMP, default=datetime.utcnow)
    
    # Engagement Metrics
    likes_count = Column(Integer, default=0)
    retweets_count = Column(Integer, default=0)
    replies_count = Column(Integer, default=0)
    views_count = Column(Integer, default=0)
    
    # Post Context
    is_retweet = Column(Boolean, default=False)
    is_reply = Column(Boolean, default=False)
    parent_post_id = Column(String(255))
    hashtags = Column(ARRAY(Text))
    mentioned_users = Column(ARRAY(Text))
    urls = Column(ARRAY(Text))
    
    # Processing Status
    is_processed = Column(Boolean, default=False)
    is_spam = Column(Boolean, default=False)
    is_bot = Column(Boolean, default=False)
    
    # Relationships
    platform = relationship("SocialMediaPlatform", back_populates="posts")
    sentiment = relationship("SentimentAnalysis", back_populates="post", uselist=False)
    
    __table_args__ = (
        UniqueConstraint('platform_id', 'external_post_id', name='uq_social_post'),
        Index('idx_social_posts_platform', 'platform_id'),
        Index('idx_social_posts_crypto', 'crypto_id'),
        Index('idx_social_posts_posted_at', 'posted_at'),
        Index('idx_social_posts_processed', 'is_processed'),
    )
    
    def __repr__(self):
        return f"<SocialMediaPost(id={self.post_id}, platform_id={self.platform_id}, posted_at={self.posted_at})>"


class SentimentAnalysis(Base):
    __tablename__ = 'sentiment_analysis'
    
    sentiment_id = Column(BigInteger, primary_key=True, autoincrement=True)
    post_id = Column(BigInteger, ForeignKey('social_media_posts.post_id', ondelete='CASCADE'), nullable=False)
    
    # Sentiment Classification
    sentiment_label = Column(String(20), nullable=False)  # 'positive', 'negative', 'neutral'
    sentiment_score = Column(DECIMAL(5, 4), nullable=False)  # -1.0 to 1.0
    confidence_score = Column(DECIMAL(5, 4))  # 0.0 to 1.0
    
    # Detailed Sentiment Scores
    positive_score = Column(DECIMAL(5, 4))
    negative_score = Column(DECIMAL(5, 4))
    neutral_score = Column(DECIMAL(5, 4))
    
    # Emotion Detection
    emotion_label = Column(String(20))  # 'fear', 'greed', 'joy', 'anger'
    emotion_score = Column(DECIMAL(5, 4))
    
    # Model Information
    model_name = Column(String(100))
    model_version = Column(String(50))
    
    analyzed_at = Column(TIMESTAMP, default=datetime.utcnow)
    
    # Relationships
    post = relationship("SocialMediaPost", back_populates="sentiment")
    
    __table_args__ = (
        CheckConstraint('sentiment_score >= -1.0 AND sentiment_score <= 1.0', name='check_sentiment_range'),
        CheckConstraint('confidence_score >= 0.0 AND confidence_score <= 1.0', name='check_confidence_range'),
        Index('idx_sentiment_post', 'post_id'),
        Index('idx_sentiment_label', 'sentiment_label'),
        Index('idx_sentiment_score', 'sentiment_score'),
    )
    
    def __repr__(self):
        return f"<SentimentAnalysis(post_id={self.post_id}, label='{self.sentiment_label}', score={self.sentiment_score})>"


class DailySentimentAggregation(Base):
    __tablename__ = 'daily_sentiment_aggregation'
    
    agg_id = Column(Integer, primary_key=True, autoincrement=True)
    crypto_id = Column(Integer, ForeignKey('cryptocurrencies.crypto_id'), nullable=False)
    platform_id = Column(Integer, ForeignKey('social_media_platforms.platform_id'))
    date_only = Column(Date, nullable=False)
    
    # Volume Metrics
    total_posts = Column(Integer, nullable=False, default=0)
    unique_authors = Column(Integer, default=0)
    
    # Sentiment Distribution
    positive_count = Column(Integer, default=0)
    negative_count = Column(Integer, default=0)
    neutral_count = Column(Integer, default=0)
    
    # Aggregated Scores
    avg_sentiment_score = Column(DECIMAL(5, 4))
    weighted_sentiment_score = Column(DECIMAL(5, 4))
    net_sentiment_score = Column(DECIMAL(5, 4))
    
    # Sentiment Percentages
    positive_percentage = Column(DECIMAL(5, 2))
    negative_percentage = Column(DECIMAL(5, 2))
    neutral_percentage = Column(DECIMAL(5, 2))
    
    # Engagement Metrics
    total_likes = Column(BigInteger, default=0)
    total_retweets = Column(BigInteger, default=0)
    total_replies = Column(BigInteger, default=0)
    avg_engagement = Column(DECIMAL(10, 2))
    
    # Emotion Aggregation
    fear_index = Column(DECIMAL(5, 4))
    greed_index = Column(DECIMAL(5, 4))
    
    calculated_at = Column(TIMESTAMP, default=datetime.utcnow)
    
    # Relationships
    cryptocurrency = relationship("Cryptocurrency", back_populates="sentiment_aggregations")
    platform = relationship("SocialMediaPlatform", back_populates="sentiment_aggregations")
    
    __table_args__ = (
        UniqueConstraint('crypto_id', 'platform_id', 'date_only', name='uq_daily_sentiment'),
        Index('idx_daily_sentiment_crypto_date', 'crypto_id', 'date_only'),
        Index('idx_daily_sentiment_platform', 'platform_id'),
    )
    
    def __repr__(self):
        return f"<DailySentimentAggregation(crypto_id={self.crypto_id}, date={self.date_only}, net_score={self.net_sentiment_score})>"


class ModelTrainingData(Base):
    __tablename__ = 'model_training_data'
    
    training_id = Column(BigInteger, primary_key=True, autoincrement=True)
    crypto_id = Column(Integer, ForeignKey('cryptocurrencies.crypto_id'), nullable=False)
    date_only = Column(Date, nullable=False)
    
    # Price Features
    open_price = Column(DECIMAL(20, 8))
    high_price = Column(DECIMAL(20, 8))
    low_price = Column(DECIMAL(20, 8))
    close_price = Column(DECIMAL(20, 8))
    volume = Column(DECIMAL(30, 8))
    
    # Lagged Price Features
    close_price_lag1 = Column(DECIMAL(20, 8))
    close_price_lag3 = Column(DECIMAL(20, 8))
    close_price_lag7 = Column(DECIMAL(20, 8))
    volume_lag1 = Column(DECIMAL(30, 8))
    
    # Technical Indicators
    rsi_14 = Column(DECIMAL(10, 4))
    macd = Column(DECIMAL(20, 8))
    sma_20 = Column(DECIMAL(20, 8))
    
    # Sentiment Features
    net_sentiment_score = Column(DECIMAL(5, 4))
    avg_sentiment_score = Column(DECIMAL(5, 4))
    total_discussion_volume = Column(Integer)
    positive_percentage = Column(DECIMAL(5, 2))
    fear_index = Column(DECIMAL(5, 4))
    
    # Target Variable
    next_day_close = Column(DECIMAL(20, 8))
    price_change_percentage = Column(DECIMAL(10, 4))
    
    # Dataset Split
    dataset_split = Column(String(20))  # 'train', 'validation', 'test'
    
    is_normalized = Column(Boolean, default=False)
    created_at = Column(TIMESTAMP, default=datetime.utcnow)
    
    # Relationships
    cryptocurrency = relationship("Cryptocurrency", back_populates="training_data")
    
    __table_args__ = (
        UniqueConstraint('crypto_id', 'date_only', name='uq_training_data'),
        Index('idx_training_data_crypto_date', 'crypto_id', 'date_only'),
        Index('idx_training_data_split', 'dataset_split'),
    )
    
    def __repr__(self):
        return f"<ModelTrainingData(crypto_id={self.crypto_id}, date={self.date_only}, split='{self.dataset_split}')>"


class ModelPrediction(Base):
    __tablename__ = 'model_predictions'
    
    prediction_id = Column(BigInteger, primary_key=True, autoincrement=True)
    model_name = Column(String(100), nullable=False)
    model_version = Column(String(50))
    crypto_id = Column(Integer, ForeignKey('cryptocurrencies.crypto_id'), nullable=False)
    prediction_date = Column(Date, nullable=False)
    
    # Predictions
    predicted_close_price = Column(DECIMAL(20, 8), nullable=False)
    actual_close_price = Column(DECIMAL(20, 8))
    prediction_error = Column(DECIMAL(20, 8))
    absolute_error = Column(DECIMAL(20, 8))
    percentage_error = Column(DECIMAL(10, 4))
    
    # Confidence
    confidence_interval_lower = Column(DECIMAL(20, 8))
    confidence_interval_upper = Column(DECIMAL(20, 8))
    prediction_confidence = Column(DECIMAL(5, 4))
    
    created_at = Column(TIMESTAMP, default=datetime.utcnow)
    
    # Relationships
    cryptocurrency = relationship("Cryptocurrency", back_populates="predictions")
    
    __table_args__ = (
        Index('idx_predictions_model_date', 'model_name', 'prediction_date'),
        Index('idx_predictions_crypto', 'crypto_id'),
    )
    
    def __repr__(self):
        return f"<ModelPrediction(model='{self.model_name}', crypto_id={self.crypto_id}, predicted={self.predicted_close_price})>"


class ModelPerformanceMetric(Base):
    __tablename__ = 'model_performance_metrics'
    
    metric_id = Column(Integer, primary_key=True, autoincrement=True)
    model_name = Column(String(100), nullable=False)
    model_version = Column(String(50))
    crypto_id = Column(Integer, ForeignKey('cryptocurrencies.crypto_id'))
    
    evaluation_date = Column(Date, nullable=False)
    dataset_type = Column(String(20))  # 'train', 'validation', 'test'
    
    # Error Metrics
    mae = Column(DECIMAL(20, 8))
    rmse = Column(DECIMAL(20, 8))
    mape = Column(DECIMAL(10, 4))
    
    # Additional Metrics
    r2_score = Column(DECIMAL(10, 6))
    total_predictions = Column(Integer)
    
    created_at = Column(TIMESTAMP, default=datetime.utcnow)
    
    def __repr__(self):
        return f"<ModelPerformanceMetric(model='{self.model_name}', mae={self.mae}, rmse={self.rmse})>"


class DataQualityLog(Base):
    __tablename__ = 'data_quality_logs'
    
    log_id = Column(BigInteger, primary_key=True, autoincrement=True)
    table_name = Column(String(100), nullable=False)
    check_type = Column(String(50), nullable=False)
    
    records_checked = Column(Integer)
    issues_found = Column(Integer)
    severity = Column(String(20))  # 'low', 'medium', 'high', 'critical'
    
    issue_description = Column(Text)
    action_taken = Column(Text)
    
    checked_at = Column(TIMESTAMP, default=datetime.utcnow)
    
    __table_args__ = (
        Index('idx_quality_logs_table', 'table_name'),
        Index('idx_quality_logs_severity', 'severity'),
    )
    
    def __repr__(self):
        return f"<DataQualityLog(table='{self.table_name}', severity='{self.severity}')>"
    

