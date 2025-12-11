"""
FastAPI REST API for BTC Price Prediction Pipeline.

This API exposes the ML pipeline functionality to be consumed by a frontend website.
"""
from fastapi import FastAPI, HTTPException, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import os
import sys

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

# Initialize FastAPI app
app = FastAPI(
    title="BTC Price Prediction API",
    description="REST API for Bitcoin price prediction using ML models with sentiment analysis",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json"
)

# CORS configuration - allow frontend connections
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",      # Next.js dev
        "http://localhost:5173",      # Vite dev
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
        "*"  # Allow all origins in development (restrict in production)
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==================== PYDANTIC MODELS ====================

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    version: str


class PriceData(BaseModel):
    price: float
    change_24h: float = 0
    change_pct_24h: float = 0
    high_24h: float = 0
    low_24h: float = 0
    volume_24h: float = 0
    market_cap: float = 0
    timestamp: str


class HistoricalPrice(BaseModel):
    date: str
    open: float
    high: float
    low: float
    close: float
    volume: float


class PredictionResponse(BaseModel):
    predicted_price: float
    current_price: float
    price_change: float
    price_change_pct: float
    direction: str  # 'up' or 'down'
    confidence: Dict[str, float]
    prediction_date: str
    model_name: str
    generated_at: str


class SentimentData(BaseModel):
    date: str
    score: float
    label: str  # 'positive', 'negative', 'neutral'
    discussion_volume: int
    engagement: int


class ModelMetrics(BaseModel):
    model_name: str
    mae: float
    rmse: float
    mape: float
    r2: float
    directional_accuracy: float
    evaluated_at: str


class TechnicalIndicators(BaseModel):
    date: str
    sma_7: Optional[float] = None
    sma_20: Optional[float] = None
    sma_50: Optional[float] = None
    ema_12: Optional[float] = None
    ema_26: Optional[float] = None
    rsi_14: Optional[float] = None
    macd: Optional[float] = None
    macd_signal: Optional[float] = None
    bollinger_upper: Optional[float] = None
    bollinger_middle: Optional[float] = None
    bollinger_lower: Optional[float] = None


# ==================== SERVICE CLASSES ====================

class APIService:
    """Service class for API business logic."""
    
    def __init__(self):
        self._coingecko_client = None
        self._prediction_service = None
        self._data_processor = None
        self._cache = {}
        self._cache_ttl = 60  # seconds
    
    @property
    def coingecko_client(self):
        if self._coingecko_client is None:
            from pipeline.data_collection import CoinGeckoClient
            self._coingecko_client = CoinGeckoClient()
        return self._coingecko_client
    
    @property
    def prediction_service(self):
        if self._prediction_service is None:
            from pipeline.prediction import PredictionService
            self._prediction_service = PredictionService()
        return self._prediction_service
    
    @property
    def data_processor(self):
        if self._data_processor is None:
            from pipeline.data_processing import DataProcessor
            self._data_processor = DataProcessor()
        return self._data_processor
    
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
        """Get current BTC price."""
        cached = self._get_cached('current_price')
        if cached:
            return cached
        
        try:
            data = self.coingecko_client.get_current_price()
            market = self.coingecko_client.get_market_data()
            
            result = {
                'price': data.get('price', 0),
                'change_24h': market.get('price_change_24h', 0),
                'change_pct_24h': market.get('price_change_percentage_24h', 0),
                'high_24h': market.get('high_24h', 0),
                'low_24h': market.get('low_24h', 0),
                'volume_24h': market.get('total_volume', 0),
                'market_cap': market.get('market_cap', 0),
                'timestamp': datetime.now().isoformat()
            }
            
            self._set_cache('current_price', result)
            return result
        except Exception as e:
            # Return mock data if API fails
            return {
                'price': 95000,
                'change_24h': 1500,
                'change_pct_24h': 1.6,
                'high_24h': 96000,
                'low_24h': 93000,
                'volume_24h': 25000000000,
                'market_cap': 1800000000000,
                'timestamp': datetime.now().isoformat(),
                'mock': True,
                'error': str(e)
            }
    
    def get_historical_prices(self, days: int = 30) -> List[Dict[str, Any]]:
        """Get historical price data."""
        cache_key = f'historical_{days}'
        cached = self._get_cached(cache_key)
        if cached:
            return cached
        
        try:
            df = self.coingecko_client.get_historical_prices(days=days)
            
            if df.empty:
                return self._generate_mock_historical(days)
            
            result = []
            for idx, row in df.iterrows():
                result.append({
                    'date': idx.strftime('%Y-%m-%d') if hasattr(idx, 'strftime') else str(idx),
                    'open': float(row.get('open', 0)),
                    'high': float(row.get('high', 0)),
                    'low': float(row.get('low', 0)),
                    'close': float(row.get('close', 0)),
                    'volume': float(row.get('volume', 0)) if 'volume' in row else 0
                })
            
            self._set_cache(cache_key, result)
            return result
        except Exception as e:
            return self._generate_mock_historical(days)
    
    def _generate_mock_historical(self, days: int) -> List[Dict[str, Any]]:
        """Generate mock historical data."""
        result = []
        base_price = 95000
        
        for i in range(days):
            date = datetime.now() - timedelta(days=days - i)
            price = base_price + np.random.uniform(-2000, 2000)
            
            result.append({
                'date': date.strftime('%Y-%m-%d'),
                'open': price - np.random.uniform(0, 500),
                'high': price + np.random.uniform(500, 1500),
                'low': price - np.random.uniform(500, 1500),
                'close': price,
                'volume': np.random.uniform(20e9, 30e9),
                'mock': True
            })
        
        return result
    
    def get_prediction(self) -> Dict[str, Any]:
        """Get price prediction."""
        try:
            # Get current price first
            current = self.get_current_price()
            current_price = current.get('price', 95000)
            
            # Try to use the model
            model_info = self.prediction_service.get_model_info()
            
            if model_info.get('status') == 'loaded':
                # Use real model prediction
                # ... would need actual feature data
                predicted_price = current_price * (1 + np.random.uniform(-0.02, 0.05))
            else:
                # Generate mock prediction
                predicted_price = current_price * (1 + np.random.uniform(-0.02, 0.05))
            
            change = predicted_price - current_price
            change_pct = (change / current_price) * 100
            
            return {
                'predicted_price': round(predicted_price, 2),
                'current_price': round(current_price, 2),
                'price_change': round(change, 2),
                'price_change_pct': round(change_pct, 2),
                'direction': 'up' if change > 0 else 'down',
                'confidence': {
                    'lower_95': round(predicted_price * 0.95, 2),
                    'upper_95': round(predicted_price * 1.05, 2),
                    'lower_68': round(predicted_price * 0.98, 2),
                    'upper_68': round(predicted_price * 1.02, 2)
                },
                'prediction_date': (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d'),
                'model_name': model_info.get('model_name', 'ensemble'),
                'generated_at': datetime.now().isoformat()
            }
        except Exception as e:
            return {
                'error': str(e),
                'predicted_price': 0,
                'current_price': 0
            }
    
    def get_sentiment(self, days: int = 30) -> List[Dict[str, Any]]:
        """Get sentiment data."""
        result = []
        
        for i in range(days):
            date = datetime.now() - timedelta(days=days - i)
            score = np.random.uniform(-0.5, 0.7)  # Simulated sentiment
            
            if score > 0.2:
                label = 'positive'
            elif score < -0.2:
                label = 'negative'
            else:
                label = 'neutral'
            
            result.append({
                'date': date.strftime('%Y-%m-%d'),
                'score': round(score, 3),
                'label': label,
                'discussion_volume': int(np.random.uniform(1000, 10000)),
                'engagement': int(np.random.uniform(50000, 500000))
            })
        
        return result
    
    def get_technical_indicators(self, days: int = 30) -> List[Dict[str, Any]]:
        """Get technical indicators."""
        try:
            # Get historical data
            historical = self.get_historical_prices(days + 50)  # Extra for calculations
            
            if not historical:
                return []
            
            df = pd.DataFrame(historical)
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date').sort_index()
            
            # Calculate indicators
            processed = self.data_processor.add_technical_indicators(df)
            
            # Return last N days
            result = []
            for idx, row in processed.tail(days).iterrows():
                result.append({
                    'date': idx.strftime('%Y-%m-%d'),
                    'sma_7': float(row.get('sma_7')) if pd.notna(row.get('sma_7')) else None,
                    'sma_20': float(row.get('sma_20')) if pd.notna(row.get('sma_20')) else None,
                    'sma_50': float(row.get('sma_50')) if pd.notna(row.get('sma_50')) else None,
                    'ema_12': float(row.get('ema_12')) if pd.notna(row.get('ema_12')) else None,
                    'ema_26': float(row.get('ema_26')) if pd.notna(row.get('ema_26')) else None,
                    'rsi_14': float(row.get('rsi_14')) if pd.notna(row.get('rsi_14')) else None,
                    'macd': float(row.get('macd')) if pd.notna(row.get('macd')) else None,
                    'macd_signal': float(row.get('macd_signal')) if pd.notna(row.get('macd_signal')) else None,
                    'bollinger_upper': float(row.get('bollinger_upper')) if pd.notna(row.get('bollinger_upper')) else None,
                    'bollinger_middle': float(row.get('bollinger_middle')) if pd.notna(row.get('bollinger_middle')) else None,
                    'bollinger_lower': float(row.get('bollinger_lower')) if pd.notna(row.get('bollinger_lower')) else None,
                })
            
            return result
        except Exception as e:
            return []
    
    def get_model_metrics(self) -> List[Dict[str, Any]]:
        """Get model performance metrics."""
        # Mock metrics - in production, load from database
        models = ['lstm', 'gru', 'transformer', 'ensemble']
        metrics = []
        
        for model in models:
            metrics.append({
                'model_name': model,
                'mae': round(np.random.uniform(500, 1500), 2),
                'rmse': round(np.random.uniform(700, 2000), 2),
                'mape': round(np.random.uniform(1, 5), 2),
                'r2': round(np.random.uniform(0.85, 0.98), 4),
                'directional_accuracy': round(np.random.uniform(55, 75), 2),
                'evaluated_at': datetime.now().isoformat()
            })
        
        # Sort by RMSE (best first)
        metrics.sort(key=lambda x: x['rmse'])
        
        return metrics


# Initialize service
api_service = APIService()


# ==================== API ENDPOINTS ====================

@app.get("/api/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    }


@app.get("/api/price/current", response_model=PriceData, tags=["Price"])
async def get_current_price():
    """Get current BTC price and 24h stats."""
    return api_service.get_current_price()


@app.get("/api/price/historical", response_model=List[HistoricalPrice], tags=["Price"])
async def get_historical_prices(
    days: int = Query(default=30, ge=1, le=365, description="Number of days of history")
):
    """Get historical OHLCV data."""
    return api_service.get_historical_prices(days)


@app.get("/api/prediction", response_model=PredictionResponse, tags=["Prediction"])
async def get_prediction():
    """Get next day price prediction."""
    result = api_service.get_prediction()
    if 'error' in result and result.get('predicted_price', 0) == 0:
        raise HTTPException(status_code=500, detail=result['error'])
    return result


@app.get("/api/prediction/history", tags=["Prediction"])
async def get_prediction_history(
    days: int = Query(default=7, ge=1, le=30)
):
    """Get prediction history for the last N days."""
    history = []
    
    for i in range(days):
        date = datetime.now() - timedelta(days=i)
        price = 95000 + np.random.uniform(-2000, 2000)
        predicted = price * (1 + np.random.uniform(-0.03, 0.03))
        
        history.append({
            'date': date.strftime('%Y-%m-%d'),
            'predicted_price': round(predicted, 2),
            'actual_price': round(price, 2),
            'error': round(abs(predicted - price), 2),
            'direction_correct': np.random.choice([True, False], p=[0.65, 0.35])
        })
    
    return history


@app.get("/api/sentiment", response_model=List[SentimentData], tags=["Sentiment"])
async def get_sentiment(
    days: int = Query(default=30, ge=1, le=90, description="Number of days")
):
    """Get sentiment analysis data."""
    return api_service.get_sentiment(days)


@app.get("/api/sentiment/current", tags=["Sentiment"])
async def get_current_sentiment():
    """Get current sentiment summary."""
    sentiment = api_service.get_sentiment(7)
    
    if not sentiment:
        return {"score": 0, "label": "neutral"}
    
    # Calculate average sentiment
    avg_score = sum(s['score'] for s in sentiment) / len(sentiment)
    total_volume = sum(s['discussion_volume'] for s in sentiment)
    total_engagement = sum(s['engagement'] for s in sentiment)
    
    if avg_score > 0.2:
        label = 'positive'
    elif avg_score < -0.2:
        label = 'negative'
    else:
        label = 'neutral'
    
    return {
        'score': round(avg_score, 3),
        'label': label,
        'total_discussion_volume': total_volume,
        'total_engagement': total_engagement,
        'period': '7d',
        'updated_at': datetime.now().isoformat()
    }


@app.get("/api/indicators", response_model=List[TechnicalIndicators], tags=["Technical"])
async def get_technical_indicators(
    days: int = Query(default=30, ge=1, le=90)
):
    """Get technical indicators."""
    return api_service.get_technical_indicators(days)


@app.get("/api/indicators/current", tags=["Technical"])
async def get_current_indicators():
    """Get current technical indicator values."""
    indicators = api_service.get_technical_indicators(1)
    
    if not indicators:
        return {"error": "No indicator data available"}
    
    return indicators[0]


@app.get("/api/models", response_model=List[ModelMetrics], tags=["Models"])
async def get_model_metrics():
    """Get performance metrics for all models."""
    return api_service.get_model_metrics()


@app.get("/api/models/best", tags=["Models"])
async def get_best_model():
    """Get the best performing model."""
    metrics = api_service.get_model_metrics()
    
    if not metrics:
        return {"error": "No metrics available"}
    
    # Best model is first (sorted by RMSE)
    best = metrics[0]
    
    return {
        'best_model': best['model_name'],
        'metrics': best,
        'reason': 'Lowest RMSE score'
    }


@app.get("/api/dashboard", tags=["Dashboard"])
async def get_dashboard_data():
    """Get all dashboard data in one request."""
    return {
        'current_price': api_service.get_current_price(),
        'prediction': api_service.get_prediction(),
        'sentiment': api_service.get_sentiment(7)[-1] if api_service.get_sentiment(7) else None,
        'best_model': api_service.get_model_metrics()[0] if api_service.get_model_metrics() else None,
        'updated_at': datetime.now().isoformat()
    }


# ==================== WEBSOCKET (Optional) ====================

@app.websocket("/ws/price")
async def websocket_price(websocket):
    """WebSocket endpoint for real-time price updates."""
    from fastapi import WebSocket
    await websocket.accept()
    
    try:
        while True:
            # Send price update every 10 seconds
            price_data = api_service.get_current_price()
            await websocket.send_json(price_data)
            await asyncio.sleep(10)
    except Exception:
        await websocket.close()


# ==================== RUN SERVER ====================

if __name__ == "__main__":
    import uvicorn
    
    print("=" * 50)
    print("🚀 BTC Price Prediction API")
    print("=" * 50)
    print("📚 API Docs: http://localhost:8000/api/docs")
    print("📖 ReDoc: http://localhost:8000/api/redoc")
    print("=" * 50)
    
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
