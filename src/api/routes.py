"""
API Routes for modular endpoint organization.
"""
from fastapi import APIRouter, Query, HTTPException
from datetime import datetime, timedelta
from typing import List, Dict, Any
import numpy as np

# Create routers for different endpoint groups
price_router = APIRouter(prefix="/api/price", tags=["Price"])
prediction_router = APIRouter(prefix="/api/prediction", tags=["Prediction"])
sentiment_router = APIRouter(prefix="/api/sentiment", tags=["Sentiment"])
models_router = APIRouter(prefix="/api/models", tags=["Models"])


# ==================== PRICE ROUTES ====================

@price_router.get("/chart")
async def get_price_chart(
    days: int = Query(default=30, ge=1, le=365),
    interval: str = Query(default="daily", enum=["hourly", "daily"])
):
    """Get price data formatted for charting."""
    from .api import api_service
    
    historical = api_service.get_historical_prices(days)
    
    # Format for chart.js / recharts
    return {
        'labels': [item['date'] for item in historical],
        'datasets': [
            {
                'label': 'Close Price',
                'data': [item['close'] for item in historical],
                'borderColor': '#3b82f6',
                'fill': False
            },
            {
                'label': 'High',
                'data': [item['high'] for item in historical],
                'borderColor': '#22c55e',
                'fill': False
            },
            {
                'label': 'Low',
                'data': [item['low'] for item in historical],
                'borderColor': '#ef4444',
                'fill': False
            }
        ]
    }


@price_router.get("/ohlc")
async def get_ohlc_data(days: int = Query(default=30, ge=1, le=365)):
    """Get OHLC data for candlestick charts."""
    from .api import api_service
    
    historical = api_service.get_historical_prices(days)
    
    # Format for candlestick chart
    return [
        {
            'x': item['date'],
            'o': item['open'],
            'h': item['high'],
            'l': item['low'],
            'c': item['close']
        }
        for item in historical
    ]


@price_router.get("/stats")
async def get_price_stats():
    """Get price statistics."""
    from .api import api_service
    
    historical = api_service.get_historical_prices(365)
    
    if not historical:
        return {"error": "No data available"}
    
    closes = [item['close'] for item in historical]
    
    return {
        'current': closes[-1] if closes else 0,
        'high_52w': max(closes),
        'low_52w': min(closes),
        'average_52w': sum(closes) / len(closes),
        'change_7d': (closes[-1] - closes[-7]) / closes[-7] * 100 if len(closes) >= 7 else 0,
        'change_30d': (closes[-1] - closes[-30]) / closes[-30] * 100 if len(closes) >= 30 else 0,
        'change_90d': (closes[-1] - closes[-90]) / closes[-90] * 100 if len(closes) >= 90 else 0,
        'change_365d': (closes[-1] - closes[0]) / closes[0] * 100,
        'volatility_30d': np.std(closes[-30:]) if len(closes) >= 30 else 0
    }


# ==================== PREDICTION ROUTES ====================

@prediction_router.get("/multi")
async def get_multi_horizon_prediction():
    """Get predictions for multiple time horizons."""
    from .api import api_service
    
    current = api_service.get_current_price()
    base_price = current.get('price', 95000)
    
    predictions = {}
    
    for horizon in [1, 3, 7, 14, 30]:
        # Simulated predictions with increasing uncertainty
        trend = np.random.uniform(-0.02, 0.04) * (horizon / 7)
        predicted = base_price * (1 + trend)
        uncertainty = base_price * 0.02 * np.sqrt(horizon)
        
        predictions[f'{horizon}d'] = {
            'horizon': horizon,
            'date': (datetime.now() + timedelta(days=horizon)).strftime('%Y-%m-%d'),
            'predicted_price': round(predicted, 2),
            'lower_bound': round(predicted - uncertainty * 1.96, 2),
            'upper_bound': round(predicted + uncertainty * 1.96, 2),
            'confidence': round(max(0.5, 0.95 - horizon * 0.02), 2)
        }
    
    return {
        'current_price': round(base_price, 2),
        'predictions': predictions,
        'generated_at': datetime.now().isoformat()
    }


@prediction_router.get("/accuracy")
async def get_prediction_accuracy():
    """Get historical prediction accuracy."""
    # Mock accuracy data
    return {
        'overall_accuracy': 0.67,
        'direction_accuracy': 0.72,
        'average_error': 1250.50,
        'average_error_pct': 1.32,
        'best_model': 'ensemble',
        'metrics_by_horizon': {
            '1d': {'accuracy': 0.75, 'mae': 850},
            '3d': {'accuracy': 0.68, 'mae': 1200},
            '7d': {'accuracy': 0.62, 'mae': 1800}
        },
        'last_updated': datetime.now().isoformat()
    }


# ==================== SENTIMENT ROUTES ====================

@sentiment_router.get("/platforms")
async def get_sentiment_by_platform():
    """Get sentiment breakdown by platform."""
    platforms = ['twitter', 'reddit', 'news', 'telegram']
    
    result = {}
    for platform in platforms:
        score = np.random.uniform(-0.3, 0.7)
        result[platform] = {
            'score': round(score, 3),
            'label': 'positive' if score > 0.2 else ('negative' if score < -0.2 else 'neutral'),
            'posts_count': int(np.random.uniform(500, 5000)),
            'engagement': int(np.random.uniform(10000, 100000))
        }
    
    return {
        'platforms': result,
        'aggregated': {
            'score': round(sum(p['score'] for p in result.values()) / len(result), 3),
            'total_posts': sum(p['posts_count'] for p in result.values()),
            'total_engagement': sum(p['engagement'] for p in result.values())
        },
        'updated_at': datetime.now().isoformat()
    }


@sentiment_router.get("/trending")
async def get_trending_topics():
    """Get trending topics in crypto discussions."""
    topics = [
        {'topic': 'Bitcoin ETF', 'mentions': 5420, 'sentiment': 0.65},
        {'topic': 'Halving 2024', 'mentions': 3850, 'sentiment': 0.72},
        {'topic': 'BTC Price', 'mentions': 2900, 'sentiment': 0.45},
        {'topic': 'Whale Alert', 'mentions': 1800, 'sentiment': 0.15},
        {'topic': 'Mining', 'mentions': 1200, 'sentiment': 0.30},
    ]
    
    return {
        'trending': topics,
        'updated_at': datetime.now().isoformat()
    }


# ==================== MODEL ROUTES ====================

@models_router.get("/comparison")
async def get_model_comparison():
    """Get detailed model comparison."""
    from .api import api_service
    
    metrics = api_service.get_model_metrics()
    
    return {
        'models': metrics,
        'best_overall': metrics[0]['model_name'] if metrics else None,
        'best_by_metric': {
            'mae': min(metrics, key=lambda x: x['mae'])['model_name'] if metrics else None,
            'rmse': min(metrics, key=lambda x: x['rmse'])['model_name'] if metrics else None,
            'r2': max(metrics, key=lambda x: x['r2'])['model_name'] if metrics else None,
            'direction': max(metrics, key=lambda x: x['directional_accuracy'])['model_name'] if metrics else None
        },
        'last_evaluation': datetime.now().isoformat()
    }


@models_router.get("/training-history")
async def get_training_history():
    """Get model training history."""
    # Mock training history
    epochs = list(range(1, 51))
    
    return {
        'lstm': {
            'train_loss': [1.0 / (1 + 0.1 * e) + np.random.uniform(-0.02, 0.02) for e in epochs],
            'val_loss': [1.2 / (1 + 0.08 * e) + np.random.uniform(-0.03, 0.03) for e in epochs]
        },
        'transformer': {
            'train_loss': [0.9 / (1 + 0.12 * e) + np.random.uniform(-0.02, 0.02) for e in epochs],
            'val_loss': [1.1 / (1 + 0.09 * e) + np.random.uniform(-0.03, 0.03) for e in epochs]
        },
        'ensemble': {
            'train_loss': [0.8 / (1 + 0.15 * e) + np.random.uniform(-0.02, 0.02) for e in epochs],
            'val_loss': [1.0 / (1 + 0.11 * e) + np.random.uniform(-0.03, 0.03) for e in epochs]
        },
        'epochs': epochs
    }
