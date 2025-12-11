# 🚀 BTC Price Prediction - Hybrid LSTM + Sentiment Analysis

> Machine Learning pipeline untuk prediksi harga Bitcoin menggunakan kombinasi LSTM, Transformer, dan Sentiment Analysis dengan metodologi CRISP-DM.

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📋 Daftar Isi

- [Overview](#-overview)
- [Arsitektur](#-architecture)
- [Fitur](#-features)
- [Instalasi](#-installation)
- [Menjalankan API](#-running-the-api)
- [API Endpoints](#-api-endpoints)
- [Pipeline ML](#-ml-pipeline)
- [Models](#-models)
- [Frontend Integration](#-frontend-integration)

---

## 📝 Overview

Sistem ini memprediksi harga Bitcoin menggunakan:
- **Data Sources**: CoinGecko, Twitter, Reddit, News API
- **Technical Indicators**: SMA, EMA, RSI, MACD, Bollinger Bands, ATR
- **Sentiment Analysis**: BERT/Transformer-based sentiment scoring
- **Models**: LSTM, GRU, Transformer, Ensemble
- **Orchestration**: Apache Airflow (optional)
- **Visualization**: Grafana Dashboard

---

## 🏗 Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      Apache Airflow (Orchestrator)               │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐           │
│  │  Twitter API │  │  Reddit API  │  │ CoinGecko API│           │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘           │
│         │                  │                  │                  │
│         └──────────────────┼──────────────────┘                  │
│                            ▼                                     │
│                 ┌─────────────────────┐                          │
│                 │   Data Processor    │                          │
│                 │ (Cleaning, Features)│                          │
│                 └──────────┬──────────┘                          │
│                            ▼                                     │
│    ┌───────────┬───────────┼───────────┬───────────┐            │
│    │   LSTM    │    GRU    │Transformer│  Ensemble │            │
│    └─────┬─────┴─────┬─────┴─────┬─────┴─────┬─────┘            │
│          └───────────┴───────────┴───────────┘                   │
│                            ▼                                     │
│                 ┌─────────────────────┐                          │
│                 │  Model Evaluator    │                          │
│                 │  (Best Model Select)│                          │
│                 └──────────┬──────────┘                          │
│                            ▼                                     │
│                 ┌─────────────────────┐                          │
│                 │ Prediction Service  │◄─────► REST API          │
│                 └─────────────────────┘                          │
│                            ▼                                     │
│                 ┌─────────────────────┐                          │
│                 │  PostgreSQL + Grafana│                         │
│                 └─────────────────────┘                          │
└─────────────────────────────────────────────────────────────────┘
```

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 📊 **Multi-Source Data** | Twitter, Reddit, CoinGecko, News API |
| 📈 **Technical Indicators** | SMA, EMA, RSI, MACD, Bollinger, ATR |
| 🤖 **4 Model Types** | LSTM, GRU, Transformer, Ensemble |
| 🎯 **Auto Model Selection** | Best model picked by lowest RMSE |
| 🔮 **Multi-Horizon Prediction** | 1, 3, 7, 14, 30 days ahead |
| 📉 **Confidence Intervals** | 68% and 95% bounds |
| 🌐 **REST API** | FastAPI with auto-generated docs |
| 📺 **Grafana Dashboard** | Real-time monitoring |

---

## 🛠 Installation

### Prerequisites
- Python 3.9+
- pip

### Setup

```bash
# Clone repository
git clone https://github.com/bimoBintang/btc-price-prediction-hybrid-lstm-sentiment-crispdm.git
cd machine-learning

# Create virtual environment
python -m venv myenv
source myenv/bin/activate  # On Windows: myenv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Setup environment variables
cp .env.example .env
# Edit .env with your API keys
```

### Environment Variables

```env
# Database
DATABASE_URL=postgresql://user:pass@localhost:5432/btc_prediction

# APIs
COINGECKO_API_KEY=your_key
NEWS_API_KEY=your_key
TWITTER_USERNAME=your_username
TWITTER_PASSWORD=your_password
REDDIT_CLIENT_ID=your_id
REDDIT_CLIENT_SECRET=your_secret
```

---

## 🚀 Running the API

### Quick Start

```bash
python run_api.py
```

### With Uvicorn

```bash
uvicorn src.api.api:app --reload --host 0.0.0.0 --port 8000
```

### Access Points

| URL | Description |
|-----|-------------|
| http://localhost:8000 | API Root |
| http://localhost:8000/api/docs | Swagger UI (Interactive Docs) |
| http://localhost:8000/api/redoc | ReDoc (Alternative Docs) |

---

## 📡 API Endpoints

### Health Check
```http
GET /api/health
```

### Price Data
```http
GET /api/price/current       # Current BTC price
GET /api/price/historical    # Historical OHLCV (params: days=30)
GET /api/price/chart         # Chart-formatted data
GET /api/price/ohlc          # Candlestick data
GET /api/price/stats         # Price statistics
```

### Predictions
```http
GET /api/prediction          # Next day prediction
GET /api/prediction/multi    # Multi-horizon (1d, 3d, 7d, 14d, 30d)
GET /api/prediction/history  # Historical predictions
GET /api/prediction/accuracy # Prediction accuracy metrics
```

### Sentiment
```http
GET /api/sentiment           # Sentiment data (params: days=30)
GET /api/sentiment/current   # Current sentiment summary
GET /api/sentiment/platforms # By platform (Twitter, Reddit, etc)
GET /api/sentiment/trending  # Trending topics
```

### Technical Indicators
```http
GET /api/indicators          # All indicators (params: days=30)
GET /api/indicators/current  # Latest indicator values
```

### Models
```http
GET /api/models              # All model metrics
GET /api/models/best         # Best performing model
GET /api/models/comparison   # Detailed comparison
GET /api/models/training-history  # Training loss history
```

### Dashboard (All-in-One)
```http
GET /api/dashboard           # All dashboard data in one request
```

### Example Response

```json
{
  "predicted_price": 97250.50,
  "current_price": 95000.00,
  "price_change": 2250.50,
  "price_change_pct": 2.37,
  "direction": "up",
  "confidence": {
    "lower_95": 92387.98,
    "upper_95": 102113.03
  },
  "prediction_date": "2024-12-12",
  "model_name": "ensemble"
}
```

---

## 🔄 ML Pipeline

### Run Full Pipeline

```bash
# Standalone mode (without Airflow)
python src/pipeline/btc_prediction_dag.py
```

### Pipeline Steps

1. **Data Collection** - Collect data from APIs
2. **Data Processing** - Clean, feature engineering
3. **Model Training** - Train LSTM, GRU, Transformer, Ensemble
4. **Evaluation** - Compare models, select best
5. **Prediction** - Generate predictions
6. **Export** - Save to database

### Directory Structure

```
src/
├── api/                          # REST API
│   ├── api.py                    # Main FastAPI app
│   ├── routes.py                 # Additional routes
│   └── config.py                 # Settings
│
├── pipeline/                     # ML Pipeline
│   ├── btc_prediction_dag.py     # Airflow DAG
│   ├── data_collection/          # Data scrapers
│   │   ├── coingecko_client.py
│   │   └── data_collector.py
│   ├── data_processing/          # Feature engineering
│   │   └── data_processor.py
│   ├── models/                   # ML models
│   │   ├── lstm_gru_model.py
│   │   ├── transformer_model.py
│   │   ├── ensemble_model.py
│   │   └── model_factory.py
│   ├── evaluation/               # Model evaluation
│   │   └── model_evaluator.py
│   ├── prediction/               # Prediction service
│   │   └── prediction_service.py
│   └── storage/                  # Data export
│       └── data_exporter.py
│
├── data_understanding/           # Sentiment scraping
│   └── scrape_sentiment.py
│
└── models/                       # Database schema
    └── databaseSchema.py
```

---

## 🤖 Models

### LSTM/GRU Model
- Multi-layer recurrent network
- Attention mechanism
- Dropout regularization
- Sequence length: 30 days

### Transformer Model
- Multi-head self-attention
- Positional encoding
- GELU activation
- Global attention pooling

### Ensemble Model
- Combines LSTM, GRU, Transformer
- Methods: Average, Weighted, Stacking
- Learnable weights
- Best of all approaches

### Metrics

| Model | MAE | RMSE | MAPE | R² |
|-------|-----|------|------|-----|
| LSTM | ~1000 | ~1500 | ~2.5% | 0.92 |
| GRU | ~1100 | ~1600 | ~2.7% | 0.90 |
| Transformer | ~900 | ~1400 | ~2.3% | 0.93 |
| **Ensemble** | **~800** | **~1200** | **~2.0%** | **0.95** |

---

## 🌐 Frontend Integration

### Fetching Data (Next.js)

```typescript
// lib/api.ts
const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export async function getCurrentPrice() {
  const res = await fetch(`${API_URL}/api/price/current`);
  return res.json();
}

export async function getPrediction() {
  const res = await fetch(`${API_URL}/api/prediction`);
  return res.json();
}

export async function getDashboardData() {
  const res = await fetch(`${API_URL}/api/dashboard`);
  return res.json();
}
```

### React Query Example

```typescript
// hooks/usePrice.ts
import { useQuery } from '@tanstack/react-query';

export function useCurrentPrice() {
  return useQuery({
    queryKey: ['price', 'current'],
    queryFn: () => fetch('/api/price/current').then(r => r.json()),
    refetchInterval: 60000, // Refresh every minute
  });
}
```

### TypeScript Types

```typescript
// types/api.ts
export interface PriceData {
  price: number;
  change_24h: number;
  change_pct_24h: number;
  high_24h: number;
  low_24h: number;
  volume_24h: number;
  market_cap: number;
  timestamp: string;
}

export interface PredictionData {
  predicted_price: number;
  current_price: number;
  price_change: number;
  price_change_pct: number;
  direction: 'up' | 'down';
  confidence: {
    lower_95: number;
    upper_95: number;
  };
  prediction_date: string;
  model_name: string;
}

export interface SentimentData {
  date: string;
  score: number;
  label: 'positive' | 'negative' | 'neutral';
  discussion_volume: number;
  engagement: number;
}
```

### Environment Variables (Frontend)

```env
# .env.local
NEXT_PUBLIC_API_URL=http://localhost:8000
```

---

## 📊 Grafana Dashboard

Import `config/grafana_config.json` to Grafana for visualization:

- Current BTC Price
- Predicted Price
- Model Accuracy (RMSE)
- Sentiment Score
- Price History Chart
- Price vs Prediction
- Sentiment Trend
- Discussion Volume
- Model Comparison Table
- Technical Indicators (RSI)

---

## 🧪 Testing

```bash
# Syntax check all modules
python -m py_compile src/api/api.py
python -m py_compile src/pipeline/btc_prediction_dag.py

# Run API tests (if available)
pytest tests/ -v
```

---

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

---

## 👥 Contributors

- **Bimo Bintang** - Initial work

---

## 🙏 Acknowledgments

- CoinGecko for market data API
- Hugging Face for Transformer models
- CRISP-DM methodology for ML project structure