# API Documentation: Crypto Price Prediction with Sentiment Analysis

**Version:** 1.0.0
**Last Updated:** 2025
**Backend:** Python (FastAPI/Flask) with SQLAlchemy
**Frontend:** Next.js

## 📝 Overview

Dokumentasi ini menjelaskan endpoint REST API yang digunakan untuk komunikasi antara backend Python dan frontend Next.js. [cite_start]Sistem ini mendukung proyek prediksi harga Bitcoin menggunakan Hybrid Model (LSTM + Sentiment Analysis) sesuai dengan metodologi CRISP-DM[cite: 16].

Base URL: `http://localhost:8000/api/v1` (Default Development)

---

## 🔐 Authentication

Saat ini API bersifat publik untuk keperluan development dashboard. Jika diperlukan di masa depan, otentikasi akan menggunakan **Bearer Token**.

- **Header:** `Authorization: Bearer <token>`

---

## 📚 Resources

### 1. Cryptocurrency Data
Mengelola data aset kripto yang tersedia.

#### 1.1 Get All Cryptocurrencies
Mengambil daftar semua cryptocurrency yang didukung untuk ditampilkan di dropdown menu atau halaman *overview*.

- **Endpoint:** `GET /cryptocurrencies`
- **Query Parameters:**
  - `is_active` (boolean, optional): Filter status aktif. Default: `true`.
- **Response (200 OK):**
  ```json
  [
    {
      "crypto_id": 1,
      "symbol": "BTC",
      "name": "Bitcoin",
      "coingecko_id": "bitcoin",
      "market_cap_rank": 1,
      "logo_url": "[https://assets.coingecko.com/coins/images/1/small/bitcoin.png](https://assets.coingecko.com/coins/images/1/small/bitcoin.png)"
    },
    {
      "crypto_id": 2,
      "symbol": "ETH",
      "name": "Ethereum",
      "coingecko_id": "ethereum",
      "market_cap_rank": 2,
      "logo_url": "..."
    }
  ]

2. Market Data (Time Series)
Mengambil data historis OHLCV untuk visualisasi grafik harga (Candlestick Chart) di Next.js (menggunakan Recharts/Chart.js).

2.1 Get Historical Prices
Data ini bersumber dari tabel CryptoPrice.

Endpoint: GET /market/prices/{symbol}

Path Parameters:

symbol: Simbol koin (misal: BTC).

Query Parameters:

start_date (string, required): Format YYYY-MM-DD.

end_date (string, required): Format YYYY-MM-DD.

timeframe (string, optional): Default 1d.

Response (200 OK):

JSON

[
  {
    "timestamp": "2024-01-01T00:00:00Z",
    "date_only": "2024-01-01",
    "open": 42000.50,
    "high": 42500.00,
    "low": 41800.00,
    "close": 42250.75,
    "volume": 15000000.00
  },
  {
    "timestamp": "2024-01-02T00:00:00Z",
    "date_only": "2024-01-02",
    "open": 42250.75,
    "high": 43000.00,
    "low": 42100.00,
    "close": 42800.00,
    "volume": 18000000.00
  }
]
2.2 Get Technical Indicators
Mengambil indikator teknikal (RSI, MACD, SMA) untuk overlay pada grafik harga.

Endpoint: GET /market/indicators/{symbol}

Query Parameters: start_date, end_date.

Response (200 OK):

JSON

[
  {
    "date_only": "2024-01-02",
    "sma_50": 41500.20,
    "rsi_14": 55.4,
    "macd": 120.5,
    "macd_signal": 115.2,
    "bollinger_upper": 44000.00,
    "bollinger_lower": 40000.00
  }
]
3. Sentiment Analysis Data
Endpoint ini krusial untuk fitur "Hybrid Model", menampilkan bagaimana sentimen sosial media (Twitter/Reddit) mempengaruhi pasar.

3.1 Get Daily Sentiment Aggregation
Mengambil skor sentimen harian yang sudah diolah. Digunakan untuk grafik korelasi antara Harga vs Sentimen.

Endpoint: GET /sentiment/daily/{symbol}

Query Parameters: start_date, end_date.

Response (200 OK):

JSON

[
  {
    "date_only": "2024-01-02",
    "total_posts": 5430,
    "net_sentiment_score": 0.45,  // Skala -1.0 s/d 1.0
    "fear_index": 0.30,
    "greed_index": 0.70,
    "positive_percentage": 65.5,
    "negative_percentage": 20.5,
    "neutral_percentage": 14.0
  }
]
3.2 Get Recent Social Posts (Feed)
Menampilkan live feed atau daftar postingan terbaru terkait aset di dashboard untuk verifikasi manual pengguna.

Endpoint: GET /sentiment/posts/{symbol}

Query Parameters:

limit (int): Jumlah post (default: 10).

platform (string): Filter platform (e.g., 'Twitter', 'Reddit').

Response (200 OK):

JSON

[
  {
    "post_id": 10239,
    "platform": "Twitter",
    "author": "CryptoWhale",
    "text": "Bitcoin looking bullish above 45k! #BTC",
    "posted_at": "2024-01-02T14:30:00Z",
    "sentiment_label": "positive",
    "sentiment_score": 0.85
  }
]
4. Prediction & Model Performance
Endpoint untuk menampilkan hasil prediksi AI (LSTM) dan metrik evaluasinya.

4.1 Get Price Predictions (Forecast)
Mengembalikan data prediksi untuk dibandingkan dengan harga asli (Actual vs Predicted).

Endpoint: GET /predictions/{symbol}

Query Parameters:

model_name: 'Hybrid-LSTM' atau 'Benchmark-LSTM' (Default: 'Hybrid-LSTM').

days_ahead: Jumlah hari prediksi ke depan (misal: 7).

Response (200 OK):

JSON

[
  {
    "date": "2024-01-03",
    "actual_price": 43000.00,       // Null jika masa depan
    "predicted_price": 43150.20,
    "lower_bound": 42800.00,        // Confidence Interval
    "upper_bound": 43500.00,
    "error_diff": 150.20,           // Selisih (hanya jika actual_price ada)
    "is_future": false
  },
  {
    "date": "2024-01-04",
    "actual_price": null,
    "predicted_price": 43400.00,
    "lower_bound": 43000.00,
    "upper_bound": 43800.00,
    "is_future": true
  }
]
4.2 Get Model Evaluation Metrics
Menampilkan performa model untuk membuktikan kriteria sukses (Target MAE < 5%).


Endpoint: GET /predictions/metrics

Query Parameters: symbol, model_name.

Response (200 OK):

JSON

{
  "model_name": "Hybrid-LSTM",
  "evaluation_date": "2024-01-01",
  "metrics": {
    "mae": 120.50,            // Mean Absolute Error
    "rmse": 150.75,           // Root Mean Squared Error
    "mape_percentage": 3.2,   // Mean Absolute Percentage Error
    "r2_score": 0.89
  },
  "improvement_vs_benchmark": {
    "mae_reduction_percentage": 6.5, // Sukses jika > 5%
    "status": "SUCCESS"
  }
}
🛠️ System Health & Utils
Health Check
Digunakan oleh Kubernetes atau Load Balancer, serta untuk cek status database/redis.

Endpoint: GET /health

Response (200 OK):

JSON

{
  "status": "healthy",
  "database": "connected",
  "redis": "connected",
  "timestamp": "2024-01-02T15:00:00Z"
}
💻 Type Definitions (TypeScript Interface)
Gunakan definisi tipe ini di frontend Next.js Anda (src/types/api.ts) untuk type-safety.

TypeScript

// Tipe untuk Grafik Harga
export interface CryptoPriceData {
  timestamp: string;
  date_only: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

// Tipe untuk Sentimen Harian
export interface DailySentiment {
  date_only: string;
  net_sentiment_score: number;
  fear_index: number;
  greed_index: number;
  total_posts: number;
}

// Tipe untuk Prediksi
export interface PredictionData {
  date: string;
  actual_price: number | null;
  predicted_price: number;
  lower_bound?: number;
  upper_bound?: number;
  is_future: boolean;
}

### Tips Integrasi Next.js

1.  **Fetching Data:** Gunakan library seperti **TanStack Query (React Query)** atau **SWR** di Next.js untuk memanggil endpoint ini. Ini akan menangani *caching*, *loading state*, dan *error handling* secara otomatis.
2.  **Server Components:** Untuk data awal (seperti daftar Crypto), Anda bisa memanggil endpoint ini langsung di *Server Components* (`app/page.tsx`) agar SEO lebih baik dan load awal lebih cepat.
3.  **Environment Variables:** Pastikan URL backend disimpan di `.env.local` frontend:
    `NEXT_PUBLIC_API_URL=http://localhost:8000/api/v1`