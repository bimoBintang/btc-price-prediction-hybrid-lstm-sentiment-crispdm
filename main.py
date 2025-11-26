import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from datetime import datetime
import numpy as np

import pandas as pd
from pathlib import Path

from src.data_understanding.load_price import load_btc_price
from src.data_understanding.scrape_sentiment import get_
from models.schema import PredictionResponse, PredictionRequest

# --- Load Model & Artefak Sekali Saat Startup ---
MODEL_PATH = Path("models/hybrid_lstm_btc_sentiment.h5")
SCALER_X_PATH = Path("scalers/scaler_X.pkl")
SCALER_Y_PATH = Path("scalers/scaler_y.pkl")
FEATURES_PATH = Path("scalers/feature_columns.pkl")

def get_best_device():
    # Check for CUDA (NVIDIA GPU) availability
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    # Check for MPS (Apple Silicon GPU) availability
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        print("Using Apple Silicon MPS device")
    # Fallback to CPU
    else:
        device = torch.device('cpu')
        print("Using CPU")
    
    return device

app = FastAPI(
    title="Prediksi Harga Bitcoin Hybrid LSTM + Sentimen",
    description="""
    <h3>Author: Bimo Bintang Siswanto (2370231011)</h3>
    <p>Proyek Tugas Akhir - Universitas Krisnadwipayana 2025</p>
    <p>Metodologi: <strong>CRISP-DM</strong> | Model: <strong>Hybrid LSTM + Sentimen Twitter/Reddit</strong></p>
    """,
    version="1.0.0",
    contact={"name": "Bimo Bintang Siswanto", "nim": "2370231011"}
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)




@app.get("/")
async def home():
    return {
        "message": "API Prediksi Bitcoin Hybrid LSTM + Sentimen",
        "status": "AKTIF & SIAP DIGUNAKAN",
        "author": "Bimo Bintang Siswanto (2370231011)",
        "university": "Universitas Krisnadwipayana",
        "docs": "/docs",
        "device_used": str(get_best_device)
    }

# @app.post('/api/data/btc/predict', response_model=PredictionResponse)
# async def predict_bitcoin_price(request: PredictionRequest):
    

def main():
    get_best_device()


    # Data Understand
    btc = load_btc_price()


