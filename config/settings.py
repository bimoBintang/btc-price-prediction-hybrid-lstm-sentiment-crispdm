import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# Data paths
DATA_RAW = BASE_DIR / "data" / "raw"
DATA_PROCESSED = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"
SCALERS_DIR = BASE_DIR / "scalers"
RESULTS_DIR = BASE_DIR / "results"

for dir_path in [DATA_RAW, DATA_PROCESSED, MODELS_DIR, SCALERS_DIR, RESULTS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Model parameters
TIME_STEPS = 60
EPOCHS = 50
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.1
RANDOM_SEED = 42

# API Keys (isi sendiri)
REDDIT_CLIENT_ID = "YOUR_REDDIT_CLIENT_ID"
REDDIT_CLIENT_SECRET = "YOUR_REDDIT_CLIENT_SECRET"
REDDIT_USER_AGENT = "btc_sentiment_bimo_2370231011"

# Success criteria
MIN_MAE_IMPROVEMENT = 5.0  # %
MIN_SHAP_SENTIMENT = 0.001