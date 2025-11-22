import torch

from src.data_understanding.load_price import load_btc_price
from src.data_understanding.scrape_sentiment import get_

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

def main():
    get_best_device()


    # Data Understand
    btc = load_btc_price()


