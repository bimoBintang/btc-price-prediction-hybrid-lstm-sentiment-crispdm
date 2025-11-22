import yfinance as yf
import json
from datetime import datetime, timedelta
import pandas as pd


# async def get_btc_price():
#     start_date = datetime(2025, 11, 17)
#     end_date = start_date - timedelta(days=1095)

def load_btc_price():
    with open('BTC-USD_historical_data.json', 'r') as f:
        existing_data = json.load(f)

    # Ambil data harga BTC dari YFinance
    start_date = datetime(2025, 11, 13)
    end_date = start_date - timedelta(days=1095) # 3 tahun

    data_btc = yf.download(tickers='BTC-USD',start=start_date, end=end_date, interval='5m')
    encoded = data_btc.to_json()
    decoded = json.loads(encoded)
    close = decoded[['Open', 'High', 'Low', 'Close', 'Volume']]
    data_btc.index = pd.to_datetime(data_btc.index)

    for kj in existing_data:
        if kj in close:
            close[kj] = existing_data[kj]
    
    # simpan data menggunakan json
    data_dict = data_btc.to_dict()
    with open('BTC-USD_historical_data.json', 'w') as f:
        json.dump(data_dict, f, indent=2, default=str)

    
    # Data Sentiment harian menggunakan raddit dan twitter API
    sentiment_df = pd.read_json('btc_sentiment_daily.json', orient='index')
    sentiment_df.index = pd.to_datetime(sentiment_df.index)
    print(f"Data sentiment: {sentiment_df.shape[0]} hari")
