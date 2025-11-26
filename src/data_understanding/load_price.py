import yfinance as yf
import json
from datetime import datetime, timedelta
import pandas as pd
import os
import requests
import time


class BitcoinPriceColletor:

    def __init__(self):
        self.source = {
            'yahoo': self._fetch_from_yahoo,
            'coingecko': self._fetch_from_coingecko,
            'coinbase': self._fetch_from_coinbase
        }

    @staticmethod
    def _fetch_from_yahoo():
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

    @staticmethod
    def _fetch_from_coingecko(self, days: int = 30) -> pd.DataFrame:
        try:
            url = os.environ.get('COINGECKO_API_KEY') if os.environ.get('COINGECKO_API_KEY') else 'value'

            params = {
                'vs_currency': 'usd',
                'days': days,
                'interval': 'daily'
            }

            response = requests.get(url=url, params=params)
            response.raise_for_status()

            data = response.json()

            prices = data['prices']
            volumes = data['total_volumes']

            df = pd.DataFrame({
                'timestamp': [p[0] for p in prices],
                'Close': [p[1] for p in prices],
                'Volume': [v[1] for v in volumes]
            })

            # convert timestamp
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)


            df['Open'] = df['Close'].shift(1)
            df['High'] = df['Close'] * 1.01
            df['Low'] = df['Close'] * 0.99

            df = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
        
            print(f"Fetched {len(df)} days from Binance")
            return df
        except Exception as e:
            print(f"CoinGecko error: {e}")
            return pd.DataFrame()
        

    @staticmethod
    def _fetch_from_coinbase(self, days: int =30) -> pd.DataFrame:
        try:
            url = os.environ.get('COINBASE_API_KEY') if os.environ.get('COINBASE_API_KEY') else 'value'

            end_date = datetime()
            start_date = end_date - timedelta(days=days)

            params = {
                'start': start_date.isoformat(),
                'end': end_date.isoformat(),
                'granularity': 86400 # 1 days in second
            }

            response = requests.get(url=url, params=params)
            response.raise_for_status()

            data = response.json()

             # Data format: [time, low, high, open, close, volume]
            df = pd.DataFrame(data=data, columns=['timestamp', 'Low', 'High', 'Open', 'Close', 'Volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
            df.set_index('timestamp', inplace=True)
            df.sort_index()

            print(f"✓ Fetched {len(df)} days from Coinbase")
            return df
            
        except Exception as e:
            print(f"✗ Coinbase error: {e}")
            return pd.DataFrame()
        

    def fetch(self, days:int = 30, source:str = 'yahoo') -> pd.DataFrame:
        if source not in self.sources:
            raise ValueError(f"Invalid source. Choose from: {list(self.sources.keys())}")
        
        return self.sources[source](days)
    
    
    def fetch_with_fallback(self, days: int = 30) -> pd.DataFrame:

        source_order = ['yahoo', 'coingecko', 'coinbase']

        for source in source_order:
            df = self.fetch(days=days, source=source)

            if not df.empty:
                return df
            
            time.sleep(1) # ratelimitng

        raise Exception("All data sources failed!")