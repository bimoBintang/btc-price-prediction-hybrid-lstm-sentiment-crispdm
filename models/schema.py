from pydantic import BaseModel
from typing import List, Optional, Dict


class PredictionRequest(BaseModel):
    model_type: str = 'lstm'
    timeframe: str = '30days'
    includes_sentiment:bool = True

class PredictionResponse(BaseModel): 
    prediction:float
    confidence:float
    signal:str # Buy, Sell, Hold
    metrics:bool
    timestamp:str

class SentimentRequest(BaseModel):
    sources: list[str] = ["twitter", "reddit", "facebook", "instagram", "telegram", "news"]
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    query: Optional[str] = None

class SentimentResponse(BaseModel):
    sentiment_scores: List[Dict]
    breakdown:Dict
    overall_sentiment: float
    timestamp: str

class HistoricalDataResponse(BaseModel):
    historical_price: List[Dict]
    technical_indicators: Dict
    timestamp: str
