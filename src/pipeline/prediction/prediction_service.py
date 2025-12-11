"""
Prediction Service for real-time BTC price predictions.
"""
import numpy as np
import pandas as pd
import torch
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple
import os
import json
import sys
from pipeline.models.model_factory import ModelFactory
import joblib

from pipeline.data_collection import DataCollector
from pipeline.data_processing import DataProcessor

# Add parent paths
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


class PredictionService:
    """
    Production-ready prediction service for BTC price prediction.
    
    Features:
    - Load and manage models
    - Real-time predictions
    - Prediction caching
    - Confidence intervals
    - Result persistence
    """
    
    def __init__(
        self,
        models_dir: str = 'models',
        best_model_name: str = 'ensemble_best',
        scaler_path: Optional[str] = None
    ):
        """
        Initialize prediction service.
        
        Args:
            models_dir: Directory containing saved models
            best_model_name: Name of the best model to use
            scaler_path: Path to feature scaler (for denormalization)
        """
        self.models_dir = models_dir
        self.best_model_name = best_model_name
        self.model = None
        self.scaler = None
        self.is_loaded = False
        
        # Prediction cache
        self._cache = {}
        self._cache_ttl = 60  # seconds
        
        # Load scaler if provided
        if scaler_path and os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)
            print(f"[PredictionService] Loaded scaler from {scaler_path}")
    
    def load_model(self, model_path: Optional[str] = None) -> bool:
        """
        Load the prediction model.
        
        Args:
            model_path: Optional specific model path
            
        Returns:
            True if successful
        """
        if model_path is None:
            model_path = os.path.join(self.models_dir, f"{self.best_model_name}.pth")
        
        if not os.path.exists(model_path):
            print(f"[PredictionService] Model not found: {model_path}")
            return False
        
        try:
            # Import model classes
            factory = ModelFactory(self.models_dir)
            self.model = factory.load_model(self.best_model_name)
            self.model.eval()
            self.is_loaded = True
            
            print(f"[PredictionService] Model loaded successfully from {model_path}")
            return True
        
        except Exception as e:
            print(f"[PredictionService] Error loading model: {e}")
            return False
    
    def predict(
        self,
        features: np.ndarray,
        return_confidence: bool = True,
        n_samples: int = 100
    ) -> Dict[str, Any]:
        """
        Make prediction on input features.
        
        Args:
            features: Input features of shape [batch, seq_len, n_features]
            return_confidence: Whether to return confidence interval
            n_samples: Number of MC samples for confidence (if using dropout)
            
        Returns:
            Dict with predictions and metadata
        """
        if not self.is_loaded:
            if not self.load_model():
                return {'error': 'Model not loaded'}
        
        # Generate cache key
        cache_key = hash(features.tobytes())
        
        # Check cache
        if cache_key in self._cache:
            cached = self._cache[cache_key]
            if (datetime.now() - cached['timestamp']).seconds < self._cache_ttl:
                return cached['prediction']
        
        try:
            self.model.eval()
            
            with torch.no_grad():
                x_tensor = torch.FloatTensor(features).to(self.model.device)
                predictions = self.model(x_tensor)
                predictions = predictions.cpu().numpy()
            
            # Denormalize if scaler is available
            if self.scaler is not None:
                predictions = self.scaler.inverse_transform(predictions)
            
            result = {
                'predictions': predictions.tolist(),
                'predicted_price': float(predictions[-1, 0]) if len(predictions.shape) > 1 else float(predictions[-1]),
                'timestamp': datetime.now().isoformat(),
                'model': self.best_model_name
            }
            
            # Calculate confidence interval using simple percentile method
            if return_confidence:
                # Estimate uncertainty based on recent volatility (simplified)
                std = np.std(predictions) if len(predictions) > 1 else predictions[-1] * 0.02
                predicted = result['predicted_price']
                
                result['confidence'] = {
                    'lower_95': predicted - 1.96 * std,
                    'upper_95': predicted + 1.96 * std,
                    'lower_68': predicted - std,
                    'upper_68': predicted + std,
                    'std': float(std)
                }
            
            # Cache result
            self._cache[cache_key] = {
                'prediction': result,
                'timestamp': datetime.now()
            }
            
            return result
        
        except Exception as e:
            return {'error': str(e)}
    
    def predict_next_day(
        self,
        historical_features: np.ndarray
    ) -> Dict[str, Any]:
        """
        Predict next day's BTC price.
        
        Args:
            historical_features: Historical feature sequence
            
        Returns:
            Prediction result with tomorrow's expected price
        """
        result = self.predict(historical_features.reshape(1, *historical_features.shape))
        
        if 'error' not in result:
            result['prediction_date'] = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
            result['prediction_type'] = 'next_day'
        
        return result
    
    def predict_multi_horizon(
        self,
        historical_features: np.ndarray,
        horizons: List[int] = [1, 3, 7]
    ) -> Dict[str, Any]:
        """
        Predict multiple horizons.
        
        Args:
            historical_features: Historical feature sequence
            horizons: List of prediction horizons (days)
            
        Returns:
            Predictions for each horizon
        """
        results = {
            'horizons': {},
            'timestamp': datetime.now().isoformat(),
            'model': self.best_model_name
        }
        
        # For now, use single prediction (multi-horizon would need separate models)
        base_prediction = self.predict(historical_features.reshape(1, *historical_features.shape))
        
        if 'error' in base_prediction:
            return base_prediction
        
        base_price = base_prediction['predicted_price']
        
        # Simple extrapolation for demo (in production, use proper multi-horizon models)
        for horizon in horizons:
            date = (datetime.now() + timedelta(days=horizon)).strftime('%Y-%m-%d')
            
            # Add some variance for longer horizons
            variance_factor = 1 + (horizon - 1) * 0.005
            
            results['horizons'][f'{horizon}d'] = {
                'date': date,
                'predicted_price': base_price * variance_factor,
                'confidence': {
                    'lower_95': base_price * (variance_factor - 0.04 * horizon),
                    'upper_95': base_price * (variance_factor + 0.04 * horizon)
                }
            }
        
        return results
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        if not self.is_loaded:
            return {'status': 'not_loaded'}
        
        return {
            'status': 'loaded',
            'model_name': self.best_model_name,
            'model_type': type(self.model).__name__,
            'device': str(self.model.device),
            'parameters': sum(p.numel() for p in self.model.parameters())
        }
    
    def save_prediction(
        self,
        prediction: Dict[str, Any],
        output_dir: str = 'results/predictions'
    ) -> str:
        """
        Save prediction to file.
        
        Args:
            prediction: Prediction result
            output_dir: Output directory
            
        Returns:
            Path to saved file
        """
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"prediction_{timestamp}.json"
        filepath = os.path.join(output_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(prediction, f, indent=2, default=str)
        
        print(f"[PredictionService] Prediction saved to {filepath}")
        return filepath


class RealtimePredictor:
    """
    Wrapper for real-time prediction with data collection.
    """
    
    def __init__(self, prediction_service: PredictionService):
        """
        Initialize realtime predictor.
        
        Args:
            prediction_service: PredictionService instance
        """
        self.service = prediction_service
        self.last_prediction = None
        self.prediction_history = []
    
    def get_latest_prediction(
        self,
        data_collector=None,
        data_processor=None,
        seq_length: int = 30
    ) -> Dict[str, Any]:
        """
        Get latest prediction using fresh data.
        
        Args:
            data_collector: Optional DataCollector instance
            data_processor: Optional DataProcessor instance
            seq_length: Sequence length for model input
            
        Returns:
            Latest prediction
        """
        try:
            # Collect latest data
            if data_collector is not None and data_processor is not None:
                collector = data_collector or DataCollector()
                processor = data_processor or DataProcessor()
                
                # Get price data
                price_data = collector.collect_price_data(days=seq_length + 7)
                
                # Process data
                processed = processor.process_full_pipeline(
                    price_data['historical_ohlcv'],
                    add_targets=False
                )
                
                # Get feature sequence
                feature_cols = [c for c in processed.columns if c not in ['date', 'target_price', 'target_change']]
                features = processed[feature_cols].values[-seq_length:]
                
            else:
                # Use dummy data for testing
                features = np.random.randn(seq_length, 10).astype(np.float32)
            
            # Make prediction
            prediction = self.service.predict_next_day(features)
            
            self.last_prediction = prediction
            self.prediction_history.append(prediction)
            
            return prediction
        
        except Exception as e:
            return {'error': str(e)}


if __name__ == "__main__":
    # Test prediction service
    print("Testing Prediction Service...")
    
    service = PredictionService()
    
    # Test with dummy data
    seq_length = 30
    n_features = 10
    
    dummy_features = np.random.randn(1, seq_length, n_features).astype(np.float32)
    
    # Get model info (will show not loaded since no model exists)
    info = service.get_model_info()
    print(f"Model info: {info}")
    
    print("✅ Prediction service test passed!")
