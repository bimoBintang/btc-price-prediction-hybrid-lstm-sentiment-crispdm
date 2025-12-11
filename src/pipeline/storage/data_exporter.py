"""
Data Exporter for persisting data to database and files.
"""
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any, Optional, List
import os
import json
import sys

from models.databaseSchema import DatabaseManager, ModelPrediction, CryptoPrice, ModelPerformanceMetric, DailySentimentAggregation

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))


class DataExporter:
    """
    Export and persist data to database and files.
    
    Features:
    - Export to PostgreSQL database
    - Export to JSON/CSV files
    - Export predictions and metrics
    """
    
    def __init__(
        self,
        db_connection_string: Optional[str] = None,
        output_dir: str = 'data/processed'
    ):
        """
        Initialize data exporter.
        
        Args:
            db_connection_string: PostgreSQL connection string
            output_dir: Directory for file exports
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.db_session = None
        
        if db_connection_string:
            self._init_db(db_connection_string)
    
    def _init_db(self, connection_string: str):
        """Initialize database connection."""
        try:
            self.db_manager = DatabaseManager(connection_string)
            self.db_session = self.db_manager.get_session()
            print("[DataExporter] Database connection established")
        except Exception as e:
            print(f"[DataExporter] Warning: Database connection failed: {e}")
            self.db_session = None
    
    # ==================== FILE EXPORTS ====================
    
    def export_to_csv(
        self,
        df: pd.DataFrame,
        filename: str,
        include_timestamp: bool = True
    ) -> str:
        """
        Export DataFrame to CSV.
        
        Args:
            df: DataFrame to export
            filename: Output filename (without extension)
            include_timestamp: Add timestamp to filename
            
        Returns:
            Path to exported file
        """
        if include_timestamp:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{filename}_{timestamp}"
        
        filepath = os.path.join(self.output_dir, f"{filename}.csv")
        df.to_csv(filepath, index=True)
        
        print(f"[DataExporter] Exported to {filepath}")
        return filepath
    
    def export_to_json(
        self,
        data: Any,
        filename: str,
        include_timestamp: bool = True
    ) -> str:
        """
        Export data to JSON.
        
        Args:
            data: Data to export (dict, list, or DataFrame)
            filename: Output filename (without extension)
            include_timestamp: Add timestamp to filename
            
        Returns:
            Path to exported file
        """
        if include_timestamp:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{filename}_{timestamp}"
        
        filepath = os.path.join(self.output_dir, f"{filename}.json")
        
        if isinstance(data, pd.DataFrame):
            data = data.to_dict(orient='records')
        
        # Convert datetime objects to strings
        def convert_datetime(obj):
            if isinstance(obj, (datetime, pd.Timestamp)):
                return obj.isoformat()
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.float64, np.float32)):
                return float(obj)
            return obj
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=convert_datetime)
        
        print(f"[DataExporter] Exported to {filepath}")
        return filepath
    
    # ==================== DATABASE EXPORTS ====================
    
    def export_price_data(
        self,
        df: pd.DataFrame,
        crypto_id: int = 1,
        source_id: int = 1
    ) -> int:
        """
        Export price data to database.
        
        Args:
            df: Price data DataFrame
            crypto_id: Cryptocurrency ID
            source_id: Data source ID
            
        Returns:
            Number of records inserted
        """
        if self.db_session is None:
            print("[DataExporter] Database not connected, exporting to file instead")
            self.export_to_csv(df, 'price_data')
            return 0
        
        try:
            
            records = []
            for idx, row in df.iterrows():
                record = CryptoPrice(
                    crypto_id=crypto_id,
                    source_id=source_id,
                    timestamp=idx if isinstance(idx, datetime) else datetime.now(),
                    date_only=idx.date() if hasattr(idx, 'date') else datetime.now().date(),
                    open_price=row.get('open', 0),
                    high_price=row.get('high', 0),
                    low_price=row.get('low', 0),
                    close_price=row.get('close', 0),
                    volume=row.get('volume', 0)
                )
                records.append(record)
            
            self.db_session.bulk_save_objects(records)
            self.db_session.commit()
            
            print(f"[DataExporter] Inserted {len(records)} price records")
            return len(records)
        
        except Exception as e:
            self.db_session.rollback()
            print(f"[DataExporter] Error exporting price data: {e}")
            return 0
    
    def export_predictions(
        self,
        predictions: List[Dict[str, Any]],
        model_name: str,
        crypto_id: int = 1
    ) -> int:
        """
        Export predictions to database.
        
        Args:
            predictions: List of prediction dicts
            model_name: Model name
            crypto_id: Cryptocurrency ID
            
        Returns:
            Number of records inserted
        """
        if self.db_session is None:
            self.export_to_json(predictions, f'predictions_{model_name}')
            return 0
        
        try:
            records = []
            for pred in predictions:
                record = ModelPrediction(
                    model_name=model_name,
                    crypto_id=crypto_id,
                    prediction_date=pred.get('date', datetime.now().date()),
                    predicted_close_price=pred.get('predicted_price', 0),
                    actual_close_price=pred.get('actual_price'),
                    prediction_confidence=pred.get('confidence', {}).get('std')
                )
                records.append(record)
            
            self.db_session.bulk_save_objects(records)
            self.db_session.commit()
            
            print(f"[DataExporter] Inserted {len(records)} prediction records")
            return len(records)
        
        except Exception as e:
            self.db_session.rollback()
            print(f"[DataExporter] Error exporting predictions: {e}")
            return 0
    
    def export_metrics(
        self,
        metrics: Dict[str, Any],
        model_name: str,
        crypto_id: int = 1
    ) -> bool:
        """
        Export model metrics to database.
        
        Args:
            metrics: Metrics dictionary
            model_name: Model name
            crypto_id: Cryptocurrency ID
            
        Returns:
            Success status
        """
        if self.db_session is None:
            self.export_to_json(metrics, f'metrics_{model_name}')
            return False
        
        try: 
            
            record = ModelPerformanceMetric(
                model_name=model_name,
                crypto_id=crypto_id,
                evaluation_date=datetime.now().date(),
                dataset_type=metrics.get('dataset_type', 'test'),
                mae=metrics.get('mae'),
                rmse=metrics.get('rmse'),
                mape=metrics.get('mape'),
                r2_score=metrics.get('r2'),
                total_predictions=metrics.get('n_samples')
            )
            
            self.db_session.add(record)
            self.db_session.commit()
            
            print(f"[DataExporter] Inserted metrics for {model_name}")
            return True
        
        except Exception as e:
            self.db_session.rollback()
            print(f"[DataExporter] Error exporting metrics: {e}")
            return False
    
    def export_sentiment_data(
        self,
        df: pd.DataFrame,
        crypto_id: int = 1,
        platform_id: int = 1
    ) -> int:
        """
        Export sentiment aggregation to database.
        
        Args:
            df: Sentiment data DataFrame
            crypto_id: Cryptocurrency ID
            platform_id: Platform ID
            
        Returns:
            Number of records inserted
        """
        if self.db_session is None:
            self.export_to_csv(df, 'sentiment_data')
            return 0
        
        try:
            
            records = []
            for idx, row in df.iterrows():
                record = DailySentimentAggregation(
                    crypto_id=crypto_id,
                    platform_id=platform_id,
                    date_only=idx if isinstance(idx, datetime) else datetime.now().date(),
                    total_posts=row.get('Discussion_Volume', 0),
                    net_sentiment_score=row.get('Net_Sentiment_Score', 0),
                    total_likes=row.get('Engagement', 0)
                )
                records.append(record)
            
            self.db_session.bulk_save_objects(records)
            self.db_session.commit()
            
            print(f"[DataExporter] Inserted {len(records)} sentiment records")
            return len(records)
        
        except Exception as e:
            self.db_session.rollback()
            print(f"[DataExporter] Error exporting sentiment: {e}")
            return 0
    
    # ==================== BATCH EXPORT ====================
    
    def export_pipeline_results(
        self,
        price_data: pd.DataFrame,
        sentiment_data: pd.DataFrame,
        predictions: List[Dict],
        metrics: Dict[str, Any],
        model_name: str
    ) -> Dict[str, Any]:
        """
        Export all pipeline results.
        
        Args:
            price_data: Processed price data
            sentiment_data: Processed sentiment data
            predictions: Model predictions
            metrics: Model metrics
            model_name: Model name
            
        Returns:
            Export summary
        """
        summary = {
            'export_timestamp': datetime.now().isoformat(),
            'model_name': model_name,
            'exports': {}
        }
        
        # Export price data
        n_price = self.export_price_data(price_data)
        summary['exports']['price_records'] = n_price
        
        # Export sentiment data
        if sentiment_data is not None and not sentiment_data.empty:
            n_sentiment = self.export_sentiment_data(sentiment_data)
            summary['exports']['sentiment_records'] = n_sentiment
        
        # Export predictions
        n_pred = self.export_predictions(predictions, model_name)
        summary['exports']['prediction_records'] = n_pred
        
        # Export metrics
        success = self.export_metrics(metrics, model_name)
        summary['exports']['metrics_exported'] = success
        
        # Save export summary
        self.export_to_json(summary, 'export_summary', include_timestamp=True)
        
        return summary
    
    def close(self):
        """Close database connection."""
        if self.db_session is not None:
            self.db_session.close()
            print("[DataExporter] Database connection closed")


if __name__ == "__main__":
    # Test the exporter
    print("Testing Data Exporter...")
    
    exporter = DataExporter(output_dir='data/test_export')
    
    # Create sample data
    dates = pd.date_range(start='2024-01-01', periods=10, freq='D')
    sample_df = pd.DataFrame({
        'open': np.random.uniform(40000, 45000, 10),
        'high': np.random.uniform(41000, 46000, 10),
        'low': np.random.uniform(39000, 44000, 10),
        'close': np.random.uniform(40000, 45000, 10),
        'volume': np.random.uniform(1e9, 5e9, 10)
    }, index=dates)
    
    # Test CSV export
    csv_path = exporter.export_to_csv(sample_df, 'test_prices')
    
    # Test JSON export
    json_path = exporter.export_to_json({'test': 'data'}, 'test_json')
    
    print("✅ Data exporter test passed!")
