"""
Apache Airflow DAG for BTC Price Prediction Pipeline.

This DAG orchestrates the complete ML pipeline:
1. Data Collection (Twitter, Reddit, CoinGecko)
2. Data Processing (Cleaning, Feature Engineering)
3. Model Training (LSTM, Transformer, Ensemble)
4. Model Evaluation
5. Prediction Generation
6. Data Export to Database

Schedule: Daily at 00:00 UTC
"""
from datetime import datetime, timedelta
from typing import Dict, Any
import os
import sys

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

# Try to import Airflow components
try:
    from airflow import DAG
    from airflow.operators.python import PythonOperator
    from airflow.operators.dummy import DummyOperator
    from airflow.utils.task_group import TaskGroup
    AIRFLOW_AVAILABLE = True
except ImportError:
    AIRFLOW_AVAILABLE = False
    print("[Warning] Apache Airflow not installed. Running in standalone mode.")


# ==================== TASK FUNCTIONS ====================

def task_collect_price_data(**context) -> Dict[str, Any]:
    """Collect price data from CoinGecko."""
    from pipeline.data_collection import CoinGeckoClient
    
    client = CoinGeckoClient()
    
    # Get historical data
    historical = client.get_historical_prices(days=365)
    market_chart = client.get_market_chart(days=30)
    current = client.get_current_price()
    
    # Push to XCom for downstream tasks
    context['ti'].xcom_push(key='price_data', value={
        'historical_shape': historical.shape if not historical.empty else (0, 0),
        'current_price': current.get('price', 0),
        'collected_at': datetime.now().isoformat()
    })
    
    # Save to temp file for data processor
    historical.to_pickle('/tmp/btc_historical_prices.pkl')
    
    return {'status': 'success', 'records': len(historical)}


def task_collect_sentiment_data(**context) -> Dict[str, Any]:
    """Collect sentiment data from social media."""
    from pipeline.data_collection import DataCollector
    
    collector = DataCollector()
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    sentiment_df = collector.collect_sentiment_data(
        start_date=start_date,
        end_date=end_date
    )
    
    # Save to temp file
    if not sentiment_df.empty:
        sentiment_df.to_pickle('/tmp/btc_sentiment_data.pkl')
    
    context['ti'].xcom_push(key='sentiment_stats', value={
        'records': len(sentiment_df) if not sentiment_df.empty else 0,
        'collected_at': datetime.now().isoformat()
    })
    
    return {'status': 'success', 'records': len(sentiment_df) if not sentiment_df.empty else 0}


def task_process_data(**context) -> Dict[str, Any]:
    """Process and engineer features."""
    import pandas as pd
    from pipeline.data_processing import DataProcessor
    
    processor = DataProcessor()
    
    # Load collected data
    price_data = pd.read_pickle('/tmp/btc_historical_prices.pkl')
    
    try:
        sentiment_data = pd.read_pickle('/tmp/btc_sentiment_data.pkl')
    except:
        sentiment_data = None
    
    # Run processing pipeline
    processed = processor.process_full_pipeline(
        price_data=price_data,
        sentiment_data=sentiment_data,
        add_targets=True
    )
    
    # Save processed data
    processed.to_pickle('/tmp/btc_processed_features.pkl')
    
    context['ti'].xcom_push(key='processed_stats', value={
        'shape': processed.shape,
        'features': list(processed.columns),
        'processed_at': datetime.now().isoformat()
    })
    
    return {'status': 'success', 'shape': processed.shape}


def task_train_models(**context) -> Dict[str, Any]:
    """Train all models."""
    import pandas as pd
    import numpy as np
    import torch
    from pipeline.models import ModelFactory
    from pipeline.models.lstm_gru_model import create_sequences, create_data_loaders, LSTMGRUTrainer
    from pipeline.models.transformer_model import TransformerTrainer
    from pipeline.models.ensemble_model import EnsembleTrainer
    
    # Load processed data
    processed = pd.read_pickle('/tmp/btc_processed_features.pkl')
    
    # Prepare features
    feature_cols = [c for c in processed.columns if c not in ['target_price', 'target_change', 'target_pct_change', 'target_direction']]
    X = processed[feature_cols].values
    y = processed['target_price'].values.reshape(-1, 1)
    
    # Create sequences
    seq_length = 30
    X_seq, y_seq = create_sequences(X, y, seq_length)
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(X_seq, y_seq, batch_size=32)
    
    # Initialize factory
    factory = ModelFactory(models_dir='models')
    input_size = X_seq.shape[2]
    
    training_results = {}
    
    # Train LSTM
    print("Training LSTM...")
    lstm_model = factory.create_model('lstm', input_size)
    lstm_trainer = LSTMGRUTrainer(lstm_model)
    lstm_history = lstm_trainer.train(train_loader, val_loader, epochs=50, save_path='models/lstm_best.pth')
    training_results['lstm'] = {'final_val_loss': lstm_history['val_loss'][-1]}
    
    # Train GRU
    print("Training GRU...")
    gru_model = factory.create_model('gru', input_size)
    gru_trainer = LSTMGRUTrainer(gru_model)
    gru_history = gru_trainer.train(train_loader, val_loader, epochs=50, save_path='models/gru_best.pth')
    training_results['gru'] = {'final_val_loss': gru_history['val_loss'][-1]}
    
    # Train Transformer
    print("Training Transformer...")
    transformer_model = factory.create_model('transformer', input_size)
    transformer_trainer = TransformerTrainer(transformer_model)
    transformer_history = transformer_trainer.train(train_loader, val_loader, epochs=50, save_path='models/transformer_best.pth')
    training_results['transformer'] = {'final_val_loss': transformer_history['val_loss'][-1]}
    
    # Train Ensemble
    print("Training Ensemble...")
    ensemble_model = factory.create_model('ensemble', input_size)
    ensemble_trainer = EnsembleTrainer(ensemble_model)
    ensemble_history = ensemble_trainer.train(train_loader, val_loader, epochs=50, save_path='models/ensemble_best.pth')
    training_results['ensemble'] = {'final_val_loss': ensemble_history['val_loss'][-1]}
    
    context['ti'].xcom_push(key='training_results', value=training_results)
    
    return {'status': 'success', 'models_trained': 4}


def task_evaluate_models(**context) -> Dict[str, Any]:
    """Evaluate and compare all models."""
    import pandas as pd
    import numpy as np
    import torch
    from pipeline.models import ModelFactory
    from pipeline.models.lstm_gru_model import create_sequences
    from pipeline.evaluation import ModelEvaluator
    
    # Load test data
    processed = pd.read_pickle('/tmp/btc_processed_features.pkl')
    
    feature_cols = [c for c in processed.columns if c not in ['target_price', 'target_change', 'target_pct_change', 'target_direction']]
    X = processed[feature_cols].values
    y_true = processed['target_price'].values
    
    # Create sequences
    seq_length = 30
    X_seq, y_seq = create_sequences(X, y_true.reshape(-1, 1), seq_length)
    
    # Use last 20% as test set
    test_idx = int(len(X_seq) * 0.8)
    X_test = X_seq[test_idx:]
    y_test = y_seq[test_idx:]
    
    # Load models and get predictions
    factory = ModelFactory(models_dir='models')
    input_size = X_seq.shape[2]
    
    predictions = {}
    for model_name in ['lstm', 'gru', 'transformer', 'ensemble']:
        try:
            model = factory.load_model(f'{model_name}_best', model_name, input_size)
            pred = model.predict(X_test)
            predictions[model_name] = pred.flatten()
        except Exception as e:
            print(f"Error loading {model_name}: {e}")
    
    # Evaluate
    evaluator = ModelEvaluator(results_dir='results')
    results = evaluator.evaluate_all_models(predictions, y_test.flatten())
    
    # Get best model
    best_model, best_metrics = evaluator.get_best_model(results)
    
    # Save results
    evaluator.save_results(results)
    evaluator.generate_report(results, best_model)
    
    context['ti'].xcom_push(key='best_model', value=best_model)
    context['ti'].xcom_push(key='evaluation_results', value=results.to_dict())
    
    return {'best_model': best_model, 'rmse': best_metrics['rmse']}


def task_generate_predictions(**context) -> Dict[str, Any]:
    """Generate predictions using best model."""
    import pandas as pd
    import numpy as np
    from pipeline.prediction import PredictionService
    from pipeline.models.lstm_gru_model import create_sequences
    
    # Get best model from XCom
    best_model = context['ti'].xcom_pull(key='best_model')
    
    # Load data
    processed = pd.read_pickle('/tmp/btc_processed_features.pkl')
    
    feature_cols = [c for c in processed.columns if c not in ['target_price', 'target_change', 'target_pct_change', 'target_direction']]
    X = processed[feature_cols].values
    
    # Get latest sequence
    seq_length = 30
    latest_features = X[-seq_length:]
    
    # Initialize prediction service
    service = PredictionService(best_model_name=f'{best_model}_best')
    
    # Generate prediction
    prediction = service.predict_next_day(latest_features)
    
    # Save prediction
    service.save_prediction(prediction)
    
    context['ti'].xcom_push(key='prediction', value=prediction)
    
    return prediction


def task_export_to_database(**context) -> Dict[str, Any]:
    """Export results to database."""
    import pandas as pd
    import os
    from pipeline.storage import DataExporter
    
    db_url = os.getenv('DATABASE_URL')
    exporter = DataExporter(db_connection_string=db_url)
    
    try:
        # Export processed data
        processed = pd.read_pickle('/tmp/btc_processed_features.pkl')
        exporter.export_price_data(processed)
        
        # Export predictions
        prediction = context['ti'].xcom_pull(key='prediction')
        if prediction:
            exporter.export_predictions([prediction], 
                model_name=context['ti'].xcom_pull(key='best_model'))
        
        # Export metrics
        eval_results = context['ti'].xcom_pull(key='evaluation_results')
        if eval_results:
            for model_name, metrics in eval_results.items():
                exporter.export_metrics(metrics, model_name)
        
        exporter.close()
        return {'status': 'success'}
    
    except Exception as e:
        return {'status': 'error', 'message': str(e)}


def task_cleanup(**context) -> Dict[str, Any]:
    """Cleanup temporary files."""
    import os
    
    temp_files = [
        '/tmp/btc_historical_prices.pkl',
        '/tmp/btc_sentiment_data.pkl',
        '/tmp/btc_processed_features.pkl'
    ]
    
    for f in temp_files:
        if os.path.exists(f):
            os.remove(f)
    
    return {'status': 'cleaned', 'files_removed': len(temp_files)}


# ==================== DAG DEFINITION ====================

if AIRFLOW_AVAILABLE:
    # Default arguments
    default_args = {
        'owner': 'btc-pipeline',
        'depends_on_past': False,
        'email_on_failure': False,
        'email_on_retry': False,
        'retries': 1,
        'retry_delay': timedelta(minutes=5),
    }
    
    # Create DAG
    with DAG(
        dag_id='btc_price_prediction_pipeline',
        default_args=default_args,
        description='BTC Price Prediction ML Pipeline',
        schedule_interval='0 0 * * *',  # Daily at midnight
        start_date=datetime(2024, 1, 1),
        catchup=False,
        tags=['ml', 'btc', 'prediction']
    ) as dag:
        
        # Start
        start = DummyOperator(task_id='start')
        
        # Data Collection Group
        with TaskGroup(group_id='data_collection') as data_collection:
            collect_price = PythonOperator(
                task_id='collect_price_data',
                python_callable=task_collect_price_data
            )
            
            collect_sentiment = PythonOperator(
                task_id='collect_sentiment_data',
                python_callable=task_collect_sentiment_data
            )
        
        # Data Processing
        process_data = PythonOperator(
            task_id='process_data',
            python_callable=task_process_data
        )
        
        # Model Training
        train_models = PythonOperator(
            task_id='train_models',
            python_callable=task_train_models
        )
        
        # Model Evaluation
        evaluate_models = PythonOperator(
            task_id='evaluate_models',
            python_callable=task_evaluate_models
        )
        
        # Prediction Generation
        generate_predictions = PythonOperator(
            task_id='generate_predictions',
            python_callable=task_generate_predictions
        )
        
        # Export to Database
        export_db = PythonOperator(
            task_id='export_to_database',
            python_callable=task_export_to_database
        )
        
        # Cleanup
        cleanup = PythonOperator(
            task_id='cleanup',
            python_callable=task_cleanup
        )
        
        # End
        end = DummyOperator(task_id='end')
        
        # Define dependencies
        start >> data_collection >> process_data >> train_models >> evaluate_models >> generate_predictions >> export_db >> cleanup >> end


# ==================== STANDALONE EXECUTION ====================

def run_pipeline_standalone():
    """Run the pipeline without Airflow (for testing/development)."""
    print("="*60)
    print("BTC PRICE PREDICTION PIPELINE - Standalone Mode")
    print("="*60)
    
    class MockContext:
        """Mock Airflow context for standalone execution."""
        def __init__(self):
            self.xcom_data = {}
        
        class TaskInstance:
            def __init__(self, parent):
                self.parent = parent
            
            def xcom_push(self, key, value):
                self.parent.xcom_data[key] = value
            
            def xcom_pull(self, key):
                return self.parent.xcom_data.get(key)
        
        @property
        def ti(self):
            return self.TaskInstance(self)
    
    context = MockContext()
    
    try:
        print("\n[1/7] Collecting price data...")
        result = task_collect_price_data(**{'ti': context.ti})
        print(f"      Result: {result}")
        
        print("\n[2/7] Collecting sentiment data...")
        result = task_collect_sentiment_data(**{'ti': context.ti})
        print(f"      Result: {result}")
        
        print("\n[3/7] Processing data...")
        result = task_process_data(**{'ti': context.ti})
        print(f"      Result: {result}")
        
        print("\n[4/7] Training models...")
        result = task_train_models(**{'ti': context.ti})
        print(f"      Result: {result}")
        
        print("\n[5/7] Evaluating models...")
        result = task_evaluate_models(**{'ti': context.ti})
        print(f"      Result: {result}")
        
        print("\n[6/7] Generating predictions...")
        result = task_generate_predictions(**{'ti': context.ti})
        print(f"      Result: {result}")
        
        print("\n[7/7] Exporting to database...")
        result = task_export_to_database(**{'ti': context.ti})
        print(f"      Result: {result}")
        
        print("\n" + "="*60)
        print("✅ Pipeline completed successfully!")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        raise


if __name__ == "__main__":
    if not AIRFLOW_AVAILABLE:
        print("Running in standalone mode (Airflow not installed)")
    run_pipeline_standalone()
