"""
Model Evaluator for comparing and selecting the best model.
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
import json
import os


class ModelEvaluator:
    """
    Evaluator for comparing model performance and selecting the best model.
    
    Metrics:
    - MAE (Mean Absolute Error)
    - RMSE (Root Mean Square Error)
    - MAPE (Mean Absolute Percentage Error)
    - R² Score
    - Directional Accuracy
    """
    
    def __init__(self, results_dir: str = 'results'):
        """
        Initialize evaluator.
        
        Args:
            results_dir: Directory to save evaluation results
        """
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
        
        self.evaluation_history: Dict[str, List[Dict]] = {}
    
    @staticmethod
    def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Mean Absolute Error."""
        return float(np.mean(np.abs(y_true - y_pred)))
    
    @staticmethod
    def root_mean_square_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Root Mean Square Error."""
        return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    
    @staticmethod
    def mean_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Mean Absolute Percentage Error."""
        # Avoid division by zero
        mask = y_true != 0
        if not mask.any():
            return 0.0
        return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)
    
    @staticmethod
    def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate R² Score."""
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        if ss_tot == 0:
            return 0.0
        return float(1 - (ss_res / ss_tot))
    
    @staticmethod
    def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Directional Accuracy (% of correct direction predictions)."""
        if len(y_true) < 2:
            return 0.0
        
        # Calculate actual and predicted directions
        actual_direction = np.sign(np.diff(y_true.flatten()))
        pred_direction = np.sign(np.diff(y_pred.flatten()))
        
        # Calculate accuracy
        correct = np.sum(actual_direction == pred_direction)
        return float(correct / len(actual_direction) * 100)
    
    def evaluate_model(
        self,
        model_name: str,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        dataset_type: str = 'test'
    ) -> Dict[str, float]:
        """
        Evaluate a single model.
        
        Args:
            model_name: Name of the model
            y_true: True values
            y_pred: Predicted values
            dataset_type: 'train', 'val', or 'test'
            
        Returns:
            Dict of metrics
        """
        y_true = np.array(y_true).flatten()
        y_pred = np.array(y_pred).flatten()
        
        metrics = {
            'model_name': model_name,
            'dataset_type': dataset_type,
            'mae': self.mean_absolute_error(y_true, y_pred),
            'rmse': self.root_mean_square_error(y_true, y_pred),
            'mape': self.mean_absolute_percentage_error(y_true, y_pred),
            'r2': self.r2_score(y_true, y_pred),
            'directional_accuracy': self.directional_accuracy(y_true, y_pred),
            'n_samples': len(y_true),
            'evaluated_at': datetime.now().isoformat()
        }
        
        # Store in history
        if model_name not in self.evaluation_history:
            self.evaluation_history[model_name] = []
        self.evaluation_history[model_name].append(metrics)
        
        return metrics
    
    def evaluate_all_models(
        self,
        predictions: Dict[str, np.ndarray],
        y_true: np.ndarray,
        dataset_type: str = 'test'
    ) -> pd.DataFrame:
        """
        Evaluate multiple models.
        
        Args:
            predictions: Dict mapping model_name to predictions
            y_true: True values
            dataset_type: Dataset type
            
        Returns:
            DataFrame with all metrics
        """
        results = []
        
        for model_name, y_pred in predictions.items():
            metrics = self.evaluate_model(
                model_name=model_name,
                y_true=y_true,
                y_pred=y_pred,
                dataset_type=dataset_type
            )
            results.append(metrics)
        
        df = pd.DataFrame(results)
        
        # Sort by RMSE (lower is better)
        df = df.sort_values('rmse')
        
        return df
    
    def get_best_model(
        self,
        results: pd.DataFrame,
        metric: str = 'rmse',
        lower_is_better: bool = True
    ) -> Tuple[str, Dict[str, float]]:
        """
        Get the best model based on a metric.
        
        Args:
            results: Evaluation results DataFrame
            metric: Metric to use for comparison
            lower_is_better: Whether lower values are better
            
        Returns:
            Tuple of (best_model_name, metrics_dict)
        """
        if results.empty:
            raise ValueError("No results to evaluate")
        
        if lower_is_better:
            best_idx = results[metric].idxmin()
        else:
            best_idx = results[metric].idxmax()
        
        best_row = results.loc[best_idx]
        best_model = best_row['model_name']
        
        print(f"\n{'='*50}")
        print(f"🏆 Best Model: {best_model}")
        print(f"   {metric}: {best_row[metric]:.6f}")
        print(f"   MAE: {best_row['mae']:.4f}")
        print(f"   RMSE: {best_row['rmse']:.4f}")
        print(f"   MAPE: {best_row['mape']:.2f}%")
        print(f"   R²: {best_row['r2']:.4f}")
        print(f"   Directional Accuracy: {best_row['directional_accuracy']:.2f}%")
        print(f"{'='*50}\n")
        
        return best_model, best_row.to_dict()
    
    def compare_models(self, results: pd.DataFrame) -> None:
        """
        Print a comparison table of all models.
        
        Args:
            results: Evaluation results DataFrame
        """
        print("\n" + "="*80)
        print("MODEL COMPARISON")
        print("="*80)
        
        # Format table
        display_cols = ['model_name', 'mae', 'rmse', 'mape', 'r2', 'directional_accuracy']
        available_cols = [c for c in display_cols if c in results.columns]
        
        for _, row in results[available_cols].iterrows():
            print(f"\n📊 {row['model_name']}")
            print(f"   MAE:  {row['mae']:.4f}")
            print(f"   RMSE: {row['rmse']:.4f}")
            print(f"   MAPE: {row['mape']:.2f}%")
            print(f"   R²:   {row['r2']:.4f}")
            print(f"   Dir Acc: {row['directional_accuracy']:.2f}%")
        
        print("\n" + "="*80)
    
    def save_results(
        self,
        results: pd.DataFrame,
        filename: str = 'model_evaluation'
    ) -> str:
        """
        Save evaluation results.
        
        Args:
            results: Results DataFrame
            filename: Output filename (without extension)
            
        Returns:
            Path to saved file
        """
        # Save CSV
        csv_path = os.path.join(self.results_dir, f"{filename}.csv")
        results.to_csv(csv_path, index=False)
        
        # Save JSON
        json_path = os.path.join(self.results_dir, f"{filename}.json")
        results.to_json(json_path, orient='records', indent=2)
        
        print(f"[Evaluator] Results saved to {csv_path}")
        return csv_path
    
    def generate_report(
        self,
        results: pd.DataFrame,
        best_model: str
    ) -> str:
        """
        Generate a markdown report of evaluation results.
        
        Args:
            results: Evaluation results
            best_model: Name of best model
            
        Returns:
            Markdown report string
        """
        report = f"""# Model Evaluation Report

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Summary

**Best Model**: {best_model}

## Metrics Comparison

| Model | MAE | RMSE | MAPE (%) | R² | Dir. Accuracy (%) |
|-------|-----|------|----------|------|-------------------|
"""
        for _, row in results.iterrows():
            report += f"| {row['model_name']} | {row['mae']:.4f} | {row['rmse']:.4f} | {row['mape']:.2f} | {row['r2']:.4f} | {row['directional_accuracy']:.2f} |\n"
        
        report += f"""

## Recommendations

Based on the evaluation metrics, **{best_model}** shows the best overall performance with the lowest RMSE score.

### Key Observations:
- Lower MAE/RMSE indicates better prediction accuracy
- Higher R² indicates better fit to the data
- Higher Directional Accuracy indicates better trend prediction capability
"""
        
        # Save report
        report_path = os.path.join(self.results_dir, 'evaluation_report.md')
        with open(report_path, 'w') as f:
            f.write(report)
        
        print(f"[Evaluator] Report saved to {report_path}")
        return report


if __name__ == "__main__":
    # Test the evaluator
    print("Testing Model Evaluator...")
    
    # Create sample data
    np.random.seed(42)
    n_samples = 100
    
    y_true = np.random.uniform(40000, 50000, n_samples)
    
    predictions = {
        'lstm': y_true + np.random.normal(0, 500, n_samples),
        'gru': y_true + np.random.normal(0, 600, n_samples),
        'transformer': y_true + np.random.normal(0, 450, n_samples),
        'ensemble': y_true + np.random.normal(0, 400, n_samples)
    }
    
    evaluator = ModelEvaluator()
    
    # Evaluate all models
    results = evaluator.evaluate_all_models(predictions, y_true)
    
    # Compare models
    evaluator.compare_models(results)
    
    # Get best model
    best_model, metrics = evaluator.get_best_model(results)
    
    print("✅ Evaluator test passed!")
