"""
Model Factory for creating and managing prediction models.
"""
import torch
import os
from typing import Dict, Any, Optional, Union, Type
import json

from .lstm_gru_model import LSTMGRUModel
from .transformer_model import TransformerPriceModel
from .ensemble_model import EnsembleModel


class ModelFactory:
    """
    Factory class for creating and managing ML models.
    
    Features:
    - Create models with standard configurations
    - Load/save models
    - Model registry
    """
    
    # Default configurations for each model type
    DEFAULT_CONFIGS = {
        'lstm': {
            'hidden_size': 128,
            'num_layers': 2,
            'dropout': 0.2,
            'cell_type': 'lstm',
            'use_attention': True
        },
        'gru': {
            'hidden_size': 128,
            'num_layers': 2,
            'dropout': 0.2,
            'cell_type': 'gru',
            'use_attention': True
        },
        'transformer': {
            'd_model': 128,
            'nhead': 8,
            'num_layers': 4,
            'dim_feedforward': 512,
            'dropout': 0.1
        },
        'ensemble': {
            'hidden_size': 128,
            'ensemble_method': 'weighted',
            'include_lstm': True,
            'include_gru': True,
            'include_transformer': True
        }
    }
    
    # Model class registry
    MODEL_CLASSES = {
        'lstm': LSTMGRUModel,
        'gru': LSTMGRUModel,
        'transformer': TransformerPriceModel,
        'ensemble': EnsembleModel
    }
    
    def __init__(self, models_dir: str = 'models'):
        """
        Initialize model factory.
        
        Args:
            models_dir: Directory for saving/loading models
        """
        self.models_dir = models_dir
        os.makedirs(models_dir, exist_ok=True)
        
        self._model_cache: Dict[str, torch.nn.Module] = {}
    
    def create_model(
        self,
        model_type: str,
        input_size: int,
        output_size: int = 1,
        config: Optional[Dict[str, Any]] = None
    ) -> torch.nn.Module:
        """
        Create a model of the specified type.
        
        Args:
            model_type: Type of model ('lstm', 'gru', 'transformer', 'ensemble')
            input_size: Number of input features
            output_size: Number of output values
            config: Optional custom configuration
            
        Returns:
            Instantiated model
        """
        model_type = model_type.lower()
        
        if model_type not in self.MODEL_CLASSES:
            raise ValueError(f"Unknown model type: {model_type}. Available: {list(self.MODEL_CLASSES.keys())}")
        
        # Merge default and custom configs
        model_config = self.DEFAULT_CONFIGS.get(model_type, {}).copy()
        if config:
            model_config.update(config)
        
        # Create model
        model_class = self.MODEL_CLASSES[model_type]
        model_config['input_size'] = input_size
        model_config['output_size'] = output_size
        
        model = model_class(**model_config)
        
        print(f"[ModelFactory] Created {model_type} model with {sum(p.numel() for p in model.parameters()):,} parameters")
        
        return model
    
    def create_all_models(
        self,
        input_size: int,
        output_size: int = 1
    ) -> Dict[str, torch.nn.Module]:
        """
        Create all available model types.
        
        Args:
            input_size: Number of input features
            output_size: Number of output values
            
        Returns:
            Dict of model_type -> model
        """
        models = {}
        
        for model_type in ['lstm', 'gru', 'transformer', 'ensemble']:
            try:
                models[model_type] = self.create_model(
                    model_type=model_type,
                    input_size=input_size,
                    output_size=output_size
                )
            except Exception as e:
                print(f"[ModelFactory] Warning: Failed to create {model_type}: {e}")
        
        return models
    
    def save_model(
        self,
        model: torch.nn.Module,
        model_name: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Save model to disk.
        
        Args:
            model: Model to save
            model_name: Name for the saved model
            metadata: Optional metadata to save with model
            
        Returns:
            Path to saved model
        """
        model_path = os.path.join(self.models_dir, f"{model_name}.pth")
        
        save_dict = {
            'model_state_dict': model.state_dict(),
            'model_class': model.__class__.__name__,
            'input_size': model.input_size if hasattr(model, 'input_size') else None,
        }
        
        if metadata:
            save_dict['metadata'] = metadata
        
        torch.save(save_dict, model_path)
        
        print(f"[ModelFactory] Model saved to {model_path}")
        return model_path
    
    def load_model(
        self,
        model_name: str,
        model_type: Optional[str] = None,
        input_size: Optional[int] = None
    ) -> torch.nn.Module:
        """
        Load model from disk.
        
        Args:
            model_name: Name of saved model
            model_type: Type of model (for creating new instance)
            input_size: Input size for the model
            
        Returns:
            Loaded model
        """
        model_path = os.path.join(self.models_dir, f"{model_name}.pth")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # Determine model type
        if model_type is None:
            class_name = checkpoint.get('model_class', '')
            if 'LSTM' in class_name or 'GRU' in class_name:
                model_type = 'lstm'
            elif 'Transformer' in class_name:
                model_type = 'transformer'
            elif 'Ensemble' in class_name:
                model_type = 'ensemble'
            else:
                raise ValueError("Cannot determine model type. Please specify model_type.")
        
        # Determine input size
        if input_size is None:
            input_size = checkpoint.get('input_size')
            if input_size is None:
                raise ValueError("Cannot determine input_size. Please specify.")
        
        # Create model and load weights
        model = self.create_model(model_type, input_size)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        print(f"[ModelFactory] Model loaded from {model_path}")
        return model
    
    def get_best_model(
        self,
        metrics: Dict[str, Dict[str, float]],
        metric_name: str = 'val_loss'
    ) -> str:
        """
        Determine the best model based on metrics.
        
        Args:
            metrics: Dict of model_name -> {metric_name: value}
            metric_name: Metric to use for comparison (lower is better)
            
        Returns:
            Name of the best model
        """
        if not metrics:
            raise ValueError("No metrics provided")
        
        best_model = None
        best_value = float('inf')
        
        for model_name, model_metrics in metrics.items():
            value = model_metrics.get(metric_name, float('inf'))
            if value < best_value:
                best_value = value
                best_model = model_name
        
        print(f"[ModelFactory] Best model: {best_model} ({metric_name}={best_value:.6f})")
        return best_model
    
    def list_saved_models(self) -> list:
        """List all saved models."""
        models = []
        for f in os.listdir(self.models_dir):
            if f.endswith('.pth'):
                models.append(f[:-4])
        return models


if __name__ == "__main__":
    # Test the factory
    print("Testing Model Factory...")
    
    factory = ModelFactory(models_dir='models')
    
    # Create all model types
    input_size = 10
    
    for model_type in ['lstm', 'gru', 'transformer', 'ensemble']:
        print(f"\n--- Creating {model_type} ---")
        model = factory.create_model(
            model_type=model_type,
            input_size=input_size
        )
        print(f"Model type: {type(model).__name__}")
    
    print("\n✅ Factory test passed!")
