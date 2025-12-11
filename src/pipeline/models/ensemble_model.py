"""
Ensemble Model combining LSTM, GRU, and Transformer for BTC Price Prediction.
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
import os

from .lstm_gru_model import LSTMGRUModel
from .transformer_model import TransformerPriceModel


def get_device():
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


class EnsembleModel(nn.Module):
    """
    Ensemble model combining multiple models for price prediction.
    
    Strategies:
    - Simple averaging
    - Weighted averaging (learnable or fixed)
    - Stacking with meta-learner
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        ensemble_method: str = 'weighted',  # 'average', 'weighted', 'stacking'
        include_lstm: bool = True,
        include_gru: bool = True,
        include_transformer: bool = True,
        output_size: int = 1
    ):
        """
        Initialize ensemble model.
        
        Args:
            input_size: Number of input features
            hidden_size: Hidden size for sub-models
            ensemble_method: How to combine predictions
            include_lstm: Include LSTM model
            include_gru: Include GRU model
            include_transformer: Include Transformer model
            output_size: Output dimension
        """
        super().__init__()
        
        self.input_size = input_size
        self.ensemble_method = ensemble_method
        self.device = get_device()
        self.models = nn.ModuleDict()
        
        # Create sub-models
        if include_lstm:
            self.models['lstm'] = LSTMGRUModel(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=2,
                dropout=0.2,
                cell_type='lstm',
                use_attention=True,
                output_size=output_size
            )
        
        if include_gru:
            self.models['gru'] = LSTMGRUModel(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=2,
                dropout=0.2,
                cell_type='gru',
                use_attention=True,
                output_size=output_size
            )
        
        if include_transformer:
            self.models['transformer'] = TransformerPriceModel(
                input_size=input_size,
                d_model=hidden_size,
                nhead=4,
                num_layers=2,
                dropout=0.1,
                output_size=output_size
            )
        
        self.num_models = len(self.models)
        
        # Ensemble weights (learnable)
        if ensemble_method == 'weighted':
            self.weights = nn.Parameter(torch.ones(self.num_models) / self.num_models)
        
        # Meta-learner for stacking
        if ensemble_method == 'stacking':
            self.meta_learner = nn.Sequential(
                nn.Linear(self.num_models * output_size, hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_size // 2, output_size)
            )
        
        self.to(self.device)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through ensemble.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, input_size]
            
        Returns:
            Combined predictions of shape [batch_size, output_size]
        """
        # Get predictions from all models
        predictions = []
        for name, model in self.models.items():
            pred = model(x)
            predictions.append(pred)
        
        # Stack predictions: [num_models, batch_size, output_size]
        stacked = torch.stack(predictions, dim=0)
        
        # Combine based on ensemble method
        if self.ensemble_method == 'average':
            output = stacked.mean(dim=0)
        
        elif self.ensemble_method == 'weighted':
            # Normalize weights with softmax
            weights = torch.softmax(self.weights, dim=0)
            # Weighted average
            output = torch.sum(stacked * weights.view(-1, 1, 1), dim=0)
        
        elif self.ensemble_method == 'stacking':
            # Concatenate all predictions and feed to meta-learner
            # Shape: [batch_size, num_models * output_size]
            concat = stacked.permute(1, 0, 2).reshape(x.size(0), -1)
            output = self.meta_learner(concat)
        
        else:
            output = stacked.mean(dim=0)
        
        return output
    
    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Make predictions on numpy array.
        
        Args:
            x: Input array
            
        Returns:
            Predictions array
        """
        self.eval()
        with torch.no_grad():
            x_tensor = torch.FloatTensor(x).to(self.device)
            predictions = self(x_tensor)
            return predictions.cpu().numpy()
    
    def get_individual_predictions(self, x: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Get predictions from each individual model.
        
        Args:
            x: Input array
            
        Returns:
            Dict mapping model name to predictions
        """
        self.eval()
        predictions = {}
        
        with torch.no_grad():
            x_tensor = torch.FloatTensor(x).to(self.device)
            for name, model in self.models.items():
                pred = model(x_tensor)
                predictions[name] = pred.cpu().numpy()
        
        return predictions
    
    def get_ensemble_weights(self) -> Dict[str, float]:
        """Get current ensemble weights."""
        if self.ensemble_method != 'weighted':
            return {name: 1.0 / self.num_models for name in self.models.keys()}
        
        weights = torch.softmax(self.weights, dim=0).detach().cpu().numpy()
        return {name: float(weights[i]) for i, name in enumerate(self.models.keys())}


class EnsembleTrainer:
    """Trainer for ensemble models."""
    
    def __init__(
        self,
        model: EnsembleModel,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-5
    ):
        """Initialize trainer."""
        self.model = model
        self.device = model.device
        
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        self.criterion = nn.MSELoss()
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10, verbose=True
        )
        
        self.history = {'train_loss': [], 'val_loss': []}
    
    def train_epoch(self, train_loader) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            
            self.optimizer.zero_grad()
            predictions = self.model(batch_x)
            loss = self.criterion(predictions, batch_y)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            total_loss += loss.item()
        
        return total_loss / len(train_loader)
    
    def validate(self, val_loader) -> float:
        """Validate model."""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                predictions = self.model(batch_x)
                loss = self.criterion(predictions, batch_y)
                total_loss += loss.item()
        
        return total_loss / len(val_loader)
    
    def train(
        self,
        train_loader,
        val_loader,
        epochs: int = 100,
        early_stopping_patience: int = 15,
        save_best: bool = True,
        save_path: str = 'models/ensemble_best.pth'
    ) -> dict:
        """Full training loop."""
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)
            
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            
            self.scheduler.step(val_loss)
            
            # Print weights if using weighted ensemble
            weights_str = ""
            if self.model.ensemble_method == 'weighted':
                weights = self.model.get_ensemble_weights()
                weights_str = f" | Weights: {weights}"
            
            print(f"Epoch {epoch+1}/{epochs} - Train: {train_loss:.6f}, Val: {val_loss:.6f}{weights_str}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                if save_best:
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    torch.save(self.model.state_dict(), save_path)
                    print(f"  → Best model saved")
            else:
                patience_counter += 1
            
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        return self.history


if __name__ == "__main__":
    # Test the ensemble model
    print("Testing Ensemble Model...")
    
    seq_length = 30
    input_size = 10
    batch_size = 16
    
    # Random sample data
    X = np.random.randn(100, seq_length, input_size).astype(np.float32)
    
    # Create model
    model = EnsembleModel(
        input_size=input_size,
        hidden_size=64,
        ensemble_method='weighted',
        include_lstm=True,
        include_gru=True,
        include_transformer=True
    )
    
    print(f"Model device: {model.device}")
    print(f"Number of sub-models: {model.num_models}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test forward pass
    predictions = model.predict(X[:batch_size])
    print(f"Ensemble predictions shape: {predictions.shape}")
    
    # Test individual predictions
    individual = model.get_individual_predictions(X[:batch_size])
    for name, pred in individual.items():
        print(f"  - {name}: {pred.shape}")
    
    print(f"Ensemble weights: {model.get_ensemble_weights()}")
    print("✅ Test passed!")
