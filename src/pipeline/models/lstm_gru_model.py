"""
LSTM/GRU Model for BTC Price Prediction.
"""
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, List
import os
import joblib


def get_device():
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


class LSTMGRUModel(nn.Module):
    """
    LSTM/GRU model for time-series price prediction.
    
    Features:
    - Configurable LSTM or GRU cells
    - Multi-layer support
    - Dropout for regularization
    - Attention mechanism (optional)
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        output_size: int = 1,
        dropout: float = 0.2,
        cell_type: str = 'lstm',  # 'lstm' or 'gru'
        bidirectional: bool = False,
        use_attention: bool = False
    ):
        """
        Initialize the LSTM/GRU model.
        
        Args:
            input_size: Number of input features
            hidden_size: Size of hidden state
            num_layers: Number of recurrent layers
            output_size: Number of output values (default: 1 for price)
            dropout: Dropout rate
            cell_type: 'lstm' or 'gru'
            bidirectional: Use bidirectional RNN
            use_attention: Apply attention mechanism
        """
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.cell_type = cell_type.lower()
        self.bidirectional = bidirectional
        self.use_attention = use_attention
        self.device = get_device()
        
        # Input projection
        self.input_projection = nn.Linear(input_size, hidden_size)
        
        # Recurrent layer
        rnn_class = nn.LSTM if self.cell_type == 'lstm' else nn.GRU
        self.rnn = rnn_class(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Attention layer (optional)
        if use_attention:
            attn_size = hidden_size * 2 if bidirectional else hidden_size
            self.attention = nn.Sequential(
                nn.Linear(attn_size, attn_size),
                nn.Tanh(),
                nn.Linear(attn_size, 1)
            )
        
        # Output layers
        fc_input_size = hidden_size * 2 if bidirectional else hidden_size
        self.fc = nn.Sequential(
            nn.Linear(fc_input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, output_size)
        )
        
        self.to(self.device)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, input_size]
            
        Returns:
            Output tensor of shape [batch_size, output_size]
        """
        # Input projection
        x = self.input_projection(x)
        
        # Recurrent layer
        rnn_out, _ = self.rnn(x)
        
        # Attention or last hidden state
        if self.use_attention:
            attn_weights = torch.softmax(self.attention(rnn_out), dim=1)
            context = torch.sum(attn_weights * rnn_out, dim=1)
        else:
            context = rnn_out[:, -1, :]
        
        # Output
        output = self.fc(context)
        
        return output
    
    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Make predictions on numpy array.
        
        Args:
            x: Input array of shape [batch_size, seq_len, input_size]
            
        Returns:
            Predictions array
        """
        self.eval()
        with torch.no_grad():
            x_tensor = torch.FloatTensor(x).to(self.device)
            predictions = self(x_tensor)
            return predictions.cpu().numpy()


class LSTMGRUTrainer:
    """Trainer for LSTM/GRU models."""
    
    def __init__(
        self,
        model: LSTMGRUModel,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-5
    ):
        """
        Initialize trainer.
        
        Args:
            model: LSTM/GRU model to train
            learning_rate: Learning rate
            weight_decay: L2 regularization
        """
        self.model = model
        self.device = model.device
        
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        self.criterion = nn.MSELoss()
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        
        self.history = {'train_loss': [], 'val_loss': []}
    
    def train_epoch(
        self, 
        train_loader: torch.utils.data.DataLoader
    ) -> float:
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
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            total_loss += loss.item()
        
        return total_loss / len(train_loader)
    
    def validate(
        self, 
        val_loader: torch.utils.data.DataLoader
    ) -> float:
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
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        epochs: int = 100,
        early_stopping_patience: int = 10,
        save_best: bool = True,
        save_path: str = 'models/lstm_gru_best.pth'
    ) -> Dict[str, List[float]]:
        """
        Full training loop.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of epochs
            early_stopping_patience: Early stopping patience
            save_best: Save best model
            save_path: Path to save best model
            
        Returns:
            Training history
        """
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)
            
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            
            self.scheduler.step(val_loss)
            
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                if save_best:
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    torch.save(self.model.state_dict(), save_path)
                    print(f"  → Best model saved to {save_path}")
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        return self.history


def create_sequences(
    data: np.ndarray,
    target: np.ndarray,
    seq_length: int = 30
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sequences for time-series prediction.
    
    Args:
        data: Feature data of shape [n_samples, n_features]
        target: Target data of shape [n_samples, 1]
        seq_length: Sequence length
        
    Returns:
        Tuple of (X, y) arrays
    """
    X, y = [], []
    
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(target[i + seq_length])
    
    return np.array(X), np.array(y)


def create_data_loaders(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int = 32,
    train_ratio: float = 0.8
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Create train and validation data loaders."""
    split_idx = int(len(X) * train_ratio)
    
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    train_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(X_train),
        torch.FloatTensor(y_train)
    )
    val_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(X_val),
        torch.FloatTensor(y_val)
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False
    )
    
    return train_loader, val_loader


if __name__ == "__main__":
    # Test the model
    print("Testing LSTM/GRU Model...")
    
    # Create sample data
    seq_length = 30
    input_size = 10
    batch_size = 16
    
    # Random sample data
    X = np.random.randn(100, seq_length, input_size).astype(np.float32)
    y = np.random.randn(100, 1).astype(np.float32)
    
    # Create model
    model = LSTMGRUModel(
        input_size=input_size,
        hidden_size=64,
        num_layers=2,
        cell_type='lstm',
        use_attention=True
    )
    
    print(f"Model device: {model.device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test forward pass
    predictions = model.predict(X[:batch_size])
    print(f"Predictions shape: {predictions.shape}")
    print("✅ Test passed!")
