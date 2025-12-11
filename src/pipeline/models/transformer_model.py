"""
Transformer Model for BTC Price Prediction.
"""
import torch
import torch.nn as nn
import numpy as np
import math
from typing import Optional, Tuple
import os


def get_device():
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer."""
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input."""
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerPriceModel(nn.Module):
    """
    Transformer model for time-series price prediction.
    
    Features:
    - Multi-head self-attention
    - Positional encoding
    - Configurable encoder layers
    """
    
    def __init__(
        self,
        input_size: int,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        output_size: int = 1,
        max_seq_len: int = 100
    ):
        """
        Initialize Transformer model.
        
        Args:
            input_size: Number of input features
            d_model: Model dimension
            nhead: Number of attention heads
            num_layers: Number of encoder layers
            dim_feedforward: Feedforward network dimension
            dropout: Dropout rate
            output_size: Output dimension
            max_seq_len: Maximum sequence length
        """
        super().__init__()
        
        self.input_size = input_size
        self.d_model = d_model
        self.device = get_device()
        
        # Input projection
        self.input_projection = nn.Linear(input_size, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_seq_len, dropout)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # Global attention pooling
        self.global_attention = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Tanh(),
            nn.Linear(d_model, 1)
        )
        
        # Output layers
        self.output_layer = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, output_size)
        )
        
        self.to(self.device)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(
        self, 
        x: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, input_size]
            src_mask: Optional attention mask
            
        Returns:
            Output tensor of shape [batch_size, output_size]
        """
        # Input projection and positional encoding
        x = self.input_projection(x) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        
        # Transformer encoder
        encoded = self.transformer_encoder(x, mask=src_mask)
        
        # Global attention pooling
        attn_weights = torch.softmax(self.global_attention(encoded), dim=1)
        pooled = torch.sum(attn_weights * encoded, dim=1)
        
        # Output
        output = self.output_layer(pooled)
        
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


class TransformerTrainer:
    """Trainer for Transformer models."""
    
    def __init__(
        self,
        model: TransformerPriceModel,
        learning_rate: float = 0.0001,
        weight_decay: float = 1e-5,
        warmup_steps: int = 1000
    ):
        """
        Initialize trainer.
        
        Args:
            model: Transformer model to train
            learning_rate: Learning rate
            weight_decay: L2 regularization
            warmup_steps: Warmup steps for learning rate scheduler
        """
        self.model = model
        self.device = model.device
        
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.98)
        )
        
        self.criterion = nn.HuberLoss(delta=1.0)  # More robust to outliers
        
        # Warmup + cosine scheduler
        self.warmup_steps = warmup_steps
        self.current_step = 0
        self.base_lr = learning_rate
        
        self.history = {'train_loss': [], 'val_loss': []}
    
    def _adjust_learning_rate(self):
        """Adjust learning rate with warmup."""
        if self.current_step < self.warmup_steps:
            lr = self.base_lr * (self.current_step + 1) / self.warmup_steps
        else:
            lr = self.base_lr
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
    
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
            
            self._adjust_learning_rate()
            self.current_step += 1
            
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
        early_stopping_patience: int = 15,
        save_best: bool = True,
        save_path: str = 'models/transformer_best.pth'
    ) -> dict:
        """
        Full training loop.
        
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
            
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                if save_best:
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    torch.save(self.model.state_dict(), save_path)
                    print(f"  → Best model saved to {save_path}")
            else:
                patience_counter += 1
            
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        return self.history


if __name__ == "__main__":
    # Test the model
    print("Testing Transformer Price Model...")
    
    seq_length = 30
    input_size = 10
    batch_size = 16
    
    # Random sample data
    X = np.random.randn(100, seq_length, input_size).astype(np.float32)
    
    # Create model
    model = TransformerPriceModel(
        input_size=input_size,
        d_model=64,
        nhead=4,
        num_layers=2
    )
    
    print(f"Model device: {model.device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test forward pass
    predictions = model.predict(X[:batch_size])
    print(f"Predictions shape: {predictions.shape}")
    print("✅ Test passed!")
