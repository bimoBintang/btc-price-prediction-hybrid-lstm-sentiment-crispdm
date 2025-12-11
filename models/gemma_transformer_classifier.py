import hashlib
from typing import Iterable, Optional

import torch
from torch import Tensor, nn
from sentence_transformers import SentenceTransformer

# Define a function to automatically get the best available device
def get_best_device():
    # Check for CUDA (NVIDIA GPU) availability
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    # Check for MPS (Apple Silicon GPU) availability
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        print("Using Apple Silicon MPS device")
    # Fallback to CPU
    else:
        device = torch.device('cpu')
        print("Using CPU")
    
    return device


# Model embedding yang tersedia
EMBEDDING_MODELS = {
    "minilm": "all-MiniLM-L6-v2",           # Cepat, tanpa auth (22MB)
    "mpnet": "all-mpnet-base-v2",            # Lebih akurat, tanpa auth (420MB)
    "bge": "BAAI/bge-small-en-v1.5",         # State-of-art, tanpa auth (130MB) 
    "gemma": "google/embeddinggemma-300m",  # Butuh HuggingFace auth
}


class SentimentTransformerClassifier(nn.Module):
    """
    Transformer-based Sentiment Classifier menggunakan pre-trained embeddings.
    
    Args:
        num_classes: Jumlah kelas output (default: 3 untuk positive/neutral/negative)
        hidden_dim: Dimensi hidden layer
        num_layers: Jumlah layer Transformer Encoder
        num_heads: Jumlah attention heads
        dropout: Dropout rate
        embedding_model: Nama model embedding ('minilm', 'mpnet', 'bge', atau 'gemma')
    """
    def __init__(
        self,
        num_classes: int = 3,
        hidden_dim: int = 256,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
        embedding_model: str = "mpnet",  # Default
    ) -> None:
        super().__init__()
        self.embedding_cache = {}
        self.device = get_best_device()
        
        # Pilih model embedding
        model_name = EMBEDDING_MODELS.get(embedding_model, embedding_model)
        print(f"Loading embedding model: {model_name}")
        self.embedding_model = SentenceTransformer(model_name, device=str(self.device))
        embedding_dim = self.embedding_model.get_sentence_embedding_dimension()

        # Proyeksi ke hidden_dim untuk LSTM
        self.project = nn.Linear(embedding_dim, hidden_dim, device=get_best_device()) if embedding_dim != hidden_dim else nn.Identity()
        
        # Note: batch_first=True adalah standar untuk PyTorch.
        encode_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=embedding_dim,
            dropout=dropout,
            batch_first=True,
            device=get_best_device(),
            activation='gelu',
        )
       
        self.transformer = nn.TransformerEncoder(encoder_layer=encode_layer, num_layers=num_layers)

        # Classifier: Tanh + Linear (untuk multi-class, tidak perlu Sigmoid)
        # CrossEntropyLoss akan menangani softmax secara internal
        self.classifier = nn.Sequential(
            nn.Tanh(), 
            nn.Linear(hidden_dim, num_classes, device=self.device),
        )

        self.to(self.device)

    def embedding(self, text: str) -> Tensor:
        key = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if key not in self.embedding_cache:
            # logika pooling dari model sentiment
            features = self.embedding_model.tokenize([text])
            features = {name: tensor.to(self.device) for name, tensor in features.items()}

            with torch.no_grad():
                outputs = self.embedding_model(features)

            token_embeddings = outputs["token_embeddings"]
            attention_mask = features["attention_mask"]

            mask = attention_mask.unsqueeze(-1)
            # representasi kalimat 
            pooled = (token_embeddings * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

            self.embedding_cache[key] = pooled.squeeze(0)

        return self.embedding_cache[key]

    def forward(self, texts: Iterable[str]) -> Tensor:
        """
        Forward pass menggunakan Transformer Encoder.
        
        Args:
            texts: Iterable of text strings to classify
            
        Returns:
            Logits tensor of shape [batch_size, num_classes]
        """
        # 1. Dapatkan embeddings untuk setiap teks
        embeddings = torch.stack([self.embedding(text) for text in texts])

        # 2. Proyeksi ke hidden dimension dan tambah sequence dimension
        # Shape: [batch_size, embedding_dim] -> [batch_size, 1, hidden_dim]
        hidden_input = self.project(embeddings).unsqueeze(1) 
        
        # 3. Lewatkan melalui Transformer Encoder
        # Shape: [batch_size, 1, hidden_dim] -> [batch_size, 1, hidden_dim]
        encoded = self.transformer(hidden_input)
        
        # 4. Pooling: ambil output dari sequence position pertama (dan satu-satunya)
        # Shape: [batch_size, 1, hidden_dim] -> [batch_size, hidden_dim]
        pooled = encoded.squeeze(1)

        # 5. Classifier: menghasilkan logits untuk setiap class
        return self.classifier(pooled)


# --- Fungsi Contoh ---

def example_forward() -> None:
    """Contoh forward pass untuk klasifikasi sentimen."""
    model = SentimentTransformerClassifier()
    texts = ["Buy the dip?", "Market is flat."]
    logits = model(texts)
    print("Raw Logits:", logits.detach().cpu())
    print("Probabilities (Softmax):", logits.softmax(dim=-1).detach().cpu())


def example_train(epochs: int = 2, lr: float = 1e-4) -> None:
    """
    Contoh training loop untuk klasifikasi sentimen.
    Labels: 0 = Negative/Sell, 1 = Neutral/Hold, 2 = Positive/Buy
    """
    model = SentimentTransformerClassifier()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    texts = [
        "Strong earnings beat expectations.",      # Positive
        "Regulatory concerns weigh on the sector.",  # Negative
        "Momentum indicators point to consolidation.",  # Neutral
    ]
    labels = torch.tensor([2, 0, 1], device=model.device)  # class ids: 0=sell, 1=hold, 2=buy

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        logits = model(texts)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        probs = logits.softmax(dim=-1).detach().cpu()
        print(f"Epoch {epoch + 1}: loss={loss.item():.4f} probs={probs}")


if __name__ == "__main__":
    example_forward()
    print("---")
    example_train()