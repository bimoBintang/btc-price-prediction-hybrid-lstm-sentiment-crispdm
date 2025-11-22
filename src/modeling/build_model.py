# src/modeling/build_model.py
from tensorflow import keras

def build_lstm_model(input_shape):
    """
    Membangun model LSTM untuk prediksi harga Bitcoin.
    
    Args:
        input_shape: Tuple (timesteps, features) untuk input layer
        
    Returns:
        Model Keras yang sudah di-compile
    """
    model = keras.Sequential([
        keras.layers.LSTM(50, return_sequences=True, input_shape=input_shape),
        keras.layers.Dropout(0.2),
        keras.layers.LSTM(50, return_sequences=False),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(25, activation='relu'),
        keras.layers.Dense(1)
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']  # Tambahan: Mean Absolute Error untuk monitoring
    )
    
    return model