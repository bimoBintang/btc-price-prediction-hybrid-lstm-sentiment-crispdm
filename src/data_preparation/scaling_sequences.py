# src/data_preparation/scaling_sequences.py
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import joblib
from config.settings import SCALERS_DIR

def prepare_sequences(data, feature_columns):
    print("Scaling & Sequence Creation (TIME_STEPS=60)...")
    
    X = data[feature_columns].values
    y = data['Close'].values.reshape(-1, 1)
    
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y)
    
    # Save scalers
    joblib.dump(scaler_X, SCALERS_DIR / "scaler_X.pkl")
    joblib.dump(scaler_y, SCALERS_DIR / "scaler_y.pkl")
    joblib.dump(feature_columns, SCALERS_DIR / "feature_columns.pkl")
    
    # Create sequences
    def create_seq(X, y, ts=60):
        Xs, ys = [], []
        for i in range(len(X) - ts):
            Xs.append(X[i:i+ts])
            ys.append(y[i+ts])
        return np.array(Xs), np.array(ys)
    
    X_seq, y_seq = create_seq(X_scaled, y_scaled)
    
    split = int(0.8 * len(X_seq))
    return (X_seq[:split], X_seq[split:], y_seq[:split], y_seq[split:], 
            scaler_X, scaler_y, feature_columns)