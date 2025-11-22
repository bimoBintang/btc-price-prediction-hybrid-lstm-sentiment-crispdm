import shap # type: ignore
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from config.settings import RESULTS_DIR

def run_shap_analysis(model, X_train, X_test, feature_names):
    """
    Menjalankan SHAP Analysis untuk model LSTM.
    
    Args:
        model: Model Keras/TensorFlow yang sudah ditraining
        X_train: Data training untuk background (numpy array)
        X_test: Data test untuk dijelaskan (numpy array)
        feature_names: List nama fitur
        
    Returns:
        float: Rata-rata absolute SHAP value untuk Net_Sentiment_Score
    """
    print("Menjalankan SHAP Analysis...")
    
    # Pastikan RESULTS_DIR ada
    Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)
    
    # Gunakan subset data untuk background (lebih cepat)
    background_data = X_train[:100]
    test_data = X_test[:50]
    
    print(f"Background data shape: {background_data.shape}")
    print(f"Test data shape: {test_data.shape}")
    
    # Create explainer
    try:
        explainer = shap.DeepExplainer(model, background_data)
        print("DeepExplainer created successfully")
    except Exception as e:
        print(f"Error creating DeepExplainer: {e}")
        print("Trying GradientExplainer instead...")
        explainer = shap.GradientExplainer(model, background_data)
    
    # Compute SHAP values
    print("Computing SHAP values...")
    shap_values = explainer.shap_values(test_data)
    
    # Handle different output formats
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    
    print(f"SHAP values shape: {shap_values.shape}")
    
    # Create summary plot
    try:
        # Untuk LSTM dengan 3D input (samples, timesteps, features)
        # Ambil timestep terakhir untuk visualisasi
        if len(shap_values.shape) == 3:
            shap_last_timestep = shap_values[:, -1, :]  # Ambil timestep terakhir
            test_last_timestep = test_data[:, -1, :]
        else:
            shap_last_timestep = shap_values
            test_last_timestep = test_data
        
        # Plot summary
        shap.summary_plot(
            shap_last_timestep, 
            test_last_timestep, 
            feature_names=feature_names, 
            show=False
        )
        
        output_path = Path(RESULTS_DIR) / "shap_summary.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"SHAP summary plot saved to: {output_path}")
        
    except Exception as e:
        print(f"Error creating summary plot: {e}")
        plt.close()
    
    # Calculate sentiment feature importance
    try:
        if 'Net_Sentiment_Score' in feature_names:
            idx = feature_names.index('Net_Sentiment_Score')
            
            # Ambil SHAP values untuk sentiment feature
            if len(shap_values.shape) == 3:
                # Untuk 3D: (samples, timesteps, features)
                shap_sentiment_values = shap_values[:, -1, idx]  # Timestep terakhir
            else:
                # Untuk 2D: (samples, features)
                shap_sentiment_values = shap_values[:, idx]
            
            # Hitung rata-rata absolute SHAP value
            shap_sentiment = np.abs(shap_sentiment_values).mean()
            print(f"Rata-rata |SHAP| Net_Sentiment_Score = {shap_sentiment:.6f}")
            
            return shap_sentiment
        else:
            print(f"Warning: 'Net_Sentiment_Score' tidak ditemukan dalam feature_names")
            print(f"Available features: {feature_names}")
            return 0.0
            
    except Exception as e:
        print(f"Error calculating sentiment SHAP: {e}")
        return 0.0


def run_shap_analysis_alternative(model, X_train, X_test, feature_names):
    """
    Alternatif SHAP analysis dengan pendekatan berbeda.
    Gunakan ini jika DeepExplainer terlalu lambat.
    """
    print("Menjalankan SHAP Analysis (Alternative method)...")
    
    # Pastikan RESULTS_DIR ada
    Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)
    
    background_data = X_train[:50]  # Lebih sedikit untuk lebih cepat
    test_data = X_test[:20]
    
    # Gunakan Sampling atau Kernel Explainer
    try:
        # Create a wrapper function for prediction
        def predict_fn(x):
            return model.predict(x, verbose=0)
        
        # Use KernelExplainer (lebih lambat tapi lebih robust)
        explainer = shap.KernelExplainer(
            predict_fn, 
            background_data[:10]  # Lebih sedikit lagi
        )
        
        # Reshape untuk 2D jika perlu
        if len(test_data.shape) == 3:
            # Flatten timesteps
            test_2d = test_data.reshape(test_data.shape[0], -1)
            background_2d = background_data[:10].reshape(background_data[:10].shape[0], -1)
            
            explainer = shap.KernelExplainer(
                lambda x: predict_fn(x.reshape(-1, test_data.shape[1], test_data.shape[2])),
                background_2d
            )
            shap_values = explainer.shap_values(test_2d, nsamples=100)
        else:
            shap_values = explainer.shap_values(test_data, nsamples=100)
        
        print(f"SHAP values computed with shape: {shap_values.shape}")
        
        return 0.0  # Placeholder
        
    except Exception as e:
        print(f"Error in alternative SHAP analysis: {e}")
        return 0.0