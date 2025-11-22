# src/evaluation/metrics.py
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import numpy as np

def evaluate_models(model_bench, model_hybrid, X_bench_test, X_test, y_test, scaler_y):
    from config.settings import RESULTS_DIR
    
    y_pred_bench = model_bench.predict(X_bench_test)
    y_pred_hybrid = model_hybrid.predict(X_test)
    
    y_test_inv = scaler_y.inverse_transform(y_test)
    y_bench_inv = scaler_y.inverse_transform(y_pred_bench)
    y_hybrid_inv = scaler_y.inverse_transform(y_pred_hybrid)
    
    mae_bench = mean_absolute_error(y_test_inv, y_bench_inv)
    mae_hybrid = mean_absolute_error(y_test_inv, y_hybrid_inv)
    improvement = ((mae_bench - mae_hybrid) / mae_bench) * 100
    
    print(f"\nMAE Benchmark : ${mae_bench:,.2f}")
    print(f"MAE Hybrid    : ${mae_hybrid:,.2f}")
    print(f"Improvement   : {improvement:.2f}%")
    
    # Plot
    plt.figure(figsize=(14,6))
    plt.plot(y_test_inv[:200], label='Actual', linewidth=2)
    plt.plot(y_hybrid_inv[:200], label='Hybrid LSTM + Sentimen', alpha=0.8)
    plt.plot(y_bench_inv[:200], label='Benchmark (tanpa sentimen)', alpha=0.6, linestyle='--')
    plt.title('Prediksi Harga BTC - Hybrid vs Benchmark')
    plt.legend()
    plt.savefig(RESULTS_DIR / "predictions_plot.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    return mae_bench, mae_hybrid, improvement