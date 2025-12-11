#!/usr/bin/env python3
"""
Run the BTC Prediction API server.

Usage:
    python main.py
    
Or with uvicorn directly:
    uvicorn src.api.api:app --reload --host 0.0.0.0 --port 8000
"""
import os
import sys

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)


def main():
    """Start the API server."""
    import uvicorn
    
    print("=" * 60)
    print("🚀 BTC PRICE PREDICTION API")
    print("=" * 60)
    print()
    print("📍 Server: http://localhost:8000")
    print("📚 API Documentation: http://localhost:8000/api/docs")
    print("📖 ReDoc: http://localhost:8000/api/redoc") 
    print()
    print("📡 Available Endpoints:")
    print("   GET /api/health              - Health check")
    print("   GET /api/price/current       - Current BTC price")
    print("   GET /api/price/historical    - Historical prices")
    print("   GET /api/prediction          - Price prediction")
    print("   GET /api/sentiment           - Sentiment data")
    print("   GET /api/indicators          - Technical indicators")
    print("   GET /api/models              - Model metrics")
    print("   GET /api/dashboard           - All dashboard data")
    print()
    print("=" * 60)
    print()
    
    uvicorn.run(
        "src.api.api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )


if __name__ == "__main__":
    main()
