# 🏃‍♂️ Running in Google Colab

This project is optimized for Google Colab. Follow these steps to get started.

## 1. Quick Start
1.  Open [Google Colab](https://colab.research.google.com/).
2.  Select **File > Upload notebook**.
3.  Upload `notebooks/btc_prediction_colab.ipynb` from this repository.

## 2. API Keys
The notebook uses Google Colab's `userdata` system for secure API key management.
Before running the "Configuration" cells, add the following secrets in Colab (key icon on the left sidebar):

| Secret Name | Value | Required? |
|-------------|-------|-----------|
| `COINGECKO_API_KEY` | Your CoinGecko API Key | Optional |
| `TWITTER_API_KEY` | Your Twitter API Key | Optional |
| `REDDIT_CLIENT_ID` | Your Reddit Client ID | Optional |

## 3. Running the Pipeline
- **Step 1**: Run the "Clone Repository" cell.
- **Step 2**: Run "Install Dependencies".
- **Step 3**: Execute the pipeline steps sequentially.

## 4. Troubleshooting
- **Missing Module Errors**: Ensure you have successfully run the setup cells and that the `os.chdir` command executed correctly.
- **GPU**: Go to **Runtime > Change runtime type** and select **T4 GPU** for faster training.
