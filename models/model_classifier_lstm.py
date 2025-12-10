import tensorflow as tf
import joblib
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class ModelClassifierLstm():
    def __init__(self):
        self.models = {}
        self.preprocessor = None
        self.sentiment_analyzer = None
        self.load_models()


    def load_models(self):
        try:
            self.models['random_forest'] = joblib.load('models/random_forest_model.pkl')

            self.preprocessor = joblib.load('models/preprocessor.pkl')

            self.sentiment_tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
            self.sentiment_model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")

        except Exception as e:
            print(f"⚠ Warning: Could not load models - {e}")
            print("Using mock predictions for development")


    def get_model(self, model_type:str):
        return self.models(model_type)