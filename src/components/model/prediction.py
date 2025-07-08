import numpy as np
import pandas as pd
import os
import pickle
import logging
from typing import Union, List
from pathlib import Path
from src.entity.config_entity import PredictionConfig
from src.utils.common import load_model, load_object, preprocess_text as common_preprocess_text

# logging configuration
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
log_filepath = os.path.join(log_dir, "prediction.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_filepath),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


class PredictionPipeline:
    def __init__(self, config: PredictionConfig):
        self.config = config
        self.model = None
        self.vectorizer = None
        
    def load_model_and_vectorizer(self):
        """Load the trained model and vectorizer"""
        try:
            self.model = load_model(Path(self.config.model_path))
            self.vectorizer = load_object(Path(self.config.vectorizer_path))
            logger.info("Model and vectorizer loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model and vectorizer: {e}")
            raise
    
    def preprocess_text(self, text: Union[str, List[str]]) -> Union[str, List[str]]:
        """Preprocess text data using the common utility function."""
        if isinstance(text, str):
            return common_preprocess_text(text)
        elif isinstance(text, list):
            return [common_preprocess_text(t) for t in text]
        else:
            raise ValueError("Input must be a string or list of strings")
    
    def predict_single(self, text: str) -> dict:
        """Make prediction for a single text"""
        if self.model is None or self.vectorizer is None:
            self.load_model_and_vectorizer()
        
        try:
            # Preprocess the text
            processed_text = self.preprocess_text(text)
            
            # Vectorize the text
            text_vector = self.vectorizer.transform([processed_text])
            
            # Make prediction
            prediction = self.model.predict(text_vector)[0]
            prediction_proba = self.model.predict_proba(text_vector)[0]
            
            # Get confidence score
            confidence = max(prediction_proba)
            
            # Map prediction to sentiment label
            sentiment_map = {0: 'negative', 1: 'neutral', 2: 'positive'}
            sentiment = sentiment_map.get(prediction, 'unknown')
            
            result = {
                'text': text,
                'processed_text': processed_text,
                'prediction': int(prediction),
                'sentiment': sentiment,
                'confidence': float(confidence),
                'probabilities': {
                    'negative': float(prediction_proba[0]),
                    'neutral': float(prediction_proba[1]),
                    'positive': float(prediction_proba[2])
                }
            }
            
            logger.info(f"Prediction made for text: {text[:50]}... - Sentiment: {sentiment}")
            return result
            
        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            raise
    
    def predict_batch(self, texts: List[str]) -> List[dict]:
        """Make predictions for a batch of texts"""
        if self.model is None or self.vectorizer is None:
            self.load_model_and_vectorizer()
        
        try:
            results = []
            
            # Process texts in batches to avoid memory issues
            batch_size = self.config.batch_size
            
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                
                # Preprocess the batch
                processed_texts = self.preprocess_text(batch_texts)
                
                # Vectorize the batch
                text_vectors = self.vectorizer.transform(processed_texts)
                
                # Make predictions
                predictions = self.model.predict(text_vectors)
                predictions_proba = self.model.predict_proba(text_vectors)
                
                # Process results for this batch
                for j, (text, processed_text, pred, pred_proba) in enumerate(
                    zip(batch_texts, processed_texts, predictions, predictions_proba)
                ):
                    confidence = max(pred_proba)
                    sentiment_map = {0: 'negative', 1: 'neutral', 2: 'positive'}
                    sentiment = sentiment_map.get(pred, 'unknown')
                    
                    result = {
                        'text': text,
                        'processed_text': processed_text,
                        'prediction': int(pred),
                        'sentiment': sentiment,
                        'confidence': float(confidence),
                        'probabilities': {
                            'negative': float(pred_proba[0]),
                            'neutral': float(pred_proba[1]),
                            'positive': float(pred_proba[2])
                        }
                    }
                    results.append(result)
            
            logger.info(f"Batch predictions made for {len(texts)} texts")
            return results
            
        except Exception as e:
            logger.error(f"Error making batch predictions: {e}")
            raise
    
    def predict_from_csv(self, csv_path: str, text_column: str, output_path: str = None) -> pd.DataFrame:
        """Make predictions for texts in a CSV file"""
        try:
            # Load CSV
            df = pd.read_csv(csv_path)
            
            if text_column not in df.columns:
                raise ValueError(f"Column '{text_column}' not found in CSV")
            
            # Get texts from the specified column
            texts = df[text_column].fillna("").astype(str).tolist()
            
            # Make predictions
            predictions = self.predict_batch(texts)
            
            # Add predictions to dataframe
            df['predicted_sentiment'] = [pred['sentiment'] for pred in predictions]
            df['confidence'] = [pred['confidence'] for pred in predictions]
            df['negative_prob'] = [pred['probabilities']['negative'] for pred in predictions]
            df['neutral_prob'] = [pred['probabilities']['neutral'] for pred in predictions]
            df['positive_prob'] = [pred['probabilities']['positive'] for pred in predictions]
            
            # Save results if output path is provided
            if output_path:
                df.to_csv(output_path, index=False)
                logger.info(f"Predictions saved to {output_path}")
            
            logger.info(f"Predictions made for CSV with {len(df)} rows")
            return df
            
        except Exception as e:
            logger.error(f"Error processing CSV: {e}")
            raise
