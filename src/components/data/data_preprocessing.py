
import numpy as np
import pandas as pd
import os
import logging
from src.utils.common import preprocess_text, download_nltk_data

# logging configuration
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
log_filepath = os.path.join(log_dir, "data_preprocessing.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_filepath),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Download required NLTK data
download_nltk_data()

def normalize_text(df):
    """Apply preprocessing to the text data in the dataframe."""
    try:
        df['clean_comment'] = df['clean_comment'].apply(preprocess_text)
        logger.debug('Text normalization completed')
        return df
    except Exception as e:
        logger.error(f"Error during text normalization: {e}")
        raise

def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str) -> None:
    """Save the processed train and test datasets."""
    try:
        interim_data_path = os.path.join(data_path, 'interim')
        logger.debug(f"Creating directory {interim_data_path}")
        
        os.makedirs(interim_data_path, exist_ok=True)  # Ensure the directory is created
        logger.debug(f"Directory {interim_data_path} created or already exists")

        train_data.to_csv(os.path.join(interim_data_path, "train_processed.csv"), index=False)
        test_data.to_csv(os.path.join(interim_data_path, "test_processed.csv"), index=False)
        
        logger.debug(f"Processed data saved to {interim_data_path}")
    except Exception as e:
        logger.error(f"Error occurred while saving data: {e}")
        raise

def main():
    try:
        logger.debug("Starting data preprocessing...")
        
        # Fetch the data from data/raw
        train_data = pd.read_csv('./data/raw/train.csv')
        test_data = pd.read_csv('./data/raw/test.csv')
        logger.debug('Data loaded successfully')

        # Preprocess the data
        train_processed_data = normalize_text(train_data)
        test_processed_data = normalize_text(test_data)

        # Save the processed data
        save_data(train_processed_data, test_processed_data, data_path='./data')
    except Exception as e:
        logger.error('Failed to complete the data preprocessing process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()
