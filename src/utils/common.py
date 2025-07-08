import os
import logging
import pandas as pd
import pickle
import nltk
import re
from box.exceptions import BoxValueError
import yaml
import json
import joblib
from ensure import ensure_annotations
from box import ConfigBox
from pathlib import Path
from typing import Any, Dict, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt
import seaborn as sns



@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """reads yaml file and returns

    Args:
        path_to_yaml (str): path like input

    Raises:
        ValueError: if yaml file is empty
        e: empty file

    Returns:
        ConfigBox: ConfigBox type
    """
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"yaml file: {path_to_yaml} loaded successfully")
            return ConfigBox(content)
    except BoxValueError:
        raise ValueError("yaml file is empty")
    except Exception as e:
        raise e
    


@ensure_annotations
def create_directories(path_to_directories: list, verbose=True):
    """create list of directories

    Args:
        path_to_directories (list): list of path of directories
        ignore_log (bool, optional): ignore if multiple dirs is to be created. Defaults to False.
    """
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info(f"created directory at: {path}")


@ensure_annotations
def save_json(path: Path, data: dict):
    """save json data

    Args:
        path (Path): path to json file
        data (dict): data to be saved in json file
    """
    with open(path, "w") as f:
        json.dump(data, f, indent=4)

    logger.info(f"json file saved at: {path}")




@ensure_annotations
def load_json(path: Path) -> ConfigBox:
    """load json files data

    Args:
        path (Path): path to json file

    Returns:
        ConfigBox: data as class attributes instead of dict
    """
    with open(path) as f:
        content = json.load(f)

    logger.info(f"json file loaded succesfully from: {path}")
    return ConfigBox(content)


@ensure_annotations
def save_bin(data: Any, path: Path):
    """save binary file

    Args:
        data (Any): data to be saved as binary
        path (Path): path to binary file
    """
    joblib.dump(value=data, filename=path)
    logger.info(f"binary file saved at: {path}")


@ensure_annotations
def load_bin(path: Path) -> Any:
    """load binary data

    Args:
        path (Path): path to binary file

    Returns:
        Any: object stored in the file
    """
    data = joblib.load(path)
    logger.info(f"binary file loaded from: {path}")
    return data

@ensure_annotations
def get_size(path: Path) -> str:
    """get size in KB

    Args:
        path (Path): path of the file

    Returns:
        str: size in KB
    """
    size_in_kb = round(os.path.getsize(path)/1024)
    return f"~ {size_in_kb} KB"

# =============================================================================
# LOGGING UTILITIES
# =============================================================================

def setup_logger(name: str, log_file: str = None, level: int = logging.DEBUG) -> logging.Logger:
    """Setup a logger with console and file handlers.
    
    Args:
        name (str): Logger name
        log_file (str, optional): Log file path
        level (int): Logging level
        
    Returns:
        logging.Logger: Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Clear existing handlers
    logger.handlers = []
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.ERROR)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

# Create default logger for the package
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
log_filepath = os.path.join(log_dir, "ete_errors.log")

logger = setup_logger('ete', log_filepath)


# =============================================================================
# CONFIGURATION UTILITIES (EXISTING - KEEP AS IS)
# =============================================================================

# =============================================================================
# DATA LOADING & SAVING UTILITIES
# =============================================================================

@ensure_annotations
def load_data_from_url(url: str) -> pd.DataFrame:
    """Load data from a URL.
    
    Args:
        url (str): URL to load data from
        
    Returns:
        pd.DataFrame: Loaded data
    """
    try:
        df = pd.read_csv(url)
        logger.info(f"Data loaded from URL: {url}")
        return df
    except pd.errors.ParserError as e:
        logger.error(f"Failed to parse CSV from URL {url}: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error loading data from URL {url}: {e}")
        raise

@ensure_annotations
def load_data_from_csv(file_path: Path) -> pd.DataFrame:
    """Load data from a CSV file.
    
    Args:
        file_path (Path): Path to CSV file
        
    Returns:
        pd.DataFrame: Loaded data
    """
    try:
        df = pd.read_csv(file_path)
        df.fillna('', inplace=True)  # Fill any NaN values
        logger.info(f"Data loaded from CSV: {file_path}")
        return df
    except Exception as e:
        logger.error(f"Error loading data from {file_path}: {e}")
        raise

@ensure_annotations
def save_dataframe(df: pd.DataFrame, file_path: Path, index: bool = False):
    """Save DataFrame to CSV file.
    
    Args:
        df (pd.DataFrame): DataFrame to save
        file_path (Path): Output file path
        index (bool): Whether to save index
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        df.to_csv(file_path, index=index)
        logger.info(f"DataFrame saved to: {file_path}")
    except Exception as e:
        logger.error(f"Error saving DataFrame to {file_path}: {e}")
        raise


# =============================================================================
# MODEL & PICKLE UTILITIES  
# =============================================================================

@ensure_annotations
def save_model(model: Any, file_path: Path):
    """Save model using pickle.
    
    Args:
        model (Any): Model to save
        file_path (Path): Output file path
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'wb') as f:
            pickle.dump(model, f)
        logger.info(f"Model saved to: {file_path}")
    except Exception as e:
        logger.error(f"Error saving model to {file_path}: {e}")
        raise

@ensure_annotations
def load_model(file_path: Path) -> object:
    """Load model from pickle file.
    
    Args:
        file_path (Path): Path to model file
        
    Returns:
        Any: Loaded model
    """
    try:
        with open(file_path, 'rb') as f:
            model = pickle.load(f)
        logger.info(f"Model loaded from: {file_path}")
        return model
    except Exception as e:
        logger.error(f"Error loading model from {file_path}: {e}")
        raise

@ensure_annotations
def save_vectorizer(vectorizer: TfidfVectorizer, file_path: Path):
    """Save TF-IDF vectorizer.
    
    Args:
        vectorizer (TfidfVectorizer): Vectorizer to save
        file_path (Path): Output file path
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'wb') as f:
            pickle.dump(vectorizer, f)
        logger.info(f"Vectorizer saved to: {file_path}")
    except Exception as e:
        logger.error(f"Error saving vectorizer to {file_path}: {e}")
        raise

@ensure_annotations
def load_object(file_path: Path) -> object:
    """Load any object (vectorizer, preprocessor, etc.) from pickle file.
    
    Args:
        file_path (Path): Path to object file
        
    Returns:
        Any: Loaded object
    """
    try:
        with open(file_path, 'rb') as f:
            obj = pickle.load(f)
        logger.info(f"Object loaded from: {file_path}")
        return obj
    except Exception as e:
        logger.error(f"Error loading object from {file_path}: {e}")
        raise


# =============================================================================
# TEXT PROCESSING UTILITIES
# =============================================================================

@ensure_annotations
def download_nltk_data():
    """Download required NLTK data."""
    try:
        nltk.download('wordnet', quiet=True)
        nltk.download('stopwords', quiet=True)
        logger.info("NLTK data downloaded successfully")
    except Exception as e:
        logger.error(f"Error downloading NLTK data: {e}")
        raise

@ensure_annotations
def preprocess_text(text: str) -> str:
    """Apply text preprocessing for sentiment analysis.
    
    Args:
        text (str): Input text
        
    Returns:
        str: Preprocessed text
    """
    try:
        # Convert to lowercase
        text = text.lower()
        
        # Remove trailing and leading whitespaces
        text = text.strip()
        
        # Remove newline characters
        text = re.sub(r'\n', ' ', text)
        
        # Remove non-alphanumeric characters, except punctuation
        text = re.sub(r'[^A-Za-z0-9\s!?.,]', '', text)
        
        # Remove stopwords but retain important ones for sentiment analysis
        stop_words = set(stopwords.words('english')) - {'not', 'but', 'however', 'no', 'yet'}
        text = ' '.join([word for word in text.split() if word not in stop_words])
        
        # Lemmatize the words
        lemmatizer = WordNetLemmatizer()
        text = ' '.join([lemmatizer.lemmatize(word) for word in text.split()])
        
        return text
    except Exception as e:
        logger.error(f"Error preprocessing text: {e}")
        return text  # Return original text if preprocessing fails


# =============================================================================
# METRICS & VISUALIZATION UTILITIES
# =============================================================================

@ensure_annotations
def save_confusion_matrix(cm, file_path: Path, title: str = "Confusion Matrix"):
    """Save confusion matrix plot.
    
    Args:
        cm: Confusion matrix array
        file_path (Path): Output file path
        title (str): Plot title
    """
    try:
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(title)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        plt.savefig(file_path)
        plt.close()
        logger.info(f"Confusion matrix saved to: {file_path}")
    except Exception as e:
        logger.error(f"Error saving confusion matrix to {file_path}: {e}")
        raise

@ensure_annotations
def save_metrics(metrics: Dict, file_path: Path):
    """Save evaluation metrics to JSON.
    
    Args:
        metrics (Dict): Metrics dictionary
        file_path (Path): Output file path
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        logger.info(f"Metrics saved to: {file_path}")
    except Exception as e:
        logger.error(f"Error saving metrics to {file_path}: {e}")
        raise

