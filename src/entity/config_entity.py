from dataclasses import dataclass
from pathlib import Path
from typing import Tuple


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path



@dataclass(frozen=True)
class DataPreprocessingConfig:
    root_dir: Path
    preprocessed_data_path: Path
    train_test_split_path: Path
    train_processed_csv: Path
    test_processed_csv: Path


@dataclass(frozen=True)
class ModelBuildingConfig:
    root_dir: Path
    trained_model_path: Path
    vectorizer_path: Path
    ngram_range: Tuple[int, int]
    max_features: int
    learning_rate: float
    max_depth: int
    n_estimators: int


@dataclass(frozen=True)
class ModelEvaluationConfig:
    root_dir: Path
    experiment_info_path: Path
    trained_model_path: Path
    vectorizer_path: Path
    test_data_path: Path


@dataclass(frozen=True)
class ModelRegistryConfig:
    root_dir: Path
    registered_model_path: Path
    model_name: str
    model_version: str
    model_stage: str
    experiment_info_path: Path


@dataclass(frozen=True)
class PredictionConfig:
    model_path: Path
    vectorizer_path: Path
    max_sequence_length: int
    batch_size: int