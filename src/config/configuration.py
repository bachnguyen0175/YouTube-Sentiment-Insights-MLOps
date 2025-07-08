from src.constants import *
import os
from pathlib import Path
from src.utils.common import read_yaml, create_directories, save_json
from src.entity.config_entity import (DataIngestionConfig,
                                      DataPreprocessingConfig,
                                      ModelBuildingConfig,
                                      ModelEvaluationConfig,
                                      ModelRegistryConfig,
                                      PredictionConfig)


class ConfigurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])


    
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_URL=config.source_URL,
            local_data_file=config.local_data_file
        )

        return data_ingestion_config
    


    
    def get_data_preprocessing_config(self) -> DataPreprocessingConfig:
        config = self.config.data_preprocessing

        create_directories([config.root_dir])

        data_preprocessing_config = DataPreprocessingConfig(
            root_dir=config.root_dir,
            preprocessed_data_path=config.preprocessed_data_path,
            train_test_split_path=config.train_test_split_path,
            train_processed_csv=config.train_processed_csv,
            test_processed_csv=config.test_processed_csv
        )

        return data_preprocessing_config

    
    def get_model_building_config(self) -> ModelBuildingConfig:
        config = self.config.model_building
        params = self.params.model_building
        
        create_directories([config.root_dir])

        model_building_config = ModelBuildingConfig(
            root_dir=config.root_dir,
            trained_model_path=config.trained_model_path,
            vectorizer_path=config.vectorizer_path,
            ngram_range=tuple(params.ngram_range),
            max_features=params.max_features,
            learning_rate=params.learning_rate,
            max_depth=params.max_depth,
            n_estimators=params.n_estimators
        )

        return model_building_config

    
    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        config = self.config.model_evaluation
        
        model_evaluation_config = ModelEvaluationConfig(
            root_dir=config.root_dir,
            experiment_info_path=config.experiment_info_path,
            trained_model_path=self.config.model_building.trained_model_path,
            vectorizer_path=self.config.model_building.vectorizer_path,
            test_data_path=self.config.data_preprocessing.test_processed_csv
        )

        return model_evaluation_config

    
    def get_model_registry_config(self) -> ModelRegistryConfig:
        config = self.config.model_registry
        
        model_registry_config = ModelRegistryConfig(
            root_dir=config.root_dir,
            registered_model_path=config.registered_model_path,
            model_name=config.model_name,
            model_version=config.model_version,
            model_stage=config.model_stage,
            experiment_info_path=self.config.model_evaluation.experiment_info_path
        )

        return model_registry_config

    
    def get_prediction_config(self) -> PredictionConfig:
        config = self.config.prediction_service
        
        prediction_config = PredictionConfig(
            model_path=config.model_path,
            vectorizer_path=config.vectorizer_path,
            max_sequence_length=config.max_sequence_length,
            batch_size=config.batch_size
        )

        return prediction_config