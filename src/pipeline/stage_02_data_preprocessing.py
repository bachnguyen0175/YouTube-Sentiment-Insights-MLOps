from src.config.configuration import ConfigurationManager
from src.components.data.data_preprocessing import DataPreprocessing
from src.utils.common import logger

STAGE_NAME = "Data Preprocessing"

class DataPreprocessingTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        data_preprocessing_config = config.get_data_preprocessing_config()
        data_preprocessing = DataPreprocessing(config=data_preprocessing_config)
        data_preprocessing.load_raw_data()
        data_preprocessing.normalize_text()
        data_preprocessing.save_processed_data()

if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = DataPreprocessingTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
