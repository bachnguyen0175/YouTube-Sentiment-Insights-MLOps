from src.config.configuration import ConfigurationManager
from src.components.model.model_building import ModelBuilding
from src.utils.common import logger

STAGE_NAME = "Model Building"

class ModelBuildingTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        model_building_config = config.get_model_building_config()
        model_building = ModelBuilding(config=model_building_config)
        model_building.load_preprocessed_data()
        model_building.apply_tfidf_vectorization()
        model_building.train_lightgbm_model()
        model_building.save_trained_model()

if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = ModelBuildingTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
