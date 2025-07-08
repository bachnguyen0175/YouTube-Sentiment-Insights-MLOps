from src.config.configuration import ConfigurationManager
from src.components.model.model_registry import ModelRegistry
from src.utils.common import logger

STAGE_NAME = "Model Registry"

class ModelRegistryPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        model_registry_config = config.get_model_registry_config()
        model_registry = ModelRegistry(config=model_registry_config)
        model_registry.load_experiment_info()
        model_registry.register_model_to_mlflow()

if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = ModelRegistryPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
