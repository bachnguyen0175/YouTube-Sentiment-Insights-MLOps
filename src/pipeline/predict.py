from src.config.configuration import ConfigurationManager
from src.components.model.prediction import PredictionPipeline
from src.utils.common import logger

STAGE_NAME = "Prediction"

class PredictionServicePipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        prediction_config = config.get_prediction_config()
        prediction_component = PredictionPipeline(config=prediction_config)
        
        # Example: Single text prediction
        sample_text = "This movie is absolutely amazing! I loved every minute of it."
        result = prediction_component.predict_single(sample_text)
        logger.info(f"Prediction result: {result}")
        
        return result

if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = PredictionServicePipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<")
    except Exception as e:
        logger.exception(e)
        raise e
