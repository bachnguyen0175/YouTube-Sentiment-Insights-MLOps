from src.config.configuration import ConfigurationManager
from src.components.model.model_evaluation import ModelEvaluation
from src.utils.common import logger

STAGE_NAME = "Model Evaluation"

class ModelEvaluationPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        model_evaluation_config = config.get_model_evaluation_config()
        model_evaluation = ModelEvaluation(config=model_evaluation_config)
        model_evaluation.load_model_and_data()
        model_evaluation.evaluate_model()
        model_evaluation.log_metrics_to_mlflow()
        model_evaluation.save_evaluation_results()

if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = ModelEvaluationPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
