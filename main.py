from src.utils.common import logger
from src.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from src.pipeline.stage_02_data_preprocessing import DataPreprocessingTrainingPipeline
from src.pipeline.stage_03_model_building import ModelBuildingTrainingPipeline
from src.pipeline.stage_04_model_evaluation import ModelEvaluationTrainingPipeline
from src.pipeline.stage_05_model_registry import ModelRegistryTrainingPipeline


# Stage 1: Data Ingestion
STAGE_NAME = "Data Ingestion stage"
try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    data_ingestion = DataIngestionTrainingPipeline()
    data_ingestion.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e


# Stage 2: Data Preprocessing
STAGE_NAME = "Data Preprocessing stage"
try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    data_preprocessing = DataPreprocessingTrainingPipeline()
    data_preprocessing.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e


# Stage 3: Model Building
STAGE_NAME = "Model Building stage"
try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    model_building = ModelBuildingTrainingPipeline()
    model_building.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e


# Stage 4: Model Evaluation
STAGE_NAME = "Model Evaluation stage"
try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    model_evaluation = ModelEvaluationTrainingPipeline()
    model_evaluation.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e


# Stage 5: Model Registry
STAGE_NAME = "Model Registry stage"
try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    model_registry = ModelRegistryTrainingPipeline()
    model_registry.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e


def main():
    """Main function to orchestrate the entire ML pipeline"""
    logger.info("="*70)
    logger.info("STARTING COMPLETE ML PIPELINE EXECUTION")
    logger.info("="*70)
    
    try:
        # All stages are executed above in sequence
        logger.info("="*70)
        logger.info("COMPLETE ML PIPELINE EXECUTION FINISHED SUCCESSFULLY!")
        logger.info("="*70)
        
    except Exception as e:
        logger.error("="*70)
        logger.error("ML PIPELINE EXECUTION FAILED!")
        logger.error("="*70)
        logger.exception(e)
        raise e


if __name__ == "__main__":
    main()
