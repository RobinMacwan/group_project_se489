from se489_group_project import logger
from se489_group_project.pipeline.pipeline_01_getting_data import GettingDataPipeline
from se489_group_project.pipeline.pipeline_02_base_model_creation import PrepareBaseModelTrainingPipeline
from se489_group_project.pipeline.pipeline_03_model_training import ModelTrainingPipeline
from se489_group_project.pipeline.pipeline_04_model_evaluation import EvaluationPipeline

try:
    logger.info(f"Data Ingestion started")
    data_ingestion = GettingDataPipeline()
    data_ingestion.main()
    logger.info(f"Data Ingestion completed")
except Exception as e:
    logger.exception(e)
    raise e

try:
    logger.info(f"Prepare base model stage started")
    prepare_base_model = PrepareBaseModelTrainingPipeline()
    prepare_base_model.main()
    logger.info(f"Prepare base model stage completed\n\n")
except Exception as e:
    logger.exception(e)
    raise e

try:
    logger.info(f"Training stage started")
    model_trainer = ModelTrainingPipeline()
    model_trainer.main()
    logger.info(f"Training stage completed\n\n")
except Exception as e:
    logger.exception(e)
    raise e

try:
    logger.info(f"Evaluation stage started")
    model_evaluation = EvaluationPipeline()
    model_evaluation.main()
    logger.info(f"Evaluation stage completed\n\n")

except Exception as e:
    logger.exception(e)
    raise e
