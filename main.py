from se489_group_project import logger
from se489_group_project.pipeline.pipeline_01_getting_data import GettingDataPipeline
from se489_group_project.pipeline.pipeline_02_base_model_creation import PrepareBaseModelTrainingPipeline
from se489_group_project.pipeline.pipeline_03_model_training import ModelTrainingPipeline
from se489_group_project.pipeline.pipeline_04_model_evaluation import EvaluationPipeline
from se489_group_project.visualizations.visualize import start_profiler, stop_profiler, visualize_profile
import asyncio

async def main():
    try:
        # Start profiler
        profiler = start_profiler()
        logger.info(f"Data Ingestion started")
        data_ingestion = GettingDataPipeline()
        data_ingestion.main()
        logger.info(f"Data Ingestion completed")
          # Stop profiler and get the profile file path
        profile_file = stop_profiler(profiler, "data_ingestion")
        # Visualize the profile using snakeviz
        visualize_profile(profile_file)

    except Exception as e:
        logger.exception(e)
        raise e

    try:
        profiler = start_profiler()
        logger.info(f"Prepare base model stage started")
        prepare_base_model = PrepareBaseModelTrainingPipeline()
        prepare_base_model.main()
        logger.info(f"Prepare base model stage completed\n\n")
        profile_file = stop_profiler(profiler, "prepare_base_model")
        visualize_profile(profile_file)

    except Exception as e:
        logger.exception(e)
        raise e

    try:
        profiler = start_profiler()
        logger.info(f"Training stage started")
        model_trainer = ModelTrainingPipeline()
        model_trainer.main()
        logger.info(f"Training stage completed\n\n")
        profile_file = stop_profiler(profiler, "model_training")
        visualize_profile(profile_file)
        
    except Exception as e:
        logger.exception(e)
        raise e

    try:
        profiler = start_profiler()
        logger.info(f"Evaluation stage started")
        model_evaluation = EvaluationPipeline()
        model_evaluation.main()
        logger.info(f"Evaluation stage completed\n\n")
        profile_file = stop_profiler(profiler, "model_evaluation")
        visualize_profile(profile_file)

    except Exception as e:
        logger.exception(e)
        raise e
if __name__ == '__main__':
    asyncio.run(main())
    # profiler = start_profiler()
    # main()
    # profile_file = stop_profiler(profiler, "main")
    # visualize_profile(profile_file)   