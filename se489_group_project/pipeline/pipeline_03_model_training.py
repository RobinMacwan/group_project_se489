from se489_group_project.config.configuration import ConfigurationManager
from se489_group_project.components.model_training import Training
from se489_group_project import logger



class ModelTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        training_config = config.get_training_config()
        training = Training(config=training_config)
        training.get_base_model()
        training.train_valid_generator()
        training.train()



if __name__ == '__main__':
    try:
        logger.info(f"Training stage started")
        obj = ModelTrainingPipeline()
        obj.main()
        logger.info(f"Training stage completed\n\n")
    except Exception as e:
        logger.exception(e)
        raise e
        