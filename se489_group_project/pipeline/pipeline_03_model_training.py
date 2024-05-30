from se489_group_project.config.configuration import ConfigurationManager
from se489_group_project.components.model_training import Training
from se489_group_project import logger



class ModelTrainingPipeline:
    """
    Class to run the training stage of the pipeline
    """
    def __init__(self):
        """
        Initialization of the ModelTrainingPipeline class.
        """
        pass

    def main(self):
        """
        Main method to run the training stage of the pipeline.
        Responsible for getting the training configuration,
        running the training, and saving the model.
        """

        #Initialize the configuration manager
        config = ConfigurationManager()
        #Retrieve the training configuration
        training_config = config.get_training_config()
        #Initialize the training class
        training = Training(config=training_config)
        #Retrieve the base model
        training.get_base_model()
        #Retrieve the data generators
        training.train_valid_generator()
        #Train the model
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
        