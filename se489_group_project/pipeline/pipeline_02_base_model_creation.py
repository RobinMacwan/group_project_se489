from se489_group_project.config.configuration import ConfigurationManager
from se489_group_project.components.prepare_base_model import PrepareBaseModel
from se489_group_project import logger


class PrepareBaseModelTrainingPipeline:
    """
    A class used to manage the prepare base model process.
    """
    def __init__(self):
        """
        Initialization of the PrepareBaseModelTrainingPipeline class.
        """
        pass

    def main(self):
        """
        Main method to execute the prepare base model process.
        This method is responsible for the configuration of the prepare base model process.
        As well as getting the base model and updating the base model.
        """
        #Initialize the configuration manager
        config = ConfigurationManager()

        #Get prepare base model configuration
        prepare_base_model_config = config.get_prepare_base_model_config()

        # Initialize PrepareBaseModel with the configuration
        prepare_base_model = PrepareBaseModel(config=prepare_base_model_config)
        
        # Get and update the base model
        prepare_base_model.get_base_model()
        prepare_base_model.update_base_model()


    
if __name__ == '__main__':
    try:
        logger.info(f"Prepare base model stage started")

        #Create an instance of PrepareBaseModelTrainingPipeline
        obj = PrepareBaseModelTrainingPipeline()
        #execute the main method
        obj.main()
        logger.info(f"Prepare base model stage completed\n\n")
    except Exception as e:
        logger.exception(e)
        raise e