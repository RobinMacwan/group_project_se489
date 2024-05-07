from se489_group_project.config.configuration import ConfigurationManager
from se489_group_project.components.prepare_base_model import PrepareBaseModel
from se489_group_project import logger


class PrepareBaseModelTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        prepare_base_model_config = config.get_prepare_base_model_config()
        prepare_base_model = PrepareBaseModel(config=prepare_base_model_config)
        prepare_base_model.get_base_model()
        prepare_base_model.update_base_model()


    
if __name__ == '__main__':
    try:
        logger.info(f"Prepare base model stage started")
        obj = PrepareBaseModelTrainingPipeline()
        obj.main()
        logger.info(f"Prepare base model stage completed\n\n")
    except Exception as e:
        logger.exception(e)
        raise e