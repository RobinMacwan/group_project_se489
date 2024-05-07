from se489_group_project.config.configuration import ConfigurationManager
from se489_group_project.components.data_ingestion import DataIngestion
from se489_group_project import logger


class GettingDataPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        getting_data_config = config.get_data_ingestion_config()
        getting_data = DataIngestion(config=getting_data_config)
        getting_data.download_file()
        getting_data.extract_zip_file()


if __name__ == '__main__':
    try:
        logger.info(f"Data Ingestion stage started")
        obj = GettingDataPipeline()
        obj.main()
        logger.info(f"Data Ingestion stage completed\n\n")
    except Exception as e:
        logger.exception(e)
        raise e
