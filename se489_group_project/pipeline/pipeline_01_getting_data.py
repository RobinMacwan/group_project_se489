# -*- coding: utf-8 -*-
"""This module is responsible for the data ingestion stage of the pipeline."""
from se489_group_project import logger
from se489_group_project.components.data_ingestion import DataIngestion
from se489_group_project.config.configuration import ConfigurationManager


class GettingDataPipeline:
    """
    A class used to manage the data ingestion process.
    """

    def __init__(self):
        """
        Initialization of the GettingDataPipeline class.
        """
        pass

    def main(self):
        """
        Main method to execute the data ingestion process.
        This method is responsible for the configuration of the data ingestion process.
        As well as downloading and extracting the data files.
        """

        config = ConfigurationManager()

        # Get data ingestion configuration
        getting_data_config = config.get_data_ingestion_config()
        # Initialize DataIngestion with the retrieved configuration
        getting_data = DataIngestion(config=getting_data_config)

        getting_data.download_file()
        # Profile the extract_zip_file method of DataIngestion
        getting_data.extract_zip_file()


if __name__ == "__main__":
    """
    The main method to run the data ingestion stage of the pipeline.
    """
    try:
        logger.info("Data Ingestion stage started")
        obj = GettingDataPipeline()
        obj.main()
        logger.info("Data Ingestion stage completed\n\n")
    except Exception as e:
        logger.exception(e)
        raise e
