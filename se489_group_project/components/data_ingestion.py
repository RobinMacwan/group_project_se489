# -*- coding: utf-8 -*-
import os
import zipfile

import gdown

from se489_group_project import logger
from se489_group_project.model_classes.config_model_classes import GettingDataConfig


class DataIngestion:
    """
    A class used for data ingestion (downloading and extracting data files).

    Attributes
    ----------
    config : GettingDataConfig
        Configuration parameters for the data ingestion process.

    Methods
    -------
    download_file() -> str:
        Download a zip file from URL to a local directory.
    extract_zip_file():
        Extract the downloaded zip file into the previously specified directory.
    """

    def __init__(self, config: GettingDataConfig):
        """
        Initialize the DataIngestion class with the given configuration.

        Parameters
        ----------
        config : GettingDataConfig
            Configuration instance containing parameters for data ingestion.
        """
        self.config = config

    def download_file(self) -> str:
        """
        Download zip file from URL to a local directory.

        The URL is provided via the configuration, and the file is downloaded using `gdown`.
        The file will be stored in `self.config.local_data_file`.

        Returns
        -------
        str
            Path to the downloaded zip file.

        """

        try:
            dataset_url = self.config.source_URL
            zip_download_dir = self.config.local_data_file
            os.makedirs("data/raw", exist_ok=True)
            logger.info(f"Downloading data from {dataset_url} into file {zip_download_dir}")

            file_id = dataset_url.split("/")[-2]
            prefix = "https://drive.google.com/uc?/export=download&id="
            gdown.download(prefix + file_id, zip_download_dir)

            logger.info(f"Downloaded data from {dataset_url} into file {zip_download_dir}")

        except Exception as e:
            raise e

    def extract_zip_file(self):
        """
        Extract the zip file into the data directory.

        The zip file is extracted to `self.config.unzip_dir`.
        Creates the output directory if it doesn't exist.

        """
        unzip_path = self.config.unzip_dir
        os.makedirs(unzip_path, exist_ok=True)
        with zipfile.ZipFile(self.config.local_data_file, "r") as zip_ref:
            zip_ref.extractall(unzip_path)
