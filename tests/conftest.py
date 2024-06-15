# -*- coding: utf-8 -*-
"""Module to for shared fixtures."""
import os
import sys
import tempfile

import pytest

# Ensure the project root directory is in the Python path, fixes import issues.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from se489_group_project.components.data_ingestion import DataIngestion
from se489_group_project.components.model_training import Training
from se489_group_project.components.prepare_base_model import PrepareBaseModel
from se489_group_project.model_classes.config_model_classes import (
    CreateBaseModelConfig,
    GettingDataConfig,
    ModelTrainingConfig,
)


@pytest.fixture(autouse=True, scope="session")
def temp_dir():
    """
    Fixture for creating a temporary directory for the duration of the test.
    Scope is set to session to ensure that the fixture is only created once.
    Auto use is set to true to ensure that the fixture is used for all tests.

    """
    with tempfile.TemporaryDirectory() as temporary_directory:
        yield temporary_directory


@pytest.fixture(autouse=True, scope="session")
def data_ingestion_config(temp_dir):
    """
    Fixture for providing a configuration object for data ingestion.
    Scope is set to session to ensure that the fixture is only created once.
    Auto use is set to true to ensure that the fixture is used for all tests.

    Parameters
    ----------
    temp_dir : str
        Temporary directory path.

    Returns
    -------
    DataIngestion
        DataIngestion object with the provided configuration.

    """
    return GettingDataConfig(
        source_URL="https://drive.google.com/file/d/1gesLApompvvnzz-AWyWM4ikmk7BOGWAp/view?usp=sharing",  # Corrected URL format
        local_data_file=os.path.join(temp_dir, "data.zip"),
        unzip_dir=os.path.join(temp_dir, "kidney-ct-scan-image"),
        root_dir=temp_dir,
    )


@pytest.fixture(autouse=True, scope="session")
def data_ingestion(data_ingestion_config):
    """
    Fixture for providing a DataIngestion object with the provided configuration.
    Scope is set to session to ensure that the fixture is only created once.
    Auto use is set to true to ensure that the fixture is used for all tests.

     Parameters
     ----------
     data_ingestion_config : GettingDataConfig
         Configuration object for data ingestion.

     Returns
     -------
     DataIngestion
         DataIngestion object with the provided configuration.
    """
    return DataIngestion(data_ingestion_config)


@pytest.fixture(scope="session")
def downloaded_files(data_ingestion_config):
    """
    Fixture to download the files and extract the zip file.
    """
    data_ingestion = DataIngestion(data_ingestion_config)
    data_ingestion.download_file()
    data_ingestion.extract_zip_file()
    return data_ingestion


@pytest.fixture(scope="session")
def base_model_config(temp_dir):
    return CreateBaseModelConfig(
        root_dir=temp_dir,
        base_model_path=os.path.join(temp_dir, "base_model"),
        updated_base_model_path=os.path.join(temp_dir, "updated_base_model"),
        params_image_size=(224, 224, 3),
        params_weights="imagenet",
        params_include_top=False,
        params_classes=2,
        params_learning_rate=0.001,
    )


@pytest.fixture(scope="session")
def prepare_base_model(base_model_config):
    """Fixture to create an instance of PrepareBaseModel."""
    return PrepareBaseModel(base_model_config)


@pytest.fixture(scope="session")
def prepared_base_model(base_model_config):
    prepare_base_model = PrepareBaseModel(base_model_config)
    prepare_base_model.get_base_model()
    prepare_base_model.update_base_model()
    return prepare_base_model


@pytest.fixture
def model_training_config(temp_dir, downloaded_files):
    """
    Fixture for providing a configuration object for model training.

    Parameters
    ----------
    temp_dir : str
        Temporary directory path.

    Returns
    -------
    ModelTrainingConfig
        Configuration object for model training.
    """
    return ModelTrainingConfig(
        root_dir=temp_dir,
        training_data=os.path.join(downloaded_files.config.unzip_dir, "kidney-ct-scan-image"),
        updated_base_model_path=os.path.join(temp_dir, "updated_base_model"),
        trained_model_path=os.path.join(temp_dir, "trained_model"),
        params_image_size=(224, 224, 3),
        params_batch_size=32,
        params_epochs=1,
        params_is_augmentation=True,
    )


@pytest.fixture
def training(model_training_config):
    """
    Fixture to create an instance of the Training class.
    """
    return Training(model_training_config)
