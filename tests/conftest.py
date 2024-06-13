# -*- coding: utf-8 -*-
"""Module to for shared fixtures."""
import os
import sys
import tempfile
from pathlib import Path
import pytest
import tensorflow as tf

# Ensure the project root directory is in the Python path, fixes import issues.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from se489_group_project.components.prepare_base_model import PrepareBaseModel
from se489_group_project.model_classes.config_model_classes import CreateBaseModelConfig
from se489_group_project.components.data_ingestion import DataIngestion
from se489_group_project.components.model_training import Training
from se489_group_project.model_classes.config_model_classes import GettingDataConfig
from se489_group_project.model_classes.config_model_classes import ModelTrainingConfig
from se489_group_project.utility.common import create_directories, save_json

@pytest.fixture
def temp_dir():
    """
    Fixture for creating a temporary directory for the duration of the test.
    """
    with tempfile.TemporaryDirectory() as temporary_directory:
        yield temporary_directory


@pytest.fixture
def data_ingestion_config(temp_dir):
    """
    Fixture for providing a configuration object for data ingestion.

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


@pytest.fixture
def data_ingestion(data_ingestion_config):
    """
    Fixture for providing a DataIngestion object with the provided configuration.

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


@pytest.fixture
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

@pytest.fixture
def prepare_base_model(base_model_config):
    """Fixture to create an instance of PrepareBaseModel."""
    return PrepareBaseModel(base_model_config)

@pytest.fixture
def model_training_config(temp_dir):
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
        training_data=temp_dir,
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

