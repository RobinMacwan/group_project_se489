# -*- coding: utf-8 -*-
"""This module contains the tests for the data module. Uses pytest fixtures to setup the tests.
 Temporary directories are created for the tests to ensure that the tests are isolated from the rest of the system.
 The tests check that the data is downloaded correctly and that the file exists and is not empty."""
import os
import sys
import tempfile

# tests/test_placeholder.py
import pytest

# Ensure the project root directory is in the Python path, fixes import issues.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from se489_group_project.components.data_ingestion import DataIngestion
from se489_group_project.model_classes.config_model_classes import GettingDataConfig


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


def test_downloaded_file(data_ingestion):
    """Test for the download_file and extract_zip_file methods of the DataIngestion class.
    Ensure that the file is downloaded and unzipped correctly.
    """
    # Call the download_file method
    data_ingestion.download_file()
    dataset = data_ingestion.config.local_data_file
    unzipped = data_ingestion.config.unzip_dir

    # Assert that the file exists
    assert os.path.exists(dataset), "Download failed: file does not exist."

    # Assert that the file is not empty
    assert os.path.getsize(dataset) > 0, "Download failed: file is empty."

    # Assert that the file name ends with .zip
    assert dataset.endswith(".zip"), "Download failed: file is not a zip file."

    # Call the extract_zip_file method to extract the downloaded file
    data_ingestion.extract_zip_file()

    # Assert that the extraction directory exists
    assert os.path.exists(unzipped), "Extraction failed: directory does not exist."

    # Assert that the extraction directory is not empty
    assert len(os.listdir(unzipped)) > 0, "Extraction failed: directory is empty."

    # Additional checks for expected directories within 'ct-scan'
    kidney_ct_scan_image = os.path.join(unzipped, "kidney-ct-scan-image")
    assert os.path.exists(kidney_ct_scan_image), "Extraction failed: 'ct-scan' directory does not exist."

    expected_directories = ["Tumor", "Normal"]  # Expected directories within 'ct-scan'
    for expected_directory in expected_directories:
        image_directory = os.path.join(kidney_ct_scan_image, expected_directory)
        assert os.path.exists(image_directory), f"Missing expected directory: {expected_directory}"

        # Ensure directories are not empty
        assert len(os.listdir(image_directory)) > 0, f"Directory is empty: {expected_directory}"

        # Check for file type
        file_types = [".jpg"]
    for file_type in file_types:
        files = [f for f in os.listdir(image_directory) if f.endswith(file_type)]
        assert len(files) > 0, f"No files with {file_type} extension found in {expected_directory}"

        # Update 'min_file_count' based on your expectations
        min_file_count = 40
        assert len(os.listdir(image_directory)) >= min_file_count, f"Insufficient files in {expected_directory}"
