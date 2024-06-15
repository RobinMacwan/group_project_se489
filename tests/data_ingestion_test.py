# -*- coding: utf-8 -*-
"""This module contains the tests for the data module. Uses pytest fixtures to setup the tests.
 Temporary directories are created for the tests to ensure that the tests are isolated from the rest of the system.
 The tests check that the data is downloaded correctly and that the file exists and is not empty."""
import os


def test_downloaded_file(downloaded_files):
    """Test for the download_file and extract_zip_file methods of the DataIngestion class.
    Ensure that the file is downloaded and unzipped correctly.
    """
    dataset = downloaded_files.config.local_data_file
    unzipped = downloaded_files.config.unzip_dir

    # Assert that the file exists
    assert os.path.exists(dataset), "Download failed: file does not exist."

    # Assert that the file is not empty
    assert os.path.getsize(dataset) > 0, "Download failed: file is empty."

    # Assert that the file name ends with .zip
    assert dataset.endswith(".zip"), "Download failed: file is not a zip file."

    # Assert that the extraction directory exists
    assert os.path.exists(unzipped), "Extraction failed: directory does not exist."

    # Assert that the extraction directory is not empty
    assert len(os.listdir(unzipped)) > 0, "Extraction failed: directory is empty."

    # Additional checks for expected directories within 'kidney-ct-scan-image'
    kidney_ct_scan_image = os.path.join(unzipped, "kidney-ct-scan-image")
    assert os.path.exists(kidney_ct_scan_image), "Extraction failed: 'ct-scan' directory does not exist."

    expected_directories = ["Tumor", "Normal"]  # Expected directories within 'kidney-ct-scan-image'
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
