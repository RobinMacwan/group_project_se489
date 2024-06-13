from pathlib import Path
import pytest
import tensorflow as tf
import pytest
import os


def test_get_base_model(training, prepare_base_model):
    #First, prepare the base model
    prepare_base_model.get_base_model()
    prepare_base_model.update_base_model()

    # Call the method
    training.get_base_model()

    # Assert that the model is loaded correctly
    assert training.model is not None

def test_train_valid_generator(training, data_ingestion):
    # Perform data ingestion to set up the training data
    data_ingestion.download_file()
    data_ingestion.extract_zip_file()
    
    # Call the method
    training.train_valid_generator()

    # Print out some details about the generators for debugging
    print(f"Training data shape: {training.train_generator.image_shape}")
    print(f"Validation data shape: {training.valid_generator.image_shape}")
    
    # Assert that the generators are created correctly
    assert training.train_generator is not None
    assert training.valid_generator is not None
    
# def test_train(training, prepare_base_model, data_ingestion):
#Working on this test


# def test_save_model(training, prepare_base_model, temp_dir):
#Working on this test

# # Additional tests can be added for the train method and other functionality
