# -*- coding: utf-8 -*-
"""Module for testing the model training functionality. Utilizes pytest fixtures to set up the necessary objects for testing in the conftest file."""
import os

import tensorflow as tf


def test_get_base_model(trained_model):

    # Call the method
    trained_model.get_base_model()

    # Assert that the model is loaded correctly
    assert trained_model.model is not None


def test_train_valid_generator(trained_model):

    # Call the method
    trained_model.train_valid_generator()

    # Assert that the generators are created correctly
    assert trained_model.train_generator is not None
    assert trained_model.valid_generator is not None


def test_train(trained_model):

    assert trained_model.train_generator is not None
    assert trained_model.valid_generator is not None
    assert trained_model.train_generator.samples > 1
    assert trained_model.valid_generator.samples > 1
    assert len(trained_model.train_generator.class_indices) > 0
    assert len(trained_model.valid_generator.class_indices) > 0

    # Assertions to verify the process
    assert trained_model.model is not None
    assert os.path.exists(trained_model.config.trained_model_path)
    assert isinstance(trained_model.model, tf.keras.Model)


# For debugging purposes, print all output to the console
# pytest -s tests/model_training_test.py
