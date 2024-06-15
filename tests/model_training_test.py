# -*- coding: utf-8 -*-
"""Module for testing the model training functionality. Utilizes pytest fixtures to set up the necessary objects for testing in the conftest file."""
import os

import tensorflow as tf


def test_get_base_model(training, prepare_base_model):
    # First, prepare the base model
    prepare_base_model.get_base_model()
    prepare_base_model.update_base_model()

    # Call the method
    training.get_base_model()

    # Assert that the model is loaded correctly
    assert training.model is not None


def test_train_valid_generator(training):

    # Call the method
    training.train_valid_generator()

    # Assert that the generators are created correctly
    assert training.train_generator is not None
    assert training.valid_generator is not None


def test_train(training, prepare_base_model):

    # First, prepare the base model
    prepare_base_model.get_base_model()
    prepare_base_model.update_base_model()
    training.get_base_model()

    training.train_valid_generator()
    assert training.train_generator is not None
    assert training.valid_generator is not None
    assert training.train_generator.samples > 1
    assert training.valid_generator.samples > 1
    assert len(training.train_generator.class_indices) > 0
    assert len(training.valid_generator.class_indices) > 0

    training.train()

    # Assertions to verify the process
    assert training.model is not None
    assert os.path.exists(training.config.trained_model_path)
    assert isinstance(training.model, tf.keras.Model)


# For debugging purposes, print all output to the console
# pytest -s tests/model_training_test.py
