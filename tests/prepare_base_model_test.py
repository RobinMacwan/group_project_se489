# -*- coding: utf-8 -*-
"""Module to test the prepare_base_model function in the prepare module."""
import tensorflow as tf


def test_initialization(base_model_config):
    """
    Test that PrepareBaseModel initializes correctly.
    """

    assert base_model_config.base_model_path is not None
    assert base_model_config.updated_base_model_path is not None
    assert base_model_config.params_classes == 2
    assert base_model_config.params_image_size == (224, 224, 3)
    assert base_model_config.params_weights == "imagenet"
    assert base_model_config.params_learning_rate == 0.001


def test_get_base_model(prepared_base_model):
    """Test the get_base_model method of PrepareBaseModel.

    Parameters
    ----------
    prepare_base_model : PrepareBaseModel
        Instance of PrepareBaseModel to test.

    """
    model = tf.keras.models.load_model(prepared_base_model.config.base_model_path)
    assert model is not None
    assert isinstance(model, tf.keras.Model)
