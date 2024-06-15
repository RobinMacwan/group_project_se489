# -*- coding: utf-8 -*-
from pathlib import Path

import tensorflow as tf

from se489_group_project.model_classes.config_model_classes import CreateBaseModelConfig


class PrepareBaseModel:
    """
    A class used to prepare a base model for further training.

    Attributes
    ----------
    config : CreateBaseModelConfig
        Configuration parameters needed for model preparation.

    Methods
    -------
    get_base_model():
        Loads a pre-trained VGG16 model and saves it to disk.
    update_base_model():
        Prepares a modified version of the base model and saves it to disk.
    save_model(path, model):
        Saves the given model to disk at the specified path.
    """

    def __init__(self, config: CreateBaseModelConfig):
        """
        Initialize PrepareBaseModel with the given configuration.

        Parameters
        ----------
        config : CreateBaseModelConfig
            Configuration instance containing parameters for model preparation.
        """
        self.config = config

    def get_base_model(self):
        """
        Load the base VGG16 model with the specified parameters and save it to disk.

        Uses the configuration parameters provided during initialization
        to load a pre-trained VGG16 model, which is then saved to disk.

        The model will be saved at the path `self.config.base_model_path`.
        """
        self.model = tf.keras.applications.vgg16.VGG16(
            input_shape=self.config.params_image_size,
            weights=self.config.params_weights,
            include_top=self.config.params_include_top,
        )

        self.save_model(path=self.config.base_model_path, model=self.model)

    @staticmethod
    def _prepare_full_model(model, classes, freeze_all, freeze_till, learning_rate):
        """
        Prepare a full model for training based on the base model.

        Parameters
        ----------
        model : tf.keras.Model
            The base model to be updated.
        classes : int
            The number of classes for the final classification layer.
        freeze_all : bool
            Whether to freeze all layers of the base model.
        freeze_till : int
            The number of layers to freeze from the start, if any.
        learning_rate : float
            The learning rate for the model optimizer.

        Returns
        -------
        full_model : tf.keras.Model
            The full model ready for training.

        Notes
        -----
        - If `freeze_all` is True, all layers will be frozen.
        - If `freeze_till` is not None and greater than 0, all layers before `freeze_till` will be frozen.
        - If both conditions are not met, no layers will be frozen.
        """
        if freeze_all:
            for layer in model.layers:
                model.trainable = False
        elif (freeze_till is not None) and (freeze_till > 0):
            for layer in model.layers[:-freeze_till]:
                model.trainable = False

        flatten_in = tf.keras.layers.Flatten()(model.output)
        batch_norm = tf.keras.layers.BatchNormalization()(flatten_in)
        prediction = tf.keras.layers.Dense(units=classes, activation="softmax")(batch_norm)

        full_model = tf.keras.models.Model(inputs=model.input, outputs=prediction)

        full_model.compile(
            optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate),
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=["accuracy"],
        )

        full_model.summary()
        return full_model

    def update_base_model(self):
        """
        Prepare and update the base model for further training.

        The base model is updated by adding new layers, optionally freezing
        some or all layers, and compiling the new model.

        The updated model will be saved at the path specified in `self.config.updated_base_model_path`.
        """
        self.full_model = self._prepare_full_model(
            model=self.model,
            classes=self.config.params_classes,
            freeze_all=True,
            freeze_till=None,
            learning_rate=self.config.params_learning_rate,
        )

        self.save_model(path=self.config.updated_base_model_path, model=self.full_model)

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        """
        Save the given model to disk at the specified path.

        Parameters
        ----------
        path : Path
            The file path where the model should be saved.
        model : tf.keras.Model
            The model to be saved.
        """
        model.save(path)
