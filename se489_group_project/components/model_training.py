import os
import urllib.request as request
from zipfile import ZipFile
import tensorflow as tf
import time
from pathlib import Path
from se489_group_project.model_classes.config_model_classes import ModelTrainingConfig


class Training:
    """
    Class used to train a model using the provided configuration.

    Parameters
    ----------
    config : ModelTrainingConfig
        Configuration object with parameters to be used in the training process.
    
    """
    def __init__(self, config: ModelTrainingConfig):
        """
        Initialize the ModelTraining class with the configuration.

        Parameters
        ----------
        config : ModelTrainingConfig
            Configuration object with parameters to be used in the training process.
        
    
        """
        self.config = config

    
    def get_base_model(self):
        """
        Load the previousl trained base model from the saved path specified in the configuration.
        Sets `self.model` to the loaded model using the path in `self.config.updated_base_model_path`.
        """
        self.model = tf.keras.models.load_model(
            self.config.updated_base_model_path
        )

    def train_valid_generator(self):
        """
        Set up data generators for training and validation.

        Uses different image data generators to augment and rescale training images and only rescale validation images.
        The training generator uses data augmentation if enabled in the configuration and sets data flows for `self.train_generator` and `self.valid_generator`
        attributes.
        """

        datagenerator_kwargs = dict(
            rescale = 1./255,
            validation_split=0.20
        )

        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            interpolation="bilinear"
        )

        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs
        )

        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="validation",
            shuffle=False,
            **dataflow_kwargs
        )

        if self.config.params_is_augmentation:
            train_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
                rotation_range=40,
                horizontal_flip=True,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                **datagenerator_kwargs
            )
        else:
            train_datagenerator = valid_datagenerator

        self.train_generator = train_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="training",
            shuffle=True,
            **dataflow_kwargs
        )

    
    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        """Save model to the path that is specified.
        
        Parameters
        ----------
        path : Path
            Path to save the model.
        model : tf.keras.Model
            Model to be saved.
            """
        model.save(path)



    
    def train(self):
        """
        Trains the model utilizing training and validation data generators.

        Calculates the number of steps per epoch for both training and validation,
        and then trains the model for the number of previously specified epochs.

        Saves the trained model to `self.config.trained_model_path`.
        """
        self.steps_per_epoch = self.train_generator.samples // self.train_generator.batch_size
        self.validation_steps = self.valid_generator.samples // self.valid_generator.batch_size

        self.model.fit(
            self.train_generator,
            epochs=self.config.params_epochs,
            steps_per_epoch=self.steps_per_epoch,
            validation_steps=self.validation_steps,
            validation_data=self.valid_generator
        )

        self.save_model(
            path=self.config.trained_model_path,
            model=self.model
        )