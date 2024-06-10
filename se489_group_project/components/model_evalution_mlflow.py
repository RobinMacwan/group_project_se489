# -*- coding: utf-8 -*-
from pathlib import Path
from urllib.parse import urlparse

import mlflow
import mlflow.keras
import tensorflow as tf

from se489_group_project.model_classes.config_model_classes import ModelEvaluationConfig
from se489_group_project.utility.common import create_directories, read_yaml, save_json

# from prometheus_client import start_http_server, Summary, Gauge
# import pdb #import for debugging


class Evaluation:
    """
    A class used to evaluate the pre-trained model.

    Attributes
    ----------
    config : ModelEvaluationConfig
        Configuration object with parameters to be used in the evaluation process.

    Methods
    -------
    _valid_generator()
        Sets up the validation data generator.
    load_model(path: Path) -> tf.keras.Model
        Loads and returns the pre-trained Keras model.
    evaluation()
        Loads the model, evaluates it using the validation data generator, and saves the score.
    save_score()
        Saves the evaluation scores (loss and accuracy) to a JSON file.
    log_into_mlflow()
        Logs evaluation parameters and metrics into MLflow for tracking.
    """

    def __init__(self, config: ModelEvaluationConfig):
        """
        Initialize the ModelEvaluation class with the configuration.

        Parameters
        ----------
        config : ModelEvaluationConfig
            Configuration object with parameters to be used in the evaluation process.

        """

        self.config = config

        # # Initialize Prometheus metrics
        # self.evaluation_time = Summary('evaluation_processing_seconds', 'Time spent evaluating the model')
        # self.model_loss = Gauge('model_loss', 'Model evaluation loss')
        # self.model_accuracy = Gauge('model_accuracy', 'Model evaluation accuracy')

        # # Start Prometheus HTTP server
        # start_http_server(8000)

    def _valid_generator(self):
        """
        Set up a validation data generator using TensorFlow's ImageDataGenerator.

        The validation generator will rescale image values and create a
        data flow from the directory specified in the configuration.
        The 'validation_split' and 'subset' parameters handle partitioning
        within a single dataset.
        """

        # Keyword arguments for the image data generator
        datagenerator_kwargs = dict(rescale=1.0 / 255, validation_split=0.30)

        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            interpolation="bilinear",
        )

        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(**datagenerator_kwargs)

        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="validation",
            shuffle=False,
            **dataflow_kwargs,
        )

    @staticmethod
    def load_model(path: Path) -> tf.keras.Model:
        """
        Load and return a pre-trained Keras model from the specified file path.

        Parameters
        ----------
        path : Path
            Path to the saved model file.

        Returns
        -------
        tf.keras.Model
            The loaded Keras model.

        """
        return tf.keras.models.load_model(path)

    def evaluation(self):
        """
        Load the model, evaluate it using the validation data generator, and save the score.

        Calls `load_model` to retrieve the pre-trained model and `_valid_generator`
        to set up the validation data generator. After evaluating the model,
        scores are saved using `save_score`.
        """
        # breakpoint for debugging
        # pdb.set_trace()
        self.model = self.load_model(self.config.path_of_model)
        self._valid_generator()
        self.score = self.model.evaluate(self.valid_generator)
        self.save_score()
        self.log_into_mlflow()

    def save_score(self):
        """
        Save the evaluation scores ,loss and accuracy, to a JSON file.

        The scores are stored as a dictionary in "scores.json" using
        `save_json`.
        """
        scores = {"loss": self.score[0], "accuracy": self.score[1]}
        save_json(path=Path("scores.json"), data=scores)

    def log_into_mlflow(self):
        """
        Log evaluation parameters and metrics into MLflow for tracking.

        It initializes an MLflow run, logs parameters and metrics,
        and registers the model if a supported tracking URI is used.
        """
        mlflow.set_registry_uri(self.config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        with mlflow.start_run():
            mlflow.log_params(self.config.all_params)
            mlflow.log_metrics({"loss": self.score[0], "accuracy": self.score[1]})
            # Model registry does not work with file store
            if tracking_url_type_store != "file":

                # Register the model
                # There are other ways to use the Model Registry, which depends on the use case,
                # please refer to the doc for more information:
                # https://mlflow.org/docs/latest/model-registry.html#api-workflow
                mlflow.keras.log_model(self.model, "model", registered_model_name="VGG16Model")
            else:
                mlflow.keras.log_model(self.model, "model")
