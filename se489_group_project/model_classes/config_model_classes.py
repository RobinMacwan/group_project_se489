# -*- coding: utf-8 -*-
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class GettingDataConfig:
    """
    A class used to store the configuration for the data ingestion process.
    """

    local_data_file: Path
    root_dir: Path
    unzip_dir: Path
    source_URL: str


@dataclass(frozen=True)
class CreateBaseModelConfig:
    """
    A class used to store the configuration for the base model creation process.
    """

    params_include_top: bool
    root_dir: Path
    updated_base_model_path: Path
    base_model_path: Path
    params_image_size: list
    params_classes: int
    params_learning_rate: float
    params_weights: str


@dataclass(frozen=True)
class ModelTrainingConfig:
    """
    A class used to store the configuration for the model training process.
    """

    params_batch_size: int
    root_dir: Path
    params_image_size: list
    trained_model_path: Path
    params_is_augmentation: bool
    updated_base_model_path: Path
    training_data: Path
    params_epochs: int


@dataclass(frozen=True)
class ModelEvaluationConfig:
    """
    A class used to store the configuration for the model evaluation process.
    """

    mlflow_uri: str
    path_of_model: Path
    all_params: dict
    training_data: Path
    params_image_size: list
    params_batch_size: int
