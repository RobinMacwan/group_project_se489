from se489_group_project.constants import *
import os
from se489_group_project.utility.common import read_yaml, create_directories, save_json
from se489_group_project.model_classes.config_model_classes import (GettingDataConfig, 
                                                                    CreateBaseModelConfig,
                                                                    ModelTrainingConfig, 
                                                                    ModelEvaluationConfig)


class ConfigurationManager:
    def __init__(
            self,
            config_filepath=CONFIG_FILE_PATH,
            params_filepath=PARAMS_FILE_PATH):
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.data_storage])

    def get_data_ingestion_config(self) -> GettingDataConfig:
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config = GettingDataConfig(
            root_dir=config.root_dir,
            source_URL=config.source_URL,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir
        )

        return data_ingestion_config

    def get_prepare_base_model_config(self) -> CreateBaseModelConfig:
        config = self.config.prepare_base_model

        create_directories([config.root_dir])

        prepare_base_model_config = CreateBaseModelConfig(
            root_dir=Path(config.root_dir),
            base_model_path=Path(config.base_model_path),
            updated_base_model_path=Path(config.updated_base_model_path),
            params_image_size=self.params.IMAGE_SIZE,
            params_learning_rate=self.params.LEARNING_RATE,
            params_include_top=self.params.INCLUDE_TOP,
            params_weights=self.params.WEIGHTS,
            params_classes=self.params.CLASSES
        )

        return prepare_base_model_config

    def get_training_config(self) -> ModelTrainingConfig:
        training = self.config.training
        prepare_base_model = self.config.prepare_base_model
        params = self.params
        training_data = os.path.join(self.config.data_ingestion.unzip_dir, "kidney-ct-scan-image")
        create_directories([
            Path(training.root_dir)
        ])

        training_config = ModelTrainingConfig(
            root_dir=Path(training.root_dir),
            trained_model_path=Path(training.trained_model_path),
            updated_base_model_path=Path(prepare_base_model.updated_base_model_path),
            training_data=Path(training_data),
            params_epochs=params.EPOCHS,
            params_batch_size=params.BATCH_SIZE,
            params_is_augmentation=params.AUGMENTATION,
            params_image_size=params.IMAGE_SIZE
        )

        return training_config

    def get_evaluation_config(self) -> ModelEvaluationConfig:
        eval_config = ModelEvaluationConfig(
            path_of_model="data/training/model.h5",
            training_data="data/raw/kidney-ct-scan-image",
            mlflow_uri="https://dagshub.com/RobinMacwan/group_project.mlflow",
            all_params=self.params,
            params_image_size=self.params.IMAGE_SIZE,
            params_batch_size=self.params.BATCH_SIZE
        )
        return eval_config
