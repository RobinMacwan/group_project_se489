"""Module to test the prepare_base_model function in the prepare module."""
import os
import sys
import tempfile

# tests/test_placeholder.py
import pytest
import tensorflow as tf
from pathlib import Path

# Ensure the project root directory is in the Python path, fixes import issues.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from se489_group_project.components.prepare_base_model import PrepareBaseModel
from se489_group_project.model_classes.config_model_classes import CreateBaseModelConfig

@pytest.fixture
def temp_dir():
    """
    Fixture for creating a temporary directory for the duration of the test.
    """
    with tempfile.TemporaryDirectory() as temporary_directory:
        yield Path(temporary_directory)

@pytest.fixture
def base_model_config(temp_dir):
    return CreateBaseModelConfig(
        root_dir=temp_dir,
        base_model_path=temp_dir / "base_model",
        updated_base_model_path=temp_dir / "updated_base_model",
        params_image_size=(224, 224, 3),
        params_weights="imagenet",
        params_include_top=False,
        params_classes=2,
        params_learning_rate=0.001
    
    )
@pytest.fixture
def prepare_base_model(base_model_config):
    """Fixture to create an instance of PrepareBaseModel."""
    return PrepareBaseModel(base_model_config)

def test_get_base_model(prepare_base_model, base_model_config):
    """Test the get_base_model method of PrepareBaseModel.
    
    Parameters
    ----------
    prepare_base_model : PrepareBaseModel
        Instance of PrepareBaseModel to test.
    base_model_config : CreateBaseModelConfig

    
    """
    # Call the get_base_model method to save the base model to disk.
    prepare_base_model.get_base_model()

    # Check that the base model was saved to disk.
    assert base_model_config.base_model_path.exists()
    assert base_model_config.params_classes == 2

def test_initialization(base_model_config):
    """
    Test that PrepareBaseModel initializes correctly.
    """
    prepare_base_model = PrepareBaseModel(base_model_config)
    assert prepare_base_model.config == base_model_config

def test_edge_case(temp_dir):
    """
    Test the get_base_model method with invalid parameters.
    """
    config = CreateBaseModelConfig(
        root_dir=temp_dir,
        base_model_path=temp_dir / "base_model_invalid",
        updated_base_model_path=temp_dir / "updated_base_model_invalid",
        params_image_size=(0, 0, 0),  # Invalid image size
        params_weights=None,  # No weights
        params_include_top=False,
        params_classes=0,  # Invalid number of classes
        params_learning_rate=-0.001  # Invalid learning rate
    )
    prepare_base_model = PrepareBaseModel(config)
    
    with pytest.raises(ValueError):
        prepare_base_model.get_base_model()
