# -*- coding: utf-8 -*-
"""
This module contains the configuration for the visualization process.
"""
import os
from pathlib import Path

# Configure paths
BASE_DIR = Path(__file__).resolve().parent
# set the path to the profile directory ./visulizations/profiles
PROFILE_DIR = BASE_DIR / "profiles"
# set the path to the visualization directory ./visualizations/outputs
VISUALIZATION_DIR = BASE_DIR / "outputs"

# Ensure directories exist for profile and visualization outputs
PROFILE_DIR.mkdir(parents=True, exist_ok=True)
VISUALIZATION_DIR.mkdir(parents=True, exist_ok=True)
