# -*- coding: utf-8 -*-
"""
This module is responsible for profiling the code and visualizing the profile data.
"""
import cProfile
import os
import pstats
import subprocess
from pathlib import Path

from se489_group_project import logger

# Configure paths
BASE_DIR = Path(__file__).resolve().parent
PROFILE_DIR = "se489_group_project/visualizations/profiles"
VISUALIZATION_DIR = "se489_group_project/visualizations/outputs"


def start_profiler():
    """
    Start the profiler.
    Returns the profiler object.
    """
    profiler = cProfile.Profile()
    profiler.enable()
    return profiler


def stop_profiler(profiler, filename):
    """
    Stop the profiler and save the profile data to a file.
    Prints the top functions by cumulative and total time.
    Returns the path to the profile file.

    Attributes:
    -----------
    profiler : cProfile.Profile
        The profiler object.
    filename : str
        The name of the file to save the profile data to.

    Returns:
    --------
    profile_file: str
        The path to the profile file.

    """
    profiler.disable()
    os.makedirs(PROFILE_DIR, exist_ok=True)
    profile_file = os.path.join(PROFILE_DIR, f"{filename}.prof")
    # save the profile data to file
    profiler.dump_stats(profile_file)
    return profile_file


def visualize_profile(profile_file):
    """
    Visualize the profile data using snakeviz .

    Attributes:
    -----------
    profile_file : str
        The path to the profile data file.

    """
    try:
        subprocess.Popen(["snakeviz", profile_file])
    except FileNotFoundError:
        print("snakeviz is not installed or not found in the system path.")
