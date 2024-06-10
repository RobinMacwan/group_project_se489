# -*- coding: utf-8 -*-
"""
This module contains the logger configuration for the project.

It sets up the logging format, directory, and handlers for logging messages to both
a file and the console.
"""

import logging
import os
import sys

# Logging format string
logging_str = "[%(asctime)s: %(levelname)s: %(module)s: %(message)s]"

# Directory for log files
log_dir = "logs"

# File path for the log file
log_filepath = os.path.join(log_dir, "running_logs.log")

# Create the log directory if it does not exist
os.makedirs(log_dir, exist_ok=True)

# Configure the logging settings
logging.basicConfig(
    level=logging.INFO,
    format=logging_str,
    handlers=[
        logging.FileHandler(log_filepath),  # Log to file
        logging.StreamHandler(sys.stdout),  # Log to console
    ],
)

# Create a logger object
logger = logging.getLogger("grp_proj_Logger")
