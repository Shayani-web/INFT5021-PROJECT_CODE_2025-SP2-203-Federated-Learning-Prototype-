#!/usr/bin/env python
# -*- coding: utf-8 -*-

# === Standard Python Modules ===
import os   # Used to work with file paths and directories (cross-platform)
import sys  # Provides access to system-specific parameters and functions (not used directly here)

# ----------------------------------------------------------------------------------------
# This script sets up key folder paths for saving logs, model outputs, and Python objects.
# These directories will be created automatically.
# ----------------------------------------------------------------------------------------

# Get the absolute path to the top-level project folder (i.e., one directory above this script)
main_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# === 1. Define path for TensorBoard logs ===
TENSORBOARD_LOG_DIR = os.path.join(main_path, 'logs')  # Path: /project_root/logs

# If the logs directory doesn't exist yet, create it
if not os.path.exists(TENSORBOARD_LOG_DIR):
    os.makedirs(TENSORBOARD_LOG_DIR)

# === 2. Define path for saving trained models or other results ===
SAVE_DIR = os.path.join(main_path, 'save')  # Path: /project_root/save

# Create the directory if it doesn't exist
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

# === 3. Define path for saving pickled (serialized) Python objects ===
SAVE_DIR_PKL = os.path.join(main_path, 'save', 'objects')  # Path: /project_root/save/objects

# Create the directory if it doesn't exist
if not os.path.exists(SAVE_DIR_PKL):
    os.makedirs(SAVE_DIR_PKL)

