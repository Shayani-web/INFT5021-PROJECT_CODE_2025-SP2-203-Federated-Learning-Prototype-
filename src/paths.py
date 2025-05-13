import os
import sys

main_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TENSORBOARD_LOG_DIR = os.path.join(main_path, 'logs')
if not os.path.exists(TENSORBOARD_LOG_DIR):
    os.makedirs(TENSORBOARD_LOG_DIR)

SAVE_DIR = os.path.join(main_path, 'save')
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)


SAVE_DIR_PKL = os.path.join(main_path, 'save','objects')
if not os.path.exists(SAVE_DIR_PKL):
    os.makedirs(SAVE_DIR_PKL)

