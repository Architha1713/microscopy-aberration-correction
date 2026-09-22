import torch

import cv2

import numpy as np

import matplotlib

import skimage

print("PyTorch version:", torch.__version__)

print("GPU available:", torch.cuda.is_available())

print("GPU name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU found")

print("All libraries loaded successfully. You are ready for Phase 2.")
