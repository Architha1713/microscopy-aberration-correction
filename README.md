# Microscopy AI — Deep Learning-Based Aberration Correction

A deep learning project exploring neural network-based image restoration
for microscopy, with a focus on correcting image degradation caused by
optical aberrations.

This project implements a U-Net-based deep learning pipeline for
microscopy image restoration, including dataset preprocessing,
degradation generation, model training, and quantitative evaluation.

The goal is to investigate how deep learning can improve the quality
of degraded microscopy images and support computational adaptive optics.

---

## Table of Contents

- [Overview](#overview)
- [Research Motivation](#research-motivation)
- [Key Features](#key-features)
- [Project Workflow](#project-workflow)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Evaluation](#evaluation)
- [Results](#results)
- [Technologies Used](#technologies-used)
- [Limitations](#limitations)
- [Future Work](#future-work)
- [Author](#author)

---

## Overview

Optical aberrations and image degradation can reduce the resolution,
contrast, and structural clarity of microscopy images.

This project explores a computational approach to microscopy image
restoration using deep learning.

A U-Net-based neural network is implemented to learn the mapping
between degraded microscopy images and their corresponding
ground-truth images.

The project combines image preprocessing, synthetic degradation,
deep learning-based restoration, and quantitative evaluation into
a unified experimental workflow.

The repository is designed to support experimentation with
microscopy image restoration and provide a foundation for further
research in computational adaptive optics.

## Research Motivation

High-quality microscopy imaging is essential for observing fine
biological structures.

However, optical aberrations and degradation can introduce distortions
that affect the quality and interpretability of acquired images.

Traditional correction techniques may require specialized hardware,
calibration, or computationally intensive processing.

Deep learning offers an alternative approach by learning image
restoration mappings directly from training data.

This project investigates the potential of neural network-based
restoration to improve degraded microscopy images while preserving
important structural details.

## Key Features

- U-Net-based deep learning model for image restoration
- Microscopy dataset preprocessing and preparation
- Synthetic degradation generation for training experiments
- Support for multiple microscopy image categories
- Training and evaluation pipelines implemented in PyTorch
- Quantitative image-quality evaluation
- Visual comparison of degraded, restored, and ground-truth images
- Training-loss visualization and evaluation figure generation

## Project Workflow

The project follows a modular image-restoration workflow.

1. Dataset Preparation
   - Organize microscopy images and ground-truth data.
   - Inspect and validate dataset structure.
   - Prepare training, validation, and testing splits.

2. Image Preprocessing
   - Convert and prepare microscopy image files.
   - Generate degraded image samples.
   - Apply degradation and point spread function (PSF)-related
     processing where configured.

3. Model Training
   - Train a U-Net-based neural network using PyTorch.
   - Optimize the model using the implemented training pipeline.
   - Track training loss and save model checkpoints.

4. Evaluation
   - Evaluate the trained model on held-out microscopy images.
   - Compare restored outputs with corresponding ground-truth images.
   - Generate evaluation summaries and visual comparisons.

5. Result Visualization
   - Generate training curves, loss plots, and image-quality
     comparison figures.

## Model Architecture

The project uses a U-Net-based neural network for image restoration.

U-Net is an encoder-decoder architecture with skip connections,
allowing the network to combine contextual information with
spatial details from the input image.

In microscopy restoration, preserving fine structural information
is particularly important.

The model is implemented in PyTorch.

Implementation:
- `model/unet.py`

Training components:
- `training/train.py`
- `training/dataset.py`
- `training/losses.py`

The architecture and training configuration can be explored
and modified for further experimentation.

## Dataset

The project uses microscopy image data organized into multiple
categories, including:

- Microtubules
- F-actin
- Centrosomes (CCPs)
- Additional microscopy image categories

The dataset includes ground-truth and degraded image data used
for training and evaluation.

### Dataset Availability

The original microscopy datasets are not included in this GitHub
repository because of their size.

Users must obtain the required data separately and organize it
according to the directory structure expected by the preprocessing,
training, and evaluation scripts.

Please ensure that you have the necessary permissions to access
and use the dataset before conducting experiments.

## Repository Structure

```text
Microscopy_AI_Project/
│
├── app/
│   └── app.py
│
├── dataset/                  # Local microscopy datasets (not tracked)
│
├── evaluation/
│   ├── evaluate.py
│   └── generate_paper_figures.py
│
├── model/
│   └── unet.py
│
├── preprocessing/
│   ├── apply_degradation.py
│   ├── cap_dataset.py
│   ├── check_dataset.py
│   ├── convert_and_merge.py
│   ├── convert_mrc.py
│   ├── generate_degraded_dataset.py
│   ├── psf_generator.py
│   ├── sample_bbbc.py
│   └── split_dataset.py
│
├── results/                  # Selected figures and summaries
│
├── testing/
│
├── training/
│   ├── dataset.py
│   ├── losses.py
│   └── train.py
│
├── test_setup.py
├── .gitignore
└── README.md
