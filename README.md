# Microscopy AI — Neural Aberration Correction

A deep learning research project exploring **computational aberration correction for two-photon microscopy** using a physics-grounded synthetic degradation pipeline and a Residual U-Net.

The project investigates whether optical aberrations and depth-related image degradation can be corrected computationally without requiring additional adaptive-optics hardware or per-image optimization.

> **Research Status:** Research project / manuscript in preparation  
> **Framework:** PyTorch  
> **Architecture:** Residual U-Net  
> **Domain:** Computational Microscopy · Deep Learning · Image Restoration · Computational Adaptive Optics

---

## Overview

Two-photon microscopy is widely used for imaging biological structures at depth, but optical aberrations caused by refractive-index variations can blur fine structures and reduce image quality.

Traditional adaptive optics can correct these distortions using specialized hardware such as deformable mirrors and wavefront sensors. This project explores a software-based alternative: learning to correct aberrations directly from degraded microscopy images.

The core idea is to:

1. Start with clean microscopy images.
2. Simulate physically motivated optical aberrations.
3. Add depth-dependent scattering and sensor noise.
4. Train a neural network to recover the original image.
5. Evaluate restoration quality using PSNR and SSIM.

Unlike a generic blur-based restoration pipeline, the degradation process is based on **Zernike wavefront aberrations and Fourier-optics-derived point spread functions (PSFs)**.

---

## Research Motivation

Deep-tissue microscopy can suffer from increasingly severe image degradation as imaging depth increases.

Hardware-based adaptive optics can address these distortions, but it introduces additional equipment, calibration requirements, and operational complexity.

This project investigates a lighter-weight computational approach:

> Can a neural network learn to correct physically simulated optical aberrations in a single forward pass, without paired real aberrated images or specialized adaptive-optics hardware?

The objective is **not to replace hardware adaptive optics**, but to investigate whether computational correction can provide a useful improvement for laboratories without access to specialized correction hardware.

---

## Key Contributions

### 1. Physics-Grounded Degradation Pipeline

Instead of applying a generic Gaussian blur, the project simulates optical degradation using:

- Randomized Zernike aberration coefficients
- Fourier-optics-based PSF generation
- Depth-dependent scattering
- Sensor noise
- FFT-based image convolution

The aberration strength, depth factor, and noise level are randomized across training samples to expose the model to a range of degradation conditions.

---

### 2. Residual U-Net

The restoration network is based on a U-Net encoder-decoder architecture with skip connections.

Instead of predicting the complete restored image directly, the network predicts a **residual correction** which is added to the degraded input.

Architecture highlights:

- 4 encoder stages
- 4 decoder stages
- DoubleConv blocks
- Batch normalization
- Transposed convolutions
- Skip connections
- Bottleneck dropout
- Residual output formulation

The final model uses a base channel configuration of 32.

---

### 3. Combined Restoration Loss

The training objective combines three complementary components:

- **L1 loss** — reduces pixel-level reconstruction error
- **SSIM loss** — encourages structural similarity
- **Sobel edge loss** — helps preserve fine boundaries and structures

The final configuration uses:

```text
λ1 = 0.5
λSSIM = 0.3
λedge = 0.2
