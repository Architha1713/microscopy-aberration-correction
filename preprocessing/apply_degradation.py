import numpy as np

import tifffile

from scipy.signal import fftconvolve

from psf_generator import generate_zernike_psf

def add_depth_scattering(image, depth_factor=0.3):

    """

    Simulates depth-dependent scattering: deeper tissue causes more attenuation (dimming)

    and additional soft blur, on top of the sharp PSF-based aberration.

    depth_factor: 0 = no extra scattering, 1 = heavy scattering

    """

    # Attenuate brightness (light loses energy traveling through tissue)

    attenuated = image * (1 - depth_factor * 0.4)

    return attenuated

def add_noise(image, noise_level=0.02):

    """

    Adds realistic sensor/shot noise, since real microscopy images are never noise-free.

    """

    noise = np.random.normal(0, noise_level * image.max(), image.shape)

    noisy = image + noise

    return np.clip(noisy, 0, image.max())

def degrade_image(clean_image, aberration_strength=1.5, depth_factor=0.3, noise_level=0.02, seed=None):

    """

    Full degradation pipeline: PSF blur (optical aberration) -> depth scattering -> noise.

    Returns a degraded image the same size as the input.

    """

    clean_image = clean_image.astype(np.float64)

    h, w = clean_image.shape

    # Generate a PSF the same size as the image

    psf, coeffs = generate_zernike_psf(image_size=h, num_zernike_terms=15,

                                        aberration_strength=aberration_strength, seed=seed)

    # Convolve image with PSF (this is the actual optical blurring step)

    blurred = fftconvolve(clean_image, psf, mode='same')

    # Add depth-dependent scattering

    scattered = add_depth_scattering(blurred, depth_factor=depth_factor)

    # Add sensor noise

    degraded = add_noise(scattered, noise_level=noise_level)

    return degraded, coeffs


if __name__ == "__main__":

    import matplotlib.pyplot as plt

    # Load one real sample image from your dataset

    sample_path = r"D:\Projects\Microscopy_AI_Project\dataset\F-actin_tif\GT_all_a_slice0.tif"

    clean = tifffile.imread(sample_path)

    degraded, coeffs = degrade_image(clean, aberration_strength=1.5, depth_factor=0.3, noise_level=0.02, seed=42)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].imshow(clean, cmap='gray')

    axes[0].set_title("Clean (Ground Truth)")

    axes[0].axis('off')

    axes[1].imshow(degraded, cmap='gray')

    axes[1].set_title("Degraded (Simulated Aberration)")

    axes[1].axis('off')

    plt.tight_layout()

    plt.savefig(r"D:\Projects\Microscopy_AI_Project\dataset\sample_degradation_comparison.png")

    print("Saved sample_degradation_comparison.png — compare the two images side by side.")
