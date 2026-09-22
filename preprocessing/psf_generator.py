import numpy as np

import aotools

def generate_zernike_psf(image_size=256, num_zernike_terms=15, aberration_strength=1.0, seed=None):

    """

    Generates a Point Spread Function (PSF) based on random Zernike aberration coefficients.

    image_size: size of the PSF array (should be same size or smaller than your images)

    num_zernike_terms: how many Zernike modes to combine (more = more complex aberration)

    aberration_strength: overall intensity of the distortion (higher = blurrier)

    seed: set this for reproducible results (same seed = same aberration every time)

    """

    if seed is not None:

        np.random.seed(seed)

    # Generate random coefficients for each Zernike term.

    # Real aberrations are typically dominated by lower-order terms (defocus, astigmatism),

    # so we scale down the influence of higher-order terms to keep it physically realistic.

    coeffs = np.random.randn(num_zernike_terms)
    decay = np.array([1.0 / np.sqrt(i + 1) for i in range(num_zernike_terms)])
    coeffs = coeffs * decay * aberration_strength * 3.0  # extra multiplier so strength actually shows up spatially

    # Build the combined wavefront from these Zernike terms

    zernike_basis = aotools.zernikeArray(num_zernike_terms, image_size, norm='rms')

    wavefront = np.zeros((image_size, image_size))

    for i in range(num_zernike_terms):

        wavefront += coeffs[i] * zernike_basis[i]

    # Create a circular pupil mask (light only passes through a circular aperture, like a real lens)

    y, x = np.ogrid[-image_size//2:image_size//2, -image_size//2:image_size//2]

    pupil_mask = (x**2 + y**2) <= (image_size // 2) ** 2

    # Convert wavefront (phase) into a complex pupil function

    pupil_function = pupil_mask * np.exp(1j * wavefront)

    # The PSF is the squared magnitude of the Fourier transform of the pupil function

    # (this is standard optics: how a lens turns a wavefront into an image)

    psf = np.abs(np.fft.fftshift(np.fft.fft2(pupil_function))) ** 2

    # Normalize so the PSF sums to 1 (so it doesn't change overall image brightness)

    psf = psf / psf.sum()

    return psf, coeffs


if __name__ == "__main__":

    import matplotlib.pyplot as plt

    psf, coeffs = generate_zernike_psf(image_size=256, num_zernike_terms=15, aberration_strength=2.5, seed=42)

    plt.imshow(psf, cmap='hot')

    plt.title("Simulated PSF from Zernike Aberrations")

    plt.colorbar()

    plt.savefig(r"D:\Projects\Microscopy_AI_Project\dataset\sample_psf.png")

    print("Saved sample_psf.png — this is what a distorted 'point of light' looks like.")

    print("Zernike coefficients used:", coeffs)
