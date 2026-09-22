import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import gradio as gr
import numpy as np
import torch
import tifffile
from PIL import Image
from scipy.ndimage import gaussian_filter

from model.unet import UNet

MODEL_PATH = r"D:\Projects\Microscopy_AI_Project\results\best_model.pth"
PATCH_SIZE = 256

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet(in_channels=1, out_channels=1, base_channels=32).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()
print(f"Model loaded on {device}")


def unsharp_mask(image, amount=1.5, sigma=1.0):
    blurred = gaussian_filter(image, sigma=sigma)
    return np.clip(image + amount * (image - blurred), 0, 1)


def predict_with_tta(model, img):
    variants = [
        (img, lambda a: a),
        (np.fliplr(img).copy(), lambda a: np.fliplr(a)),
        (np.flipud(img).copy(), lambda a: np.flipud(a)),
        (np.fliplr(np.flipud(img)).copy(), lambda a: np.fliplr(np.flipud(a))),
    ]
    outputs = []
    for variant, unflip in variants:
        tensor = torch.from_numpy(variant).unsqueeze(0).unsqueeze(0).float().to(device)
        with torch.no_grad():
            out = model(tensor).squeeze().cpu().numpy()
        outputs.append(unflip(out))
    return np.mean(outputs, axis=0)


def load_any_image(file_path):
    """Accepts .tif/.tiff via tifffile, or standard formats (.png/.jpg) via PIL."""
    ext = os.path.splitext(file_path)[1].lower()
    if ext in ['.tif', '.tiff']:
        img = tifffile.imread(file_path).astype(np.float32)
        if img.ndim == 3:
            img = img[..., 0]  # take first channel if RGB-like
    else:
        img = np.array(Image.open(file_path).convert('L')).astype(np.float32)
    return img


def correct_image(uploaded_file):
    if uploaded_file is None:
        return None, None, "Please upload an image."

    img = load_any_image(uploaded_file)
    max_val = img.max() if img.max() > 0 else 1.0
    img_norm = img / max_val

    h, w = img_norm.shape

    # Resize to a multiple of 16 (U-Net requirement) using center-crop or pad
    def prep(im, size=PATCH_SIZE):
        h, w = im.shape
        if h >= size and w >= size:
            top, left = (h - size) // 2, (w - size) // 2
            return im[top:top+size, left:left+size]
        else:
            pad_h, pad_w = max(0, size - h), max(0, size - w)
            im = np.pad(im, ((0, pad_h), (0, pad_w)))
            return im[:size, :size]

    input_crop = prep(img_norm)
    corrected = predict_with_tta(model, input_crop)
    corrected_sharp = unsharp_mask(corrected, amount=1.5, sigma=1.0)

    input_display = (input_crop * 255).astype(np.uint8)
    output_display = (corrected_sharp * 255).astype(np.uint8)

    info = (f"Processed on {device}. Input size cropped/padded to {PATCH_SIZE}x{PATCH_SIZE}. "
            f"Correction uses test-time augmentation (4-view averaging) and unsharp masking for display.")

    return Image.fromarray(input_display), Image.fromarray(output_display), info


# ---- Custom theme for a polished look ----
theme = gr.themes.Soft(
    primary_hue="teal",
    secondary_hue="slate",
).set(
    body_background_fill="*neutral_50",
)

with gr.Blocks(theme=theme, title="Microscopy Aberration Correction") as demo:
    gr.Markdown(
        """
        # Self-Supervised Deep-Tissue Microscopy Aberration Correction
        Upload a blurry/aberrated two-photon or fluorescence microscopy image (.tif, .png, .jpg).
        The model will attempt to correct optical aberrations using a residual U-Net trained
        with a physics-based degradation model and test-time augmentation.
        """
    )
    with gr.Row():
        file_input = gr.File(label="Upload microscopy image", file_types=[".tif", ".tiff", ".png", ".jpg", ".jpeg"])
    run_btn = gr.Button("Run Correction", variant="primary")
    with gr.Row():
        input_out = gr.Image(label="Input (degraded)", type="pil")
        output_out = gr.Image(label="Corrected Output", type="pil")
    info_out = gr.Textbox(label="Details", interactive=False)

    run_btn.click(fn=correct_image, inputs=file_input, outputs=[input_out, output_out, info_out])

    gr.Markdown(
        """
        ---
        **Note:** This model was trained on BioSR and BBBC005 microscopy datasets with simulated
        physics-based aberrations (PSF derived from Zernike coefficients). Performance on
        real-world unseen images may vary. Reported test-set results: PSNR +2.00 dB, SSIM +0.079
        improvement over degraded input.
        """
    )

demo.launch()