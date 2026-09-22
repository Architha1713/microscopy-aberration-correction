import os
import tifffile
import numpy as np
from apply_degradation import degrade_image

SPLIT_BASE = r"D:\Projects\Microscopy_AI_Project\dataset\split"
OUTPUT_BASE = r"D:\Projects\Microscopy_AI_Project\dataset\degraded"

splits = ["train", "val", "test"]

for split in splits:
    input_folder = os.path.join(SPLIT_BASE, split)
    clean_out = os.path.join(OUTPUT_BASE, split, "clean")
    degraded_out = os.path.join(OUTPUT_BASE, split, "degraded")
    os.makedirs(clean_out, exist_ok=True)
    os.makedirs(degraded_out, exist_ok=True)

    files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.tif', '.tiff'))]
    print(f"Processing {split}: {len(files)} images")

    for i, fname in enumerate(files):
        path = os.path.join(input_folder, fname)
        clean = tifffile.imread(path).astype(np.float64)

        # WIDENED RANGES: covers mild to fairly severe aberration in one dataset,
        # so the model learns to handle a variety of blur strengths -- this is
        # what helps it generalize to images it hasn't seen before, rather than
        # only performing well at one fixed difficulty level.
        aberration_strength = np.random.uniform(1.0, 2.8)
        depth_factor = np.random.uniform(0.05, 0.4)
        noise_level = np.random.uniform(0.005, 0.025)

        degraded, coeffs = degrade_image(clean, aberration_strength, depth_factor, noise_level, seed=i)

        tifffile.imwrite(os.path.join(clean_out, fname), clean.astype(np.float32))
        tifffile.imwrite(os.path.join(degraded_out, fname), degraded.astype(np.float32))

        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(files)} done")

    print(f"{split} complete.\n")

print("All degraded datasets generated. Check dataset/degraded/ folder.")