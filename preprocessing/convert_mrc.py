import os
import mrcfile
import tifffile
import numpy as np

# EDIT THIS to point to the folder containing your .mrc files
INPUT_PATH = r"D:\Projects\Microscopy_AI_Project\dataset\F-actin"
OUTPUT_PATH = r"D:\Projects\Microscopy_AI_Project\dataset\F-actin_tif"

os.makedirs(OUTPUT_PATH, exist_ok=True)

mrc_files = []
for root, dirs, files in os.walk(INPUT_PATH):
    for f in files:
        if f.lower().endswith('.mrc'):
            mrc_files.append(os.path.join(root, f))

print(f"Found {len(mrc_files)} .mrc files.")

for i, path in enumerate(mrc_files):
    with mrcfile.open(path, permissive=True) as mrc:
        data = mrc.data  # this is a numpy array

    # Some .mrc files store multiple images stacked together (a 3D array).
    # If so, save each slice as its own .tif file.
    filename = os.path.splitext(os.path.basename(path))[0]

    if data.ndim == 2:
        out_path = os.path.join(OUTPUT_PATH, f"{filename}.tif")
        tifffile.imwrite(out_path, data)
    elif data.ndim == 3:
        for slice_idx in range(data.shape[0]):
            out_path = os.path.join(OUTPUT_PATH, f"{filename}_slice{slice_idx}.tif")
            tifffile.imwrite(out_path, data[slice_idx])

    if (i + 1) % 20 == 0:
        print(f"Converted {i+1}/{len(mrc_files)}...")

print("Conversion done. Check the output folder:", OUTPUT_PATH)