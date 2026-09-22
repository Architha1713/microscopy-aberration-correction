import os

import mrcfile

import tifffile

import numpy as np

# List every raw category folder you've downloaded (add more if you get them)

CATEGORY_FOLDERS = [

    r"D:\Projects\Microscopy_AI_Project\dataset\F-actin",

    r"D:\Projects\Microscopy_AI_Project\dataset\Microtubules",

    r"D:\Projects\Microscopy_AI_Project\dataset\CCPs",

]

OUTPUT_PATH = r"D:\Projects\Microscopy_AI_Project\dataset\all_categories_tif"

os.makedirs(OUTPUT_PATH, exist_ok=True)

total_converted = 0

for category_folder in CATEGORY_FOLDERS:

    if not os.path.exists(category_folder):

        print(f"Skipping (not found): {category_folder}")

        continue

    category_name = os.path.basename(category_folder)

    mrc_files = []

    for root, dirs, files in os.walk(category_folder):

        for f in files:

            if f.lower().endswith('.mrc'):

                mrc_files.append(os.path.join(root, f))

    print(f"{category_name}: found {len(mrc_files)} .mrc files")

    for path in mrc_files:

        with mrcfile.open(path, permissive=True) as mrc:

            data = mrc.data

        filename = os.path.splitext(os.path.basename(path))[0]

        # Prefix with category name so filenames never collide across categories

        prefixed_name = f"{category_name}_{filename}"

        if data.ndim == 2:

            out_path = os.path.join(OUTPUT_PATH, f"{prefixed_name}.tif")

            tifffile.imwrite(out_path, data)

            total_converted += 1

        elif data.ndim == 3:

            for slice_idx in range(data.shape[0]):

                out_path = os.path.join(OUTPUT_PATH, f"{prefixed_name}_slice{slice_idx}.tif")

                tifffile.imwrite(out_path, data[slice_idx])

                total_converted += 1

print(f"\nTotal images converted and merged: {total_converted}")

# Add BBBC005 training-pool images (already .tif, just copy with a prefix)
BBBC_FOLDER = r"D:\Projects\Microscopy_AI_Project\dataset\BBBC005_for_training"
if os.path.exists(BBBC_FOLDER):
    bbbc_files = [f for f in os.listdir(BBBC_FOLDER) if f.lower().endswith(('.tif', '.tiff'))]
    print(f"BBBC005: found {len(bbbc_files)} files")
    for f in bbbc_files:
        src = os.path.join(BBBC_FOLDER, f)
        dst = os.path.join(OUTPUT_PATH, f"BBBC005_{f}")
        import shutil
        shutil.copy(src, dst)
        total_converted += 1
print("Output folder:", OUTPUT_PATH)
