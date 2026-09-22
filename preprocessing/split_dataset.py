import os
import shutil
import random

DATASET_PATH = r"D:\Projects\Microscopy_AI_Project\dataset\all_categories_tif"
OUTPUT_PATH = r"D:\Projects\Microscopy_AI_Project\dataset\split"

# Collect all image files
image_files = []
for root, dirs, files in os.walk(DATASET_PATH):
    for f in files:
        if f.lower().endswith(('.tif', '.tiff')):
            image_files.append(os.path.join(root, f))

print(f"Total images found: {len(image_files)}")

# Shuffle for a random, unbiased split
random.seed(42)
random.shuffle(image_files)

n = len(image_files)
train_end = int(n * 0.70)
val_end = train_end + int(n * 0.15)

train_files = image_files[:train_end]
val_files = image_files[train_end:val_end]
test_files = image_files[val_end:]

print(f"Train: {len(train_files)}  Validation: {len(val_files)}  Test: {len(test_files)}")

for split_name, files in [("train", train_files), ("val", val_files), ("test", test_files)]:
    split_folder = os.path.join(OUTPUT_PATH, split_name)
    os.makedirs(split_folder, exist_ok=True)
    for f in files:
        shutil.copy(f, split_folder)

print("Done. Check the dataset/split folder — it should now have train/val/test subfolders.")