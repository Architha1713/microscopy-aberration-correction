import os
import shutil
import random

SOURCE = r"D:\Projects\Microscopy_AI_Project\dataset\BBBC005_v1_images"
TRAINPOOL_OUT = r"D:\Projects\Microscopy_AI_Project\dataset\BBBC005_for_training"
HOLDOUT_OUT = r"D:\Projects\Microscopy_AI_Project\dataset\BBBC005_holdout_generalization_test"

os.makedirs(TRAINPOOL_OUT, exist_ok=True)
os.makedirs(HOLDOUT_OUT, exist_ok=True)

all_files = [f for f in os.listdir(SOURCE) if f.lower().endswith(('.tif', '.tiff'))]
print(f"Total BBBC005 files found: {len(all_files)}")

random.seed(42)
random.shuffle(all_files)

# 500 images to blend into training/val/test (adds diversity)
# 100 images set aside as a PURE holdout -- never used in training at all,
# only used at the very end to test true generalization to an unseen dataset
training_pool = all_files[:500]
holdout_set = all_files[500:600]

for f in training_pool:
    shutil.copy(os.path.join(SOURCE, f), os.path.join(TRAINPOOL_OUT, f))

for f in holdout_set:
    shutil.copy(os.path.join(SOURCE, f), os.path.join(HOLDOUT_OUT, f))

print(f"Copied {len(training_pool)} images to training pool: {TRAINPOOL_OUT}")
print(f"Copied {len(holdout_set)} images to generalization holdout: {HOLDOUT_OUT}")
print("IMPORTANT: never mix the holdout folder into training/val/test splits.")