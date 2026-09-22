import os
import random

SOURCE = r"D:\Projects\Microscopy_AI_Project\dataset\all_categories_tif"
CAP = 3000  # a solid, diverse training pool without blowing up training time

all_files = [f for f in os.listdir(SOURCE) if f.lower().endswith(('.tif', '.tiff'))]
print(f"Total available: {len(all_files)}")

random.seed(42)
random.shuffle(all_files)

# Keep the cap, delete the rest so split_dataset.py only sees the capped set
keep = set(all_files[:CAP])
removed = 0
for f in all_files[CAP:]:
    os.remove(os.path.join(SOURCE, f))
    removed += 1

print(f"Kept {len(keep)} images, removed {removed} extra images from the pool.")
print("Ready for split_dataset.py")