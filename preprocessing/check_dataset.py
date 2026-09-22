import os
import tifffile
import matplotlib.pyplot as plt

# EDIT THIS PATH to point to the folder containing your actual .tif image files
DATASET_PATH = r"D:\Projects\Microscopy_AI_Project\dataset\F-actin_tif"

# List all .tif files in that folder (and subfolders)
image_files = []
for root, dirs, files in os.walk(DATASET_PATH):
    for f in files:
        if f.lower().endswith(('.tif', '.tiff')):
            image_files.append(os.path.join(root, f))

print(f"Found {len(image_files)} image files.")

if len(image_files) == 0:
    print("No images found — double check DATASET_PATH is correct.")
else:
    # Load and show the first image
    sample_path = image_files[0]
    print("Loading sample image:", sample_path)
    img = tifffile.imread(sample_path)
    print("Image shape:", img.shape)
    print("Image data type:", img.dtype)
    print("Min pixel value:", img.min(), "Max pixel value:", img.max())

    plt.imshow(img, cmap='gray')
    plt.title("Sample Microscopy Image")
    plt.axis('off')
    plt.savefig(r"D:\Projects\Microscopy_AI_Project\dataset\sample_check.png")
    print("Saved a preview image as sample_check.png — open it in File Explorer to look at it.")