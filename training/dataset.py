import os

import torch

from torch.utils.data import Dataset

import tifffile

import numpy as np

class MicroscopyDataset(Dataset):

    def __init__(self, clean_dir, degraded_dir, patch_size=256):

        """

        clean_dir: folder with clean ground-truth .tif images

        degraded_dir: folder with matching degraded .tif images (same filenames)

        patch_size: we crop a fixed-size square patch from each image (keeps training fast and consistent)

        """

        self.clean_dir = clean_dir

        self.degraded_dir = degraded_dir

        self.patch_size = patch_size

        self.filenames = sorted([f for f in os.listdir(clean_dir) if f.lower().endswith(('.tif', '.tiff'))])

    def __len__(self):

        return len(self.filenames)

    def __getitem__(self, idx):

        fname = self.filenames[idx]

        clean = tifffile.imread(os.path.join(self.clean_dir, fname)).astype(np.float32)

        degraded = tifffile.imread(os.path.join(self.degraded_dir, fname)).astype(np.float32)

        # Normalize both images to the 0-1 range using the clean image's max

        # (keeps brightness scale consistent, which helps training stability)

        max_val = clean.max() if clean.max() > 0 else 1.0

        clean = clean / max_val

        degraded = degraded / max_val

        degraded = np.clip(degraded, 0, 1)

        # Random crop to patch_size x patch_size (same crop location for both images)

        h, w = clean.shape

        if h > self.patch_size and w > self.patch_size:

            top = np.random.randint(0, h - self.patch_size)

            left = np.random.randint(0, w - self.patch_size)

            clean = clean[top:top+self.patch_size, left:left+self.patch_size]

            degraded = degraded[top:top+self.patch_size, left:left+self.patch_size]

        else:

            # If image is smaller than patch_size, resize won't be needed at our resolution,

            # but as a safeguard we pad with zeros

            pad_h = max(0, self.patch_size - h)

            pad_w = max(0, self.patch_size - w)

            clean = np.pad(clean, ((0, pad_h), (0, pad_w)))

            degraded = np.pad(degraded, ((0, pad_h), (0, pad_w)))

            clean = clean[:self.patch_size, :self.patch_size]

            degraded = degraded[:self.patch_size, :self.patch_size]

         # ---- Data augmentation: random flips and 90-degree rotations ----

        # Applied identically to both images so they stay perfectly aligned

        if np.random.rand() > 0.5:

            clean = np.fliplr(clean).copy()

            degraded = np.fliplr(degraded).copy()

        if np.random.rand() > 0.5:

            clean = np.flipud(clean).copy()

            degraded = np.flipud(degraded).copy()

        k = np.random.randint(0, 4)  # 0, 90, 180, or 270 degree rotation

        clean = np.rot90(clean, k).copy()

        degraded = np.rot90(degraded, k).copy()

        # -------------------------------------------------------------

        clean_tensor = torch.from_numpy(clean).unsqueeze(0).float()

        degraded_tensor = torch.from_numpy(degraded).unsqueeze(0).float()

        return degraded_tensor, clean_tensor

