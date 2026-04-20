import os
from PIL import Image
import numpy as np
import torch

class SceneFlowStereo(torch.utils.data.Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform

        # Left images
        self.left_paths = sorted([
            os.path.join(root, "image_2", f)
            for f in os.listdir(os.path.join(root, "image_2"))
            if f.endswith(".png")
        ])

        # Right images
        self.right_paths = sorted([
            os.path.join(root, "image_3", f)
            for f in os.listdir(os.path.join(root, "image_3"))
            if f.endswith(".png")
        ])

        # Disparity maps
        self.disp_paths = sorted([
            os.path.join(root, "disp_occ_0", f)
            for f in os.listdir(os.path.join(root, "disp_occ_0"))
            if f.endswith(".png")
        ])

        assert len(self.left_paths) == len(self.right_paths) == len(self.disp_paths), \
            "SceneFlow: mismatch in number of left/right/disp files"

    def __len__(self):
        return len(self.left_paths)

    def __getitem__(self, idx):
        # Load RGB stereo pair
        left = Image.open(self.left_paths[idx]).convert("RGB")
        right = Image.open(self.right_paths[idx]).convert("RGB")

        # Load disparity as single-channel grayscale (critical fix)
        disp_np = np.array(Image.open(self.disp_paths[idx]).convert("I")).astype(np.float32)
        disp_np = disp_np / 256.0  # SceneFlow scale

        # Valid mask (all positive disparities)
        valid_mask_np = (disp_np > 0).astype(np.float32)

        # Convert disparity + mask to PIL for augmentation
        disp = Image.fromarray(disp_np)
        valid_mask = Image.fromarray(valid_mask_np)

        sample = {
            "left_raw": np.array(left),   # for visualization
            "left": left,
            "right": right,
            "disparity": disp,
            "valid_mask": valid_mask
        }

        # Apply augmentations + tensor conversion
        if self.transform:
            sample = self.transform(sample)

        return sample
