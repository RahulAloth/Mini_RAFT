# data/kitti_dataset.py
import os
import glob
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class KITTIStereo(Dataset):
    """
    Loads:
        - left image
        - right image
        - disparity map (float32)
        - valid mask (0/1)
    """
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        self.left_paths = sorted(glob.glob(os.path.join(root_dir, "training/image_2/*.png")))
        self.right_paths = sorted(glob.glob(os.path.join(root_dir, "training/image_3/*.png")))
        self.disp_paths = sorted(glob.glob(os.path.join(root_dir, "training/disp_occ_0/*.png")))

        assert len(self.left_paths) == len(self.right_paths), "Left/right mismatch"
        assert len(self.left_paths) == len(self.disp_paths), "Image/disparity mismatch"

    def __len__(self):
        return len(self.left_paths)

    def __getitem__(self, idx):
        left = Image.open(self.left_paths[idx]).convert("RGB")
        right = Image.open(self.right_paths[idx]).convert("RGB")

        disp_raw = Image.open(self.disp_paths[idx])
        disp_np = np.array(disp_raw).astype(np.float32) / 256.0
        valid_np = (disp_np > 0).astype(np.float32)

        disp = Image.fromarray(disp_np, mode="F")
        valid = Image.fromarray(valid_np, mode="F")

        sample = {
            "left": left,
            "right": right,
            "disparity": disp,
            "valid_mask": valid
        }

        if self.transform:
            sample = self.transform(sample)

        return sample
