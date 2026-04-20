import os
from PIL import Image
import numpy as np
import torch


class SceneFlowStereoTest(torch.utils.data.Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        print("Loading SceneFlow test dataset...")
        self.left_paths = []
        self.right_paths = []
        self.transform = transform
        print("Loading SceneFlow test dataset...")
        for f in os.listdir(os.path.join(root, "image_2")):
            if f.endswith(".png"):
                self.left_paths.append(os.path.join(root, "image_2", f))

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

        assert len(self.left_paths) == len(self.right_paths), \
            "SceneFlow TEST: mismatch in number of left/right files"

    def __len__(self):
        return len(self.left_paths)

    def __getitem__(self, idx):
        # Always convert to RGB (important!)
        left = Image.open(self.left_paths[idx]).convert("RGB")
        right = Image.open(self.right_paths[idx]).convert("RGB")

        sample = {
            "left_raw": np.array(left),  # for visualization if needed
            "left": left,
            "right": right
        }

        if self.transform:
            sample = self.transform(sample)

        return sample
