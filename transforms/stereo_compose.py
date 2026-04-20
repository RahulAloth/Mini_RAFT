# transforms/stereo_compose.py
import numpy as np
import torchvision.transforms as T
import torch


class StereoCompose:
    """
    Applies image transforms (ColorJitter + ToTensor + Normalize)
    and converts disparity + mask to proper float tensors.
    """

    def __init__(self, image_transforms):
        self.image_transforms = image_transforms

    def __call__(self, sample):

        sample["left"] = self.image_transforms(sample["left"])
        sample["right"] = self.image_transforms(sample["right"])

        disp_np = np.array(sample["disparity"]).astype(np.float32)
        sample["disparity"] = torch.from_numpy(disp_np).unsqueeze(0)

        mask_np = np.array(sample["valid_mask"]).astype(np.float32)
        sample["valid_mask"] = torch.from_numpy(mask_np).unsqueeze(0)

        return sample
