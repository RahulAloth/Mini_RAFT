import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import os

from models.mini_raft_stereo import MiniRAFTStereo
from data.sceneflow_dataset import SceneFlowStereo
from transforms.stereo_augment import StereoAugmentation
from transforms.stereo_compose import StereoCompose
import torchvision.transforms as T


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # -----------------------------
    # Load trained model
    # -----------------------------
    ckpt = "mini_raft_kitti.pth"
    model = MiniRAFTStereo().to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()
    print("Loaded checkpoint:", ckpt)

    # -----------------------------
    # Dataset path (SceneFlow)
    # -----------------------------
    sceneflow_root = "data/data_scene_flow/training"

    augment = StereoAugmentation(output_size=(288, 576))
    compose = StereoCompose(
        T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.5]*3, std=[0.5]*3)
        ])
    )

    def transform(sample):
        sample = augment(sample)
        sample = compose(sample)
        return sample

    dataset = SceneFlowStereo(sceneflow_root, transform=transform)

    idx = 0
    sample = dataset[idx]
    print("Loaded sample:", idx)

    left = sample["left"].unsqueeze(0).to(device)
    right = sample["right"].unsqueeze(0).to(device)

    # Convert disparity PIL → numpy (2D)
    disp_gt = np.array(sample["disparity"])

    # -----------------------------
    # Run inference
    # -----------------------------
    with torch.no_grad():
        preds = model(left, right)
        disp_pred = preds[-1]

    disp_pred_full = F.interpolate(
        disp_pred, scale_factor=4, mode="bilinear", align_corners=False
    ).cpu().numpy()[0, 0]

    # -----------------------------
    # Plot (NO OpenCV, NO colormap)
    # -----------------------------
    plt.figure(figsize=(18, 6))

    plt.subplot(1, 3, 1)
    plt.title("Left Image")
    plt.imshow(sample["left_raw"])
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.title("GT Disparity (grayscale)")
    plt.imshow(disp_gt, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.title("Predicted Disparity (grayscale)")
    plt.imshow(disp_pred_full, cmap="gray")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
