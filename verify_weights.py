import torch
from models.mini_raft_stereo import MiniRAFTStereo

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using:", device)

    # Load model
    model = MiniRAFTStereo().to(device)
    model.load_state_dict(torch.load("mini_raft_kitti.pth", map_location=device))
    model.eval()
    print("Checkpoint loaded successfully.")

    # Create dummy stereo pair (batch=1, channels=3, H=288, W=576)
    left = torch.randn(1, 3, 288, 576).to(device)
    right = torch.randn(1, 3, 288, 576).to(device)

    # Run forward pass
    with torch.no_grad():
        preds = model(left, right)

    print("Forward pass successful.")
    print("Number of predictions:", len(preds))
    print("Final prediction shape:", preds[-1].shape)

if __name__ == "__main__":
    main()
