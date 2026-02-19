import torch
from torch.utils.data import DataLoader
import os
import time
from model.model_params import ModelParams
from model.cvae import TransformerCVAE
from dataset.oxford_torch_dataset import OxfordrajectoryPredictionDataset
from utils.cvae_training import validate

DATA_ROOT = "data"
TEST_PATH = f"{DATA_ROOT}/oxford/test_sample"
CKPT_DIR = "checkpoints"
CKPT_PATH = f"{CKPT_DIR}/model_epoch_001.pth"
OUTPUT_BASE_DIR = "trajectory_plots"

def load_checkpoint(model, ckpt_path, device):
    if not os.path.exists(ckpt_path):
        print(f"⚠️ No checkpoint found at {ckpt_path}, using random weights")
        return

    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict):
        key = next((k for k in ('model_state_dict', 'state_dict', 'model') if k in checkpoint), None)
        state_dict = checkpoint[key] if key else checkpoint
        if key == 'model_state_dict':
            if 'epoch' in checkpoint:
                print(f"   Checkpoint epoch: {checkpoint['epoch']}")
            if 'val_loss' in checkpoint:
                print(f"   Validation loss: {checkpoint['val_loss']:.4f}")
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    print(f"✅ Loaded checkpoint from {ckpt_path}")

def main():
    params = ModelParams()
    seq_len = params.seq_len
    pred_len = params.pred_len

    test_dataset = OxfordrajectoryPredictionDataset(
        data_path=TEST_PATH,
        seq_len=seq_len,
        pred_len=pred_len,
        stride=1
    )

    batch_size = 32
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        drop_last=False
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TransformerCVAE(params=params).to(device)
    load_checkpoint(model, CKPT_PATH, device)

    os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Starting Testing on {len(test_dataset)} samples...")
    print(f"Device: {device} | Batch size: {batch_size}")
    print(f"{'='*60}\n")

    start_time = time.time()
    v_min, v_mean, total_hits = validate(
        model=model,
        dataloader=test_loader,
        device=device,
        description="Testing",
        visualize_traj=False,
    )
    total_time = time.time() - start_time

    total_samples = len(test_dataset)
    time_per_sample = total_time / total_samples

    print(f"\n{'='*60}")
    print(f"Testing complete!")
    print(f"Best-of-N Loss (v_min): {v_min:.4f}")
    print(f"Mean Loss (v_mean):     {v_mean:.4f}")
    print(f"Total Obstacle Hits:    {total_hits}")
    print(f"Total Samples:          {total_samples}")
    print(f"Total Time:             {total_time:.2f}s")
    print(f"Time per Sample:        {time_per_sample*1000:.2f}ms ({1/time_per_sample:.1f} samples/sec)")
    print(f"Visualizations saved in: '{OUTPUT_BASE_DIR}'")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()