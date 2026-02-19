import torch
from torch.utils.data import DataLoader
from model.model_params import ModelParams
from dataset.synth_torch_dataset import TrajectoryPredictionDataset
from dataset.oxford_torch_dataset import OxfordrajectoryPredictionDataset
from model.cvae import TransformerCVAE
from utils.cvae_training import train

DATA_ROOT = "data"

TRAIN_PATH = f"{DATA_ROOT}/synthetic/256"
VAL_PATH = f"{DATA_ROOT}/oxford/processed"
TEST_PATH = f"{DATA_ROOT}/oxford/test_sample"
CKPT_DIR = "checkpoints"

def main():
    params = ModelParams()
    seq_len = params.seq_len
    pred_len = params.pred_len

    train_dataset = TrajectoryPredictionDataset(
        data_path=TRAIN_PATH,
        seq_len=seq_len,
        pred_len=pred_len,
        stride=2
    )
    val_dataset = OxfordrajectoryPredictionDataset(
        data_path=VAL_PATH,
        seq_len=seq_len,
        pred_len=pred_len,
        stride=1
    )
    test_dataset = OxfordrajectoryPredictionDataset(
        data_path=VAL_PATH,
        seq_len=seq_len,
        pred_len=pred_len,
        stride=24
    )

    batch_size = 32
    num_workers = 8
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=num_workers, drop_last=False)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TransformerCVAE(params=params).to(device)

    history = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        num_epochs=15,
        patience=10,
        learning_rate=1e-4,
        weight_decay=1e-5,
        ckpt_dir=CKPT_DIR,
        train_metric='mse',
    )

    print("Done. Last epoch losses:", {k: v[-1] for k, v in history.items() if len(v) > 0})

if __name__ == '__main__':
    main()