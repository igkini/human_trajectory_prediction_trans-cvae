import torch
from torch.utils.data import DataLoader
from .model.model_params import ModelParams
from .dataset.synthetic.torch_dataset_task_synth import TrajectoryPredictionDataset
from .dataset.torch_dataset_station import StationTrajectoryPredictionDataset

from .model.cvae import TransformerCVAE
from .utils.cvae_training import train

# === PATHS ===
DATA_ROOT = "scripts/data/synthetic"

#========================================================
#                    TRAIN SYNTH-VAL OXFORD
#========================================================

TRAIN_PATH = f"{DATA_ROOT}/256_train"
# VAL_PATH = f"{DATA_ROOT}/400_test"
# TEST_PATH = f"{DATA_ROOT}/400_cell"
# VAL_PATH = "/home/gkini/Human-Traj-Prediction/scripts/data/oxford_task/all_320"
# TEST_PATH = "/home/gkini/Human-Traj-Prediction/scripts/data/oxford_task/test_320"

TRAIN_PATH = f"{DATA_ROOT}/256_train"
VAL_PATH = "/home/gkini/Human-Traj-Prediction/scripts/data/oxford_task/all_stations"
# TEST_PATH = "/home/gkini/Human-Traj-Prediction/scripts/data/oxford_task/all_stations_test"
TEST_PATH = f"{DATA_ROOT}/256_cell_new"

CKPT_DIR = "/home/gkini/Human-Traj-Prediction/scripts/checkpoints"

# === MODEL PARAMS ===
params = ModelParams()
seq_len = params.seq_len
pred_len = params.pred_len

# === DATASETS ===
train_dataset = TrajectoryPredictionDataset(
    data_path=TRAIN_PATH,
    seq_len=seq_len,
    pred_len=pred_len,
    stride=4)

val_dataset = StationTrajectoryPredictionDataset(
    data_path=VAL_PATH,
    seq_len=seq_len,
    pred_len=pred_len,
    stride=1)

test_dataset = StationTrajectoryPredictionDataset(
    data_path=VAL_PATH,
    seq_len=seq_len,
    pred_len=pred_len,
    stride=24)

# === DATALOADERS ===
batch_size = 32
num_workers = 8

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=num_workers, drop_last=False)
val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False)
test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False)

# === MODEL ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TransformerCVAE(params=params).to(device)

# === TRAIN ===
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