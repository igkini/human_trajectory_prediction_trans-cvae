import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from .visualize import visualize_vae_samples
from .loss import compute_occupancy_loss, kl_gaussian, compute_cvae_best_of_n_loss, compute_tracking_loss
import numpy as np

OUTPUT_BASE_DIR = "trajectory_plots"

def to_device(batch, device: torch.device):
    """Recursively move tensors (and nested dict/list/tuple) to device."""
    if isinstance(batch, dict):
        return {k: to_device(v, device) for k, v in batch.items()}
    elif isinstance(batch, (list, tuple)):
        return type(batch)(to_device(x, device) for x in batch)
    elif torch.is_tensor(batch):
        return batch.to(device)
    return batch


class EarlyStopping:
    def __init__(self, patience: int = 7, verbose: bool = False, delta: float = 0.0) -> None:
        self.patience: int = patience
        self.verbose: bool = verbose
        self.counter: int = 0
        self.best_score: Optional[float] = None
        self.early_stop: bool = False
        self.val_loss_min: float = float("inf")
        self.delta: float = delta

    def __call__(self, val_loss: float, model: nn.Module, path: str):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} / {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss: float, model: nn.Module, path: str):
        if self.verbose:
            print(f"Validation loss decreased ({self.val_loss_min:.6f} → {val_loss:.6f}). Saving best checkpoint...")
        torch.save(model.state_dict(), os.path.join(path, "checkpoint.pth"))
        self.val_loss_min = val_loss


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    description: str,
    val_metric: str = 'ade',
    n_samples: int = 10,
    visualize_traj: bool = False,
) -> Tuple[float, float, int]:
    """
    Validation loop - encoder does NOT see future trajectories.
    
    Args:
        val_metric: Metric for evaluation ('ade', 'fde', 'mse')
    """
    model.eval()

    total_best_loss = 0.0
    total_all_samples_mean = 0.0
    total_obstacle_hits = 0
    n_batches = 0
    
    bar = tqdm(dataloader, total=len(dataloader), desc=description, leave=False)

    with torch.no_grad():
        for batch_idx, (past_seq, future_seq) in enumerate(bar):
            past_seq = to_device(past_seq, device)
            future_seq = to_device(future_seq, device)

            occ_grid, coord_grid = model.get_grid(past_seq)
            
            preds_list = []
            for _ in range(n_samples):
                pred = model.sample(past_seq, temperature=0.2)
                preds_list.append(pred)

            preds_all = torch.stack(preds_list, dim=0).transpose(0, 1)  # (b, n_samples, a, t, 2)

            mask = future_seq.get("prediction_pos_mask", None)
            best_loss, best_preds, all_samples_mean, all_samples_std = compute_cvae_best_of_n_loss(
                preds_all,
                future_seq["prediction_pos"],
                mask,
                metric=val_metric,
            )

            obstacle_hits = 0
            for mode_idx in range(n_samples):
                _, mode_hits, _ = compute_occupancy_loss(
                    preds_all[:, mode_idx, :, :],
                    occ_grid,
                    coord_grid
                )
                obstacle_hits += mode_hits

            total_best_loss += best_loss.item()
            total_all_samples_mean += all_samples_mean.item()
            total_obstacle_hits += obstacle_hits
            n_batches += 1
            
            bar.set_postfix({val_metric: total_best_loss / n_batches})

            if visualize_traj:
                batch_output_dir = os.path.join(OUTPUT_BASE_DIR, f"batch_{batch_idx}")
                os.makedirs(batch_output_dir, exist_ok=True)

                visualize_vae_samples(
                    coord_grid=coord_grid,
                    target_seq=future_seq['prediction_pos'],
                    preds_list=preds_list,
                    past_seq=past_seq['human_pos'],
                    output_dir=batch_output_dir,
                    map_image=occ_grid,
                    top_k=2
                )

    avg_best = total_best_loss / max(1, n_batches)
    avg_all_samples = total_all_samples_mean / max(1, n_batches)
    
    return avg_best, avg_all_samples, total_obstacle_hits


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    train_metric: str,
    kl_weight: float,
) -> Tuple[float, int]:
    """
    Training loop - encoder DOES see future trajectories.
    """
    model.train()

    total_loss = 0.0
    total_obstacle_hits = 0
    n_batches = 0
    
    bar = tqdm(dataloader, total=len(dataloader), desc="Training", leave=False)

    for past_seq, future_seq in bar:
        past_seq = to_device(past_seq, device)
        future_seq = to_device(future_seq, device)

        optimizer.zero_grad()

        future_batch = {
            'human_pos': future_seq['prediction_pos'],
            'human_pos/mask': future_seq.get('prediction_pos/mask', None),
        }

        out, mu, logvar = model.forward(past_seq, future_batch)

        occ_grid, coord_grid = model.get_grid(past_seq)

        mask = future_seq.get("prediction_pos_mask", None)
        track_loss = compute_tracking_loss(
            out,
            future_seq["prediction_pos"],
            mask,
            metric=train_metric,
        )

        _, obstacle_hits, _ = compute_occupancy_loss(out, occ_grid, coord_grid)

        kl_term = kl_gaussian(mu, logvar)

        loss = track_loss + kl_weight * kl_term

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_obstacle_hits += obstacle_hits
        n_batches += 1
        
        bar.set_postfix(loss=total_loss / n_batches)

    return total_loss / max(1, n_batches), total_obstacle_hits


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    num_epochs: int,
    patience: int,
    learning_rate: float,
    weight_decay: float,
    ckpt_dir: str,
    train_metric: str = 'mse',
    val_metric: str = 'ade',
    initial_kl_weight: float = 0.1,
    max_kl_weight: float = 1.0,
    kl_warmup_epochs: int = 7,
) -> Dict[str, List[float]]:
    """
    Train trajectory prediction model.
    
    Args:
        train_metric: Loss metric for training ('mse', 'l1', 'huber')
        val_metric: Metric for validation ('ade', 'fde', 'mse')
        initial_kl_weight: Starting KL weight for warmup
        max_kl_weight: Final KL weight after warmup
        kl_warmup_epochs: Number of epochs for KL warmup
    """
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=1, factor=0.3)
    early_stopping = EarlyStopping(patience=patience, verbose=True)

    history: Dict[str, List[float]] = {"train_loss": [], "val_loss": [], "test_loss": []}

    Path(ckpt_dir).mkdir(parents=True, exist_ok=True)

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")

        if epoch < kl_warmup_epochs:
            current_kl_weight = initial_kl_weight + (max_kl_weight - initial_kl_weight) * np.log(1 + epoch) / np.log(1 + kl_warmup_epochs)
        else:
            current_kl_weight = max_kl_weight

        train_loss, train_obs_hits = train_one_epoch(
            model, train_loader, optimizer, device,
            train_metric=train_metric,
            kl_weight=current_kl_weight,
        )

        val_best, val_all_samples, val_obs_hits = validate(
            model, val_loader, device, 
            description="Validation",
            val_metric=val_metric,
        )
        scheduler.step(val_best)

        test_best, test_all_samples, test_obs_hits = validate(
            model, test_loader, device, 
            description="Test",
            val_metric=val_metric,
            visualize_traj=True,
        )

        print(f"train obstacle hits: {train_obs_hits} | val obstacle hits: {val_obs_hits} | test obstacle hits: {test_obs_hits}")
        print(f"train_loss ({train_metric}): {train_loss:.6f} | val_{val_metric}: {val_best:.6f} (all: {val_all_samples:.6f}) | test_{val_metric}: {test_best:.6f} (all: {test_all_samples:.6f})")

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_best)
        history["test_loss"].append(test_best)

        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": train_loss,
                "val_loss": val_best,
                "test_loss": test_best,
            },
            os.path.join(ckpt_dir, f"model_epoch_{epoch + 1:03d}.pth"),
        )
        print(f"✅ Saved epoch checkpoint to {ckpt_dir}/model_epoch_{epoch + 1:03d}.pth")

        early_stopping(val_best, model, ckpt_dir)
        if early_stopping.early_stop:
            print(f"⏹️ Early stopping triggered at epoch {epoch + 1}")
            break

        torch.cuda.empty_cache()

    return history