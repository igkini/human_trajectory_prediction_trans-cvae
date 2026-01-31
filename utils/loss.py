from typing import Dict, Tuple, Optional
import torch
from .sample_grid import check_trajectory_collisions
import torch.nn.functional as F

def kl_gaussian(mu, logvar, mask=None):
    logvar = logvar.clamp(-10., 10.)
    kl = 0.5 * (logvar.exp() + mu.pow(2) - 1.0 - logvar)  # [*, Z]
    kl = kl.sum(dim=-1) 
    if mask is not None:
        kl = (kl * mask).sum() / mask.sum().clamp_min(1)
    else:
        kl = kl.mean()
    return kl

def compute_cvae_best_of_n_loss(
    predictions: torch.Tensor,      # (b, n_samples, a, t, 2)
    ground_truth: torch.Tensor,     # (b, a, t, 2)
    mask: Optional[torch.Tensor] = None,  # (b, a, t) or (b, a, t, 1)
    metric: str = 'ade',
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute loss using best trajectory sample.
    
    Returns:
        best_loss: loss of the best prediction
        best_preds: (b, a, t, 2) best predictions
        metric_mean: mean metric across all samples
        metric_std: std metric across all samples
    """
    batch_size, n_samples, n_agents, n_timesteps, _ = predictions.shape
    
    gt_expanded = ground_truth.unsqueeze(1).expand_as(predictions)
    
    if metric == 'mse':
        errors = ((predictions - gt_expanded) ** 2).sum(dim=-1)
    elif metric == 'fde':
        errors = torch.norm(predictions[..., -1, :] - gt_expanded[..., -1, :], dim=-1)
        errors = errors.unsqueeze(-1).expand(batch_size, n_samples, n_agents, n_timesteps)
    else:  # ADE
        errors = torch.norm(predictions - gt_expanded, dim=-1)
    
    if mask is not None:
        if mask.dim() == 4:
            mask = mask.squeeze(-1)
        mask_expanded = mask.unsqueeze(1).expand_as(errors)
        error_per_sample = (errors * mask_expanded).sum(dim=(2, 3)) / (
            mask_expanded.sum(dim=(2, 3)) + 1e-6
        )
    else:
        error_per_sample = errors.mean(dim=(2, 3))
    
    min_errors, best_indices = torch.min(error_per_sample, dim=1)
    
    batch_indices = torch.arange(batch_size, device=predictions.device)
    best_preds = predictions[batch_indices, best_indices]
    
    best_loss = min_errors.mean()
    metric_mean = error_per_sample.mean()
    metric_std = error_per_sample.std()

    return best_loss, best_preds, metric_mean, metric_std

def compute_tracking_loss(
    pred: torch.Tensor,         # (b, a, t, 2)
    target: torch.Tensor,       # (b, a, t, 2)
    mask: torch.Tensor = None,  # (b, a, t) or (b, a, t, 1)
    metric: str = 'mse'
) -> torch.Tensor:
    """
    Compute trajectory prediction loss.
    
    Args:
        pred: predicted trajectories
        target: ground truth trajectories
        mask: validity mask
        metric: 'mse', 'l1', or 'huber'
    
    Returns:
        scalar loss
    """
    diff = pred - target
    
    if metric == 'mse':
        loss = diff.pow(2)  # (b, a, t, 2)
    elif metric == 'l1':
        loss = diff.abs()  # (b, a, t, 2)
    elif metric == 'huber':
        loss = F.smooth_l1_loss(pred, target, reduction='none')  # (b, a, t, 2)
    else:
        raise ValueError(f"Unknown metric: {metric}")
    
    if mask is not None:
        # Handle both (b, a, t) and (b, a, t, 1) masks
        if mask.dim() == 3:
            mask = mask.unsqueeze(-1)  # (b, a, t) -> (b, a, t, 1)
        mask2 = mask.expand_as(pred).to(pred.dtype)  # (b, a, t, 2)
        valid_count = mask2.sum()
        eps = torch.finfo(pred.dtype).eps
        denom = torch.clamp(valid_count, min=eps)
        loss = (loss * mask2).sum() / denom
    else:
        loss = loss.mean()
    
    return loss

def compute_occupancy_loss(
    preds: torch.Tensor,        # (b, a, t, 2)
    raw_grid: torch.Tensor,
    scene_coord: torch.Tensor,
) -> Tuple[torch.Tensor, int, int]:
    """
    Compute occupancy-related losses and counts.

    Returns:
        occ_loss: combined loss value
        obstacle_hits: number of predicted points inside obstacles
        oob_hits: number of predicted points out of bounds
    """
    is_in_obstacle, is_out_of_bounds = check_trajectory_collisions(
        predicted_trajectory=preds,
        raw_grid=raw_grid,
        scene_coord=scene_coord,
        obstacle_threshold=0.9
    )
    
    obstacle_hits = is_in_obstacle.sum().item()
    oob_hits = is_out_of_bounds.sum().item()
    
    obstacle_loss = is_in_obstacle.float().mean()
    oob_loss = is_out_of_bounds.float().mean()
    
    occ_loss = obstacle_loss + oob_loss
    
    return occ_loss, obstacle_hits, oob_hits