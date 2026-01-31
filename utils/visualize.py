import matplotlib.pyplot as plt
import torch
import os
import numpy as np
from matplotlib import cm
from typing import Optional, List
from .sample_grid import check_trajectory_collisions

# Visualization constants
PAST_COLOR = 'blue'
GROUND_TRUTH_COLOR = 'green'
VALID_PRED_COLOR = 'cyan'
OBSTACLE_COLOR = 'orange'
OOB_COLOR = 'purple'
OBSTACLE_EDGE_COLOR = 'red'

MARKER_SIZE = 30
MARKER_SIZE_OOB = 50
LINE_WIDTH = 2.0
LINE_WIDTH_PRED = 1.5
MARKER_SIZE_SMALL = 4
EDGE_WIDTH = 0.5
EDGE_WIDTH_OBSTACLE = 1.5
EDGE_WIDTH_OOB = 2.0

FIGURE_SIZE = (8, 8)
MAP_ALPHA = 0.5
GRID_ALPHA = 0.3


def _plot_trajectory_with_collision(
    pred_x: np.ndarray,
    pred_y: np.ndarray,
    obstacles: Optional[np.ndarray] = None,
    oob: Optional[np.ndarray] = None,
    color='cyan',
    alpha: float = 0.7,
    label: Optional[str] = None,
):
    """
    Helper function to plot trajectory with collision detection.
    
    Args:
        pred_x, pred_y: Trajectory coordinates
        obstacles: Boolean mask for obstacle points
        oob: Boolean mask for out-of-bounds points
        color: Color for valid points (or all points if no collision data)
        alpha: Transparency
        label: Legend label
    """
    if obstacles is None or oob is None:
        # No collision detection - plot normally
        plt.plot(pred_x, pred_y, '--', color=color, linewidth=LINE_WIDTH, 
                marker='o', markersize=MARKER_SIZE_SMALL, alpha=alpha, label=label)
        return
    
    valid_mask = ~obstacles & ~oob
    obstacle_mask = obstacles
    oob_mask = oob
    
    # Plot valid points
    if valid_mask.any():
        valid_indices = np.where(valid_mask)[0]
        plt.scatter(pred_x[valid_mask], pred_y[valid_mask],
                   color=color, s=MARKER_SIZE, marker='o', 
                   edgecolors='black', linewidths=EDGE_WIDTH,
                   label=label, zorder=3, alpha=alpha)
        
        # Connect consecutive valid points
        for i in range(len(valid_indices) - 1):
            if valid_indices[i+1] - valid_indices[i] == 1:
                plt.plot([pred_x[valid_indices[i]], pred_x[valid_indices[i+1]]], 
                        [pred_y[valid_indices[i]], pred_y[valid_indices[i+1]]], 
                        '--', color=color, linewidth=LINE_WIDTH_PRED, alpha=alpha * 0.7)
    
    # Plot obstacle points
    if obstacle_mask.any():
        obstacle_indices = np.where(obstacle_mask)[0]
        obs_label = 'In Obstacle' if label and valid_mask.any() else label
        plt.scatter(pred_x[obstacle_mask], pred_y[obstacle_mask],
                   color=OBSTACLE_COLOR, s=MARKER_SIZE, marker='o',
                   edgecolors=OBSTACLE_EDGE_COLOR, linewidths=EDGE_WIDTH_OBSTACLE,
                   label=obs_label, zorder=3, alpha=alpha)
        
        for i in range(len(obstacle_indices) - 1):
            if obstacle_indices[i+1] - obstacle_indices[i] == 1:
                plt.plot([pred_x[obstacle_indices[i]], pred_x[obstacle_indices[i+1]]], 
                        [pred_y[obstacle_indices[i]], pred_y[obstacle_indices[i+1]]], 
                        'r--', linewidth=LINE_WIDTH_PRED, alpha=alpha * 0.7)
    
    # Plot out-of-bounds points
    if oob_mask.any():
        oob_indices = np.where(oob_mask)[0]
        oob_label = 'Out of Bounds' if label and (valid_mask.any() or obstacle_mask.any()) else label
        plt.scatter(pred_x[oob_mask], pred_y[oob_mask],
                   color=OOB_COLOR, s=MARKER_SIZE_OOB, marker='x',
                   linewidths=EDGE_WIDTH_OOB, label=oob_label, zorder=3, alpha=alpha)
        
        for i in range(len(oob_indices) - 1):
            if oob_indices[i+1] - oob_indices[i] == 1:
                plt.plot([pred_x[oob_indices[i]], pred_x[oob_indices[i+1]]], 
                        [pred_y[oob_indices[i]], pred_y[oob_indices[i+1]]], 
                        'm--', linewidth=LINE_WIDTH_PRED, alpha=alpha * 0.7)


def visualize_trajectories(
    coord_grid: torch.Tensor,           # (b, 2, h, w)
    target_seq: torch.Tensor,           # (b, a, t, 2)
    predicted_seq: torch.Tensor,        # (b, a, t, 2) or (b, a, t, n_modes, 2)
    past_seq: torch.Tensor,             # (b, a, past_t, 2)
    output_dir: str = "trajectory_plots",
    map_image: Optional[torch.Tensor] = None,  # (b, 1, h, w)
    mode_probs: Optional[torch.Tensor] = None,  # (b, a, n_modes) for multimodal
    top_k: Optional[int] = None,
):
    """
    Visualize trajectory predictions with collision detection.
    
    Handles both single-mode and multi-mode predictions:
    - Single-mode: predicted_seq shape (b, a, t, 2)
    - Multi-mode: predicted_seq shape (b, a, t, n_modes, 2) with mode_probs
    
    Args:
        coord_grid: Coordinate grid for mapping positions
        target_seq: Ground truth trajectories
        predicted_seq: Predicted trajectories (single or multi-mode)
        past_seq: Past trajectories
        output_dir: Save directory
        map_image: Occupancy grid/map image
        mode_probs: Mode probabilities for multi-mode predictions
        top_k: Number of top modes to visualize (for multi-mode)
    """
    # Detect if multi-mode
    is_multimodal = predicted_seq.dim() == 5
    
    if is_multimodal:
        b, a, t, n_modes, _ = predicted_seq.shape
        if mode_probs is None:
            mode_probs = torch.ones(b, a, n_modes, device=predicted_seq.device) / n_modes
        if top_k is None:
            top_k = n_modes
    else:
        b, a, t, _ = predicted_seq.shape
        n_modes = 1
    
    # Check collisions if map available
    collision_data = None
    if map_image is not None and coord_grid is not None:
        if is_multimodal:
            # Reshape to check all modes: (b, a*n_modes, t, 2)
            pred_reshaped = predicted_seq.permute(0, 1, 3, 2, 4).reshape(b, a * n_modes, t, 2)
            is_in_obstacle, is_out_of_bounds = check_trajectory_collisions(
                pred_reshaped, map_image, coord_grid
            )
            # Reshape back: (b, a, n_modes, t)
            collision_data = {
                'obstacles': is_in_obstacle.reshape(b, a, n_modes, t).cpu().numpy(),
                'oob': is_out_of_bounds.reshape(b, a, n_modes, t).cpu().numpy()
            }
        else:
            is_in_obstacle, is_out_of_bounds = check_trajectory_collisions(
                predicted_seq, map_image, coord_grid
            )
            collision_data = {
                'obstacles': is_in_obstacle.cpu().numpy(),
                'oob': is_out_of_bounds.cpu().numpy()
            }
    
    # Convert to numpy
    target_traj = target_seq.cpu().numpy()
    predicted_traj = predicted_seq.cpu().numpy()
    past_traj = past_seq.cpu().numpy()
    coord_grid_np = coord_grid.cpu().numpy()
    
    if mode_probs is not None:
        mode_probs_np = mode_probs.cpu().numpy()
    
    batch_size, channels, height, width = coord_grid_np.shape
    os.makedirs(output_dir, exist_ok=True)
    
    total_plots = 0
    
    for batch_idx in range(b):
        for agent_idx in range(a):
            plt.figure(figsize=FIGURE_SIZE)
            
            # Plot background
            grid_x = coord_grid_np[batch_idx, 0]
            grid_y = coord_grid_np[batch_idx, 1] if channels > 1 else grid_x
            extent = (grid_x.min(), grid_x.max(), grid_y.min(), grid_y.max())
            
            if map_image is not None:
                map_img = map_image.cpu().numpy()[batch_idx, 0]
                plt.imshow(map_img, cmap='gray', alpha=MAP_ALPHA, extent=extent)
            else:
                plt.imshow(grid_x, cmap='viridis', alpha=GRID_ALPHA, extent=extent)
            
            # Extract trajectories
            target_x = target_traj[batch_idx, agent_idx, :, 0]
            target_y = target_traj[batch_idx, agent_idx, :, 1]
            past_x = past_traj[batch_idx, agent_idx, :, 0]
            past_y = past_traj[batch_idx, agent_idx, :, 1]
            
            # Plot past and ground truth
            plt.plot(past_x, past_y, '-o', color=PAST_COLOR, linewidth=LINE_WIDTH,
                    markersize=MARKER_SIZE_SMALL, alpha=0.7, label='Past')
            plt.plot(target_x, target_y, '-s', color=GROUND_TRUTH_COLOR, linewidth=LINE_WIDTH,
                    markersize=MARKER_SIZE_SMALL, alpha=0.7, label='Ground Truth')
            
            # Plot predictions
            if is_multimodal:
                # Get top-k modes
                agent_probs = mode_probs_np[batch_idx, agent_idx]
                topk_indices = np.argsort(agent_probs)[-top_k:][::-1]
                colors = cm.rainbow(np.linspace(0, 1, top_k))
                
                for i, mode_idx in enumerate(topk_indices):
                    prob = agent_probs[mode_idx]
                    pred_x = predicted_traj[batch_idx, agent_idx, :, mode_idx, 0]
                    pred_y = predicted_traj[batch_idx, agent_idx, :, mode_idx, 1]
                    
                    alpha = np.clip(0.5 + 0.5 * prob, 0.0, 1.0)
                    
                    obstacles = collision_data['obstacles'][batch_idx, agent_idx, mode_idx] if collision_data else None
                    oob = collision_data['oob'][batch_idx, agent_idx, mode_idx] if collision_data else None
                    
                    _plot_trajectory_with_collision(
                        pred_x, pred_y, obstacles, oob,
                        color=colors[i], alpha=alpha,
                        label=f'Mode {mode_idx+1} (p={prob:.2f})'
                    )
            else:
                # Single mode
                pred_x = predicted_traj[batch_idx, agent_idx, :, 0]
                pred_y = predicted_traj[batch_idx, agent_idx, :, 1]
                
                obstacles = collision_data['obstacles'][batch_idx, agent_idx] if collision_data else None
                oob = collision_data['oob'][batch_idx, agent_idx] if collision_data else None
                
                _plot_trajectory_with_collision(
                    pred_x, pred_y, obstacles, oob,
                    color=VALID_PRED_COLOR, alpha=0.7,
                    label='Predicted'
                )
            
            plt.xlabel('X')
            plt.ylabel('Y')
            title = f'Trajectory Visualization (Batch {batch_idx}, Agent {agent_idx})'
            plt.title(title)
            plt.legend()
            plt.axis('equal')
            plt.tight_layout()
            
            plot_path = os.path.join(output_dir, f'trajectory_batch_{batch_idx}_agent_{agent_idx}.png')
            plt.savefig(plot_path)
            plt.close()
            
            total_plots += 1
    
    print(f"Saved {total_plots} plots to {output_dir}")


def visualize_vae_samples(
    coord_grid: torch.Tensor,
    target_seq: torch.Tensor,
    preds_list: List[torch.Tensor],
    past_seq: torch.Tensor,
    output_dir: str,
    map_image: Optional[torch.Tensor] = None,
    top_k: Optional[int] = None,
):
    """
    Convenience wrapper for VAE multi-sample visualization.
    
    Args:
        preds_list: List of n_samples tensors, each (b, a, t, 2)
    """
    # Stack: [n_samples, b, a, t, 2] -> permute to [b, a, t, n_samples, 2]
    preds_all = torch.stack(preds_list, dim=0).permute(1, 2, 3, 0, 4)
    b, a, t, n_samples, _ = preds_all.shape
    
    # Uniform probabilities
    mode_probs = torch.full((b, a, n_samples), 1.0 / n_samples, device=preds_all.device)
    
    if top_k is None:
        top_k = n_samples
    
    visualize_trajectories(
        coord_grid=coord_grid,
        target_seq=target_seq,
        predicted_seq=preds_all,
        past_seq=past_seq,
        output_dir=output_dir,
        map_image=map_image,
        mode_probs=mode_probs,
        top_k=top_k
    )