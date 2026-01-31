from typing import Tuple
import torch
import torch.nn.functional as F


def check_trajectory_collisions(
    predicted_trajectory: torch.Tensor,  # (b, a, t, 2)
    raw_grid: torch.Tensor,              # (b, 1, h, w)
    scene_coord: torch.Tensor,           # (b, 2, h, w)
    obstacle_threshold: float = 0.9
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Check which trajectory points are in obstacles or out of bounds.
    
    Samples the occupancy grid at predicted trajectory positions to determine
    collision status. Grid convention: lower values = obstacles, higher = free space.
    
    Args:
        predicted_trajectory: Predicted trajectory positions in world coordinates
        raw_grid: Occupancy grid (lower values = obstacles, higher = free)
        scene_coord: Maps grid cells to world coordinates
        obstacle_threshold: Values below this are considered obstacles
    
    Returns:
        is_in_obstacle: (b, a, t) boolean mask for obstacle collisions
        is_out_of_bounds: (b, a, t) boolean mask for out-of-bounds points
    """
    b, a, t, _ = predicted_trajectory.shape
    
    # Compute scene bounds from coordinate grid
    x_min = scene_coord[:, 0, ...].reshape(b, -1).min(dim=1, keepdim=True)[0]
    x_max = scene_coord[:, 0, ...].reshape(b, -1).max(dim=1, keepdim=True)[0]
    y_min = scene_coord[:, 1, ...].reshape(b, -1).min(dim=1, keepdim=True)[0]
    y_max = scene_coord[:, 1, ...].reshape(b, -1).max(dim=1, keepdim=True)[0]
    
    # Extract x and y coordinates
    traj_x, traj_y = predicted_trajectory.split(1, dim=-1)  # (b, a, t, 1) each
    
    # Reshape bounds for broadcasting
    x_min, x_max = x_min.view(b, 1, 1, 1), x_max.view(b, 1, 1, 1)
    y_min, y_max = y_min.view(b, 1, 1, 1), y_max.view(b, 1, 1, 1)
    
    # Normalize to [-1, 1] range for grid_sample
    eps = 1e-6
    norm_x = 2 * (traj_x - x_min) / (x_max - x_min + eps) - 1
    norm_y = 2 * (traj_y - y_min) / (y_max - y_min + eps) - 1
    
    # Combine normalized coordinates (flip y for grid convention)
    normalized_trajectory = torch.cat([norm_x, -norm_y], dim=-1)  # (b, a, t, 2)
    
    # Check for out of bounds points
    out_of_bounds = (norm_x < -1) | (norm_x > 1) | (norm_y < -1) | (norm_y > 1)
    out_of_bounds = out_of_bounds.squeeze(-1)  # (b, a, t)
    
    # Reshape for grid_sample: (b, 1, a*t, 2)
    grid_sampler = normalized_trajectory.reshape(b, 1, a * t, 2)
    
    # Sample the occupancy grid at trajectory positions
    sampled = F.grid_sample(
        input=raw_grid.float(),
        grid=grid_sampler,
        mode='bilinear',
        padding_mode='border',
        align_corners=False
    )  # (b, 1, 1, a*t)
    
    # Check if points are in obstacles (values below threshold)
    is_obstacle = (sampled < obstacle_threshold)
    is_obstacle = is_obstacle.reshape(b, a, t)
    
    return is_obstacle, out_of_bounds