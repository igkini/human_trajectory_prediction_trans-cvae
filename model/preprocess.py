import torch
import torch.nn as nn


class GridPreprocessLayer(nn.Module):
    def __init__(self, params,scale_coords=False):
        super().__init__()
        self.output_size = params.grid_output_size
        self.coord_scale = params.grid_coord_scale
        self.scale_coords=scale_coords
        # Can add learnable parameters here if needed
        
    def create_coordinate_grid(self, batch_size, height, width, origin, resolution, device):
        """Create coordinate grids for a batch of maps."""
        # Create pixel coordinate grids
        y_coords = torch.arange(height, device=device, dtype=torch.float32)
        x_coords = torch.arange(width, device=device, dtype=torch.float32)

        pixel_y, pixel_x = torch.meshgrid(y_coords, x_coords, indexing='ij')
        
        pixel_x = pixel_x.unsqueeze(0).expand(batch_size, -1, -1)
        pixel_y = pixel_y.unsqueeze(0).expand(batch_size, -1, -1)
        origin_x = origin[:, 0:1].unsqueeze(-1)  
        origin_y = origin[:, 1:2].unsqueeze(-1)
        # print(origin_x.shape)

        resolution = resolution.unsqueeze(-1).unsqueeze(-1) 
        
        world_x = origin_x + pixel_x * resolution
        world_y = origin_y + (height - 1 - pixel_y) * resolution
        
        coord_grid = torch.stack([world_x, world_y], dim=1)
        
        if self.scale_coords:
            coord_grid=coord_grid / self.coord_scale
        else:
            coord_grid=coord_grid

        return coord_grid
    
    def forward(self, input_batch):
    
        map_images = input_batch['map_image']
        origins = input_batch['map_origin']
        resolutions = input_batch['map_resolution']
        
        batch_size = map_images.shape[0]
        device = map_images.device
        
        scene_coord = self.create_coordinate_grid(
            batch_size,
            self.output_size[0], 
            self.output_size[1],
            origins,
            resolutions,
            device
        )
        
        scene_grid = input_batch['map_image']
        # scene_grid = torch.where(scene_grid > 0.9, scene_grid, torch.zeros_like(scene_grid))
        
        return scene_grid, scene_coord