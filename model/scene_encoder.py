import torch
import torch.nn as nn
import torch.nn.functional as F
from .embed import SinusoidalEmbeddingLayer
from .attention import PostAttentionFFLayer 


class ConvOccupancyGridEncoderLayer(nn.Module):
    """Uses the frame on the current step."""
    
    def __init__(self, 
                 params,
                 hidden_size):
        super().__init__()
        
        self.num_filters = params.num_conv_filters
        drop_prob = params.drop_prob
        self.coord_scale=params.coord_scale
        layers = []
        in_channels = 3 
        
        for i, num_filter in enumerate(self.num_filters):
            if i == 0 or i == 1:
                strides = 2
                use_pooling = False  # Don't pool when using stride=2
            else:
                strides = 1
                use_pooling = True  # Pool when stride=1
            
            conv_layer = nn.Conv2d(
                in_channels=in_channels,
                out_channels=num_filter,
                kernel_size=3,
                stride=strides,
                padding=1
            )
    
            layers.append(conv_layer)
            layers.append(nn.BatchNorm2d(num_filter))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout2d(params.drop_prob))
            
            if use_pooling:
                pooling_layer = nn.MaxPool2d(kernel_size=2, stride=1)
                layers.append(pooling_layer)
            
            in_channels = num_filter
        
        # Flatten
        layers.append(nn.Flatten())
        layers.append(nn.LazyLinear(hidden_size))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(drop_prob))
        layers.append(nn.LayerNorm(hidden_size))
        
        self.seq_layers = nn.Sequential(*layers)
    
    def forward(self, occ_grid, coord_grid):
        
        coord_grid = coord_grid/self.coord_scale
        
        grid = torch.cat([occ_grid, coord_grid], dim=1)
        
        # Apply convolutional layers to grid
        occ_grid = self.seq_layers(grid)
        
        out = occ_grid

        return out

class ConvGridEncoderLayer(nn.Module):
    """Uses the frame on the current step."""
    
    def __init__(self, 
                 params):
        super().__init__()
        
        self.num_filters = params.num_conv_filters
        drop_prob = params.drop_prob
        layers = []
        in_channels = 1
        
        for i, num_filter in enumerate(self.num_filters):
            # if i == 0 or i == 1:
            #     strides = 2
            #     use_pooling = False  
            # else:
            #     strides = 1
            #     use_pooling = False
            
            conv_layer = nn.Conv2d(
                in_channels=in_channels,
                out_channels=num_filter,
                kernel_size=3,
                stride=1,
                padding=1
            )

            layers.append(conv_layer)
            layers.append(nn.BatchNorm2d(num_filter))
            layers.append(nn.SiLU())
            layers.append(nn.Dropout2d(drop_prob))
            
            # if use_pooling:
            #     pooling_layer = nn.MaxPool2d(kernel_size=2, stride=1)
            #     layers.append(pooling_layer)
            
            in_channels = num_filter
        
        self.seq_layers = nn.Sequential(*layers)
    
    def forward(self, input_batch):

        # input_batch = input_batch.copy()
        occ_grid = self.seq_layers(input_batch)
        
        return occ_grid

class TransformerOccupancyGridEncoderLayer(nn.Module):
    """Uses the frame on the current step."""
    
    def __init__(self, 
                 params,
                 prefilter=False,
                 hidden_size=256,
                 position_encoding='origin'):
        super().__init__()
        
        self.patch_size=params.patch_size
        self.prefilter=prefilter
        self.position_encoding=position_encoding
        self.grid_coord_scale=params.grid_coord_scale
        self.grid_channels=3
        self.hidden_size=hidden_size

        if position_encoding=='origin':
            self.num_channels=1
        elif self.position_encoding=='grid':
            self.num_channels=self.grid_channels
        
        if self.prefilter:
            self.filter=ConvGridEncoderLayer(params)
        else:
            self.filter=nn.Identity()
        
        self.occ_embedding_layer=nn.Linear(self.num_channels*(self.patch_size**2), hidden_size)
        
        self.sin_embedding_layer=SinusoidalEmbeddingLayer(
            min_freq = 0.1,
            max_freq = 2*params.grid_coord_scale, 
            hidden_size=params.sin_embedding_size)
        
        self.orig_embedding_layer=nn.Linear(2*params.sin_embedding_size, hidden_size)
        self.concat_projection=nn.Linear(2*hidden_size, hidden_size)
        self.pre_lnorm=nn.LayerNorm(hidden_size)

    def _create_patches(self, grid):
        patches=grid.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        patches=patches.permute(0, 2, 3, 1, 4, 5)

        return patches

    def _flatten_patches(self, occ_grid):
        
        b=occ_grid.shape[0]
        
        occ_patches=self._create_patches(occ_grid)
        flat_occ_patches=occ_patches.reshape(b, -1, self.num_channels * self.patch_size * self.patch_size)
        
        return flat_occ_patches

    def _create_coord_origins(self, coord_grid):
        coord_patches = self._create_patches(coord_grid)
        center_coordinates = coord_patches[:, :, :, :, self.patch_size // 2, self.patch_size // 2] 
        b, num_h, num_w, _ = center_coordinates.shape
        center_coordinates = center_coordinates.reshape(b, num_h * num_w, 2)  # Flatten to (B, H*W, 2)

        return center_coordinates
    
    def _embed_coords(self,scene_coord):
        b, c, h, w = scene_coord.shape
    
        scene_coord_flat = scene_coord.permute(0, 2, 3, 1).reshape(-1, c)
        
        # Apply embedding: [B*H*W, 2] -> [B*H*W, 2, sin_embedding_size]
        scene_coord_embedded = self.sin_embedding_layer(scene_coord_flat)
        scene_coord_embedded = scene_coord_embedded.flatten(-2)
        # Reshape back to [B, 2*sin_embedding_size, H, W]
        scene_coord_embedded = scene_coord_embedded.reshape(b, h, w, -1).permute(0, 3, 1, 2)

        return scene_coord_embedded
 
    
    def forward(self, input_batch, scene_grid, scene_coord):
        
        # human_pos=input_batch['human_pos']# get last timestep
        # b=input_batch.shape[0]
        # a=input_batch.shape[1]
            
        scene_coord=scene_coord/self.grid_coord_scale
        filtered_occ_grid=self.filter(scene_grid)

        if self.position_encoding=='origin':
            
            flat_patches=self._flatten_patches(filtered_occ_grid)
            occ_embedding=self.occ_embedding_layer(flat_patches)
            patch_origins=self._create_coord_origins(scene_coord)
            origin_embeddings=self.sin_embedding_layer(patch_origins)
            origin_embeddings=origin_embeddings.flatten(-2)
            origin_embeddings = self.orig_embedding_layer(origin_embeddings)
            x=occ_embedding+origin_embeddings
            # x = torch.cat([occ_embedding, origin_embeddings], dim=-1)
            # x = self.concat_projection(x)
            x=self.pre_lnorm(x)
        
        elif self.position_encoding=='grid':
            
            # embedded_coords=self._embed_coords(scene_coord)
            stacked_grid = torch.cat([scene_grid, scene_coord], dim=1)

            flat_patches=self._flatten_patches(stacked_grid)
            occ_embedding=self.occ_embedding_layer(flat_patches)
            x=occ_embedding

        scene_enc = x

        return scene_enc

class SequenceFilterLayer(nn.Module):
    def __init__(self, params, hidden_size, kernel_size, stride, padding, pool=True, output_size=None, dilation=1):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        
        bottleneck_size = hidden_size // 2
        if output_size is None:
            self.output_size = hidden_size
        else:
            self.output_size = output_size
            
        # Bottleneck: compress -> convolve -> expand
        self.conv_compress = nn.Conv1d(hidden_size, bottleneck_size, kernel_size=1, dilation=self.dilation)
        self.conv_main = nn.Conv1d(bottleneck_size, bottleneck_size, kernel_size=kernel_size, stride=stride, padding=padding, dilation=int(self.dilation))
        self.conv_expand = nn.Conv1d(bottleneck_size, self.output_size, kernel_size=1, dilation=self.dilation)
        
        self.conv_norm = nn.LayerNorm(self.output_size)
        
        # Shortcut connection: needs projection if stride > 1 OR if channel counts mismatch
        if stride > 1 or hidden_size != self.output_size:
            self.shortcut = nn.Conv1d(hidden_size, self.output_size, kernel_size=1, stride=stride)
        else:
            self.shortcut = nn.Identity()
        
        # Feedforward block
        self.ff_layer1 = nn.Linear(self.output_size, self.output_size // 2)
        self.ff_layer2 = nn.Linear(self.output_size // 2, self.output_size)
        self.ff_norm = nn.LayerNorm(self.output_size)
        
        # Pooling
        if pool:
            self.max_pool = nn.MaxPool1d(kernel_size=self.kernel_size, stride=self.stride)
        else:
            self.max_pool = nn.Identity()
            
    def forward(self, input):
        if input.dim() == 4:
            b, a, t, h = input.shape
            input_reshaped = input.reshape(b*a, t, h)
            x = input_reshaped.transpose(2, 1)
        else:
            x = input.transpose(2, 1)
        
        identity = self.shortcut(x)
        
        # Bottleneck convolutions
        conv_out = F.silu(self.conv_compress(x))
        conv_out = F.silu(self.conv_main(conv_out))
        conv_out = self.conv_expand(conv_out)
        
        conv_out = conv_out.transpose(1, 2)
        identity = identity.transpose(1, 2)
        
        # (B, T, 256) + (B, T, 256)
        x = self.conv_norm(conv_out + identity)
        
        # Feedforward block
        ff_out = self.ff_layer1(x)
        ff_out = F.gelu(ff_out)
        ff_out = self.ff_layer2(ff_out)
        x = self.ff_norm(ff_out + x)
        
        # Pooling
        x = x.transpose(2, 1)
        x = self.max_pool(x)
        x = x.transpose(1, 2)
        
        if input.dim() == 4:
            _, t_pooled, h = x.shape
            x = x.reshape(b, a, t_pooled, h)

        return x