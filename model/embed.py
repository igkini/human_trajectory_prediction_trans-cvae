
import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class SinusoidalEmbeddingLayer(nn.Module):
    """Sinusoidal Positional Embedding for xyz and time."""

    def __init__(self, min_freq=4, max_freq=256, hidden_size=10):
        super().__init__()
        self.min_freq = float(min_freq)
        self.max_freq = float(max_freq)
        self.hidden_size = hidden_size
        
        if hidden_size % 2 != 0:
            raise ValueError(f'hidden_size ({hidden_size}) must be divisible by 2.')
        
        self.num_freqs_int32 = hidden_size // 2
        self.num_freqs = float(self.num_freqs_int32)
        
        log_freq_increment = (
            math.log(float(self.max_freq) / float(self.min_freq)) /
            max(1.0, self.num_freqs - 1)
        )
        inv_freqs = self.min_freq * torch.exp(
            torch.arange(self.num_freqs, dtype=torch.float32) * -log_freq_increment
        )
        self.register_buffer('inv_freqs', inv_freqs)
    
    def forward(self, input_tensor):

        """
            Input: [B, A, T, F] --> [B, A, T, F, 1] --> [B, A, T, F, hidden/2]
            Output: [B, A, T, F, hidden]
        """

        input_tensor = input_tensor.unsqueeze(-1).repeat_interleave(self.num_freqs_int32, dim=-1)
        embedded = torch.cat([
            torch.sin(input_tensor * self.inv_freqs),
            torch.cos(input_tensor * self.inv_freqs)
        ], dim=-1)

        return embedded

class AgentTemporalEncoder(nn.Module):
  """Encodes agents temporal positions."""

  def __init__(self, 
               key, 
               params,
               output_size):
    super().__init__()
    self.key = key

    self.embedding_layer = SinusoidalEmbeddingLayer(
        min_freq=0.5,
        max_freq=2*params.seq_len,
        hidden_size=params.sin_embedding_size)

    self.mlp = nn.Linear(
            in_features=params.sin_embedding_size,
            out_features=output_size,
            bias=True
        )

  def _get_temporal_embedding(self, input_batch):
    
    if self.key is not None:
        b, a, t, _ = input_batch[self.key].shape
    
    else:
        b, a, t, _ = input_batch.shape

    t = torch.arange(0, t, dtype=torch.float32, device=input_batch[self.key].device)
    t = t[None, None, :]  # Add batch and agent dimensions
    t = t.repeat(b, a, 1)  # Expand to [b, num_agents, num_steps]
    # print(t.unsqueeze(-1).shape)

    return self.embedding_layer(t) #[b, num_agents, num_steps, feature_embedding_size]
    
  def forward(self, input_batch):
        
        temporal_embedding = self._get_temporal_embedding(input_batch)
        # print(temporal_embedding.shape)
        mlp_output = self.mlp(temporal_embedding)
        return mlp_output

class Agent2DOrientationEncoder(nn.Module):
    """Encodes agents 2d orientation. Input should be given in radians"""
    
    def __init__(self, key, output_shape, params):
        super().__init__()
        self.key = key
        self.embedding_layer = SinusoidalEmbeddingLayer(
            max_freq=2,
            hidden_size=params.feature_embedding_size//2)
        
        self.mlp = nn.Linear(
            in_features=params.feature_embedding_size, 
            out_features=output_shape,
            bias=True
        )
    
    def forward(self, input_batch):
        orientation = input_batch[self.key]
        orientation_embedding = torch.cat([
            self.embedding_layer(torch.sin(orientation)),
            self.embedding_layer(torch.cos(orientation))
        ], dim=-1)
        
        not_is_hidden = torch.logical_not(input_batch['is_hidden'])
        mask = torch.logical_and(input_batch[f'has_data/{self.key}'], not_is_hidden)
        
        return self.mlp(orientation_embedding), mask

class AgentPositionEncoder(nn.Module):
    """Encodes agents spatial positions with optional masking."""
    def __init__(self, 
                 key,  
                 params,
                 output_size):
        super().__init__()
        self.key = key
        self.mask_key = f'{key}/mask'  # e.g., 'human_pos/mask'
        self.coord_scale = params.grid_coord_scale
        
        self.sin_emb_layer = SinusoidalEmbeddingLayer(
            min_freq=0.1,
            max_freq=2*params.grid_coord_scale,
            hidden_size=params.sin_embedding_size)
        
        self.mask_emb_layer = nn.Linear(
            in_features=1,
            out_features=output_size,
            bias=True)
        
        self.ff_layer2 = nn.Linear(
            in_features=params.sin_embedding_size,
            out_features=output_size,
            bias=True)

        self.lnorm = nn.LayerNorm(output_size)
    
    def forward(self, input_batch):
        # Get positions
        positions = input_batch[self.key]  # [b, a, t, 2]
        # Encode positions with sinusoidal embeddings
        pos_emb = self.sin_emb_layer(positions)  # [b, a, t, 2, h1]
        pos_emb = self.ff_layer2(pos_emb)  # [b, a, t, 2, h]
        
        if self.mask_key in input_batch:
            mask = input_batch[self.mask_key] # [b, a, t, 1]

            pos_emb = pos_emb * mask.unsqueeze(-2)

            combined = pos_emb
            # combined = self.lnorm(combined)

        else:
            combined = pos_emb

        return combined
    
class StationPositionEncoder(nn.Module):
    """Encodes station spatial positions."""

    def __init__(self, 
                 key,  
                 params,
                 output_size):
        super().__init__()
        self.key = key
        self.coord_scale = params.grid_coord_scale
        self.sin_emb_layer = SinusoidalEmbeddingLayer(
            min_freq=0.1,
            max_freq=2*params.grid_coord_scale,
            hidden_size=params.sin_embedding_size)
        
        self.mask_emb_layer = nn.Linear(
            in_features=1,
            out_features=output_size,
            bias=True)

        self.ff_layer1 = nn.Linear(
            in_features=2,
            out_features=output_size,
            bias=True
        )

        self.ff_layer2 = nn.Linear(
            in_features=params.sin_embedding_size,
            out_features=output_size,
            bias=True
        )

        self.lnorm = nn.LayerNorm(output_size)

    def forward(self, input_batch):
        # input_batch[self.key] shape: [b, s, t, 2]
        pos_input = input_batch[self.key]
        b, s, t, _ = pos_input.shape
        
        # Apply sinusoidal embedding
        pos_emb = self.sin_emb_layer(pos_input)  # [b, s, t, 2, sin_emb_size]
        
        # Apply ff_layer2
        pos_emb = self.ff_layer2(pos_emb)  # [b, s, t, 2, h]
        
        # Reshape to [b, 1, t, 2*s, h]
        # We need to interleave the s and 2 dimensions
        pos_emb = pos_emb.permute(0, 2, 1, 3, 4)  # [b, t, s, 2, h]
        pos_emb = pos_emb.reshape(b, t, 2*s, -1)  # [b, t, 2*s, h]
        pos_emb = pos_emb.unsqueeze(1)  # [b, 1, t, 2*s, h]
        
        return pos_emb

class AgentScalarEncoder(nn.Module):
    """Encodes a agent's scalar."""
    
    def __init__(self, key, params, output_shape):
        super().__init__()
        self.key = key
        self.scale=1.6
        # Input size depends on the scalar feature dimension
        self.mlp = nn.Linear(
            in_features=1,  # assuming scalar input
            out_features=output_shape,
            bias=True
        )
    
    def forward(self, input_batch):
        
        velocity=input_batch[self.key]/self.scale
        mlp_output = self.mlp(velocity.unsqueeze(-1))
        mlp_output = mlp_output.unsqueeze(-2)  
        
        return mlp_output

class AgentOneHotEncoder(nn.Module):
    
    def __init__(self, key, params, output_shape):
        super().__init__()
        self.key = key
        self.depth = params.depth
        self.seq_len=params.seq_len
        self.mlp = nn.Linear(
            in_features=self.depth,
            out_features=output_shape,
            bias=True
        )

    def forward(self, input_batch):
        # Create one-hot encoding
        stage_one_hot = F.one_hot(
            input_batch[self.key].squeeze(-1).long(), 
            num_classes=self.depth
        ).float()

        # not_is_hidden = input_batch['missing_data']
        # mask = torch.logical_and(input_batch[f'has_data/{self.key}'], not_is_hidden)
        
        mlp_output = self.mlp(stage_one_hot)

        mlp_output = mlp_output.repeat(1,1,self.seq_len,1)
        mlp_output = mlp_output.unsqueeze(-2) 
        
        return mlp_output

class AgentKeypointsEncoder(nn.Module):
    """Encodes the agent's keypoints."""
    
    def __init__(self, 
                 key, 
                 params,
                 output_size):
        super().__init__()
        self.key = key
        
        self.ff_layer1 =nn.Linear(
            in_features=params.keypoints_dim if hasattr(params, 'keypoints_dim') else 3,
            out_features=params.kpt_hidden,
            )
        
        self.kpt_attn=nn.MultiheadAttention(
            embed_dim=params.kpt_hidden, 
            num_heads=params.num_heads,
            batch_first=True,
            dropout=params.drop_prob)
        
        self.attn_ln=nn.LayerNorm(params.kpt_hidden)

        self.ff_layer2 =nn.Linear(in_features=25*params.kpt_hidden, out_features=2*params.ff_dim)
        self.ff_layer3 = nn.Linear(in_features=2*params.ff_dim, out_features=output_size)
        
        self.ff_ln=nn.LayerNorm(output_size)
        self.dropout=nn.Dropout(params.drop_prob)
    
    def forward(self, input_batch):
        
        # input['keypoints']: (b,a,t,k,3)
        keypoints = input_batch[self.key]
        kpt_emb = self.ff_layer1(keypoints) #(b,a,t,k,kpt_hidden)

        b,a,t,k,f= kpt_emb.shape

        kpt_emb = kpt_emb.reshape(b*a*t,k,f)
        kpt_mask = (input_batch[f"{self.key}/mask"]).squeeze(-1)
        kpt_mask=kpt_mask.view(b*a*t,k)

        attn_out, _=self.kpt_attn(
            query=kpt_emb,
            key=kpt_emb,
            value=kpt_emb,
            key_padding_mask=kpt_mask)


        attn_out=self.attn_ln(kpt_emb + attn_out)
        attn_out=attn_out.reshape(b,a,t,k*f)
        
        kpt_emb=self.ff_layer2(attn_out)
        kpt_emb=F.gelu(kpt_emb)
        kpt_emb=self.ff_layer3(kpt_emb)
        kpt_emb=self.dropout(kpt_emb)
        # kpt_emb=self.ff_ln(kpt_emb+attn_out)

        return kpt_emb.unsqueeze(-2)

class AgentHeadOrientationEncoder(nn.Module):
    """Encodes the detection stage."""
    
    def __init__(self, key, output_shape, params):
        super().__init__()
        self.key = key
        
        input_features = params.head_orientation_dim if hasattr(params, 'head_orientation_dim') else 3
        
        self.mlp = nn.Linear(
            in_features=input_features,
            out_features=output_shape,
            bias=True
        )
    
    def forward(self, input_batch):
        not_is_hidden = torch.logical_not(input_batch['is_hidden'])
        mask = torch.logical_and(input_batch[f'has_data/{self.key}'], not_is_hidden)
        
        mlp_output = self.mlp(input_batch[self.key])
        mlp_output = mlp_output.unsqueeze(-2)
        
        return mlp_output, mask