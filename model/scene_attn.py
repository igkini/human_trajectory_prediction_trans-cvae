import torch.nn as nn
import torch.nn.functional as F
from .attention import PostAttentionFFLayer
import torch

class ConvSceneCrossAttnTransformerLayer(nn.Module):
    """Performs cross-attention between the occupancy grid and agents."""
    
    def __init__(
        self,
        params,
        hidden_size
    ):
        super().__init__()
        
        if hidden_size % params.num_heads != 0:
            raise ValueError(
                f'hidden_size ({hidden_size}) must be an integer times'
                f' bigger than num_heads ({params.num_heads}).'
            )
        
        self.attn_layer = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=params.num_heads,
            dropout=params.drop_prob, 
            batch_first=True
        )
        
        self.ff_layer=PostAttentionFFLayer(params=params, hidden_size=hidden_size)
    
    def forward(self, input_batch, scene_enc):
        
        # [b, a, t, h]
        b, a, t, h = input_batch.shape
        
        # b, H= scene_ctx.shape
        scene_enc=scene_enc.unsqueeze(1).unsqueeze(1)
        scene_enc = scene_enc.expand(b, a, 1, h)
        query = input_batch.reshape(b * a, t, h)
        key_value = scene_enc.reshape(b * a, 1, h)
        
        # Cross-attention: each agent-timestep queries all scene elements
        attn_out,_ = self.attn_layer(
            query=query,
            key=key_value,
            value=key_value,
            average_attn_weights=True
        )
        
        # attn_out shape: [b*a, t, h]
        
        attn_out = attn_out.reshape(b, a, t, h)
        out = self.ff_layer(input_batch, attn_out)
        
        return out


class PatchSceneCrossAttnTransformerLayer(nn.Module):
    """Performs cross-attention between the occupancy grid and agents."""
    
    def __init__(
        self,
        params,
        hidden_size,
        expansion_factor=None
    ):
        super().__init__()
        
        if hidden_size % params.num_heads != 0:
            raise ValueError(
                f'hidden_size ({hidden_size}) must be an integer times'
                f' bigger than num_heads ({params.num_heads}).'
            )
        
        self.attn_layer = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=params.num_heads,
            dropout=params.drop_prob,
            batch_first=True
        )
        
        self.ff_layer=PostAttentionFFLayer(params=params, hidden_size=hidden_size, expansion_factor=expansion_factor)
    
    def forward(self, input_batch, scene_enc):
        
        # [b, a, t, h]
        if input_batch.dim()==5:
            b, a, t, n, h = input_batch.shape
            query = input_batch.reshape(b*a, t*n, h)

        elif input_batch.dim()==4:
            b, a, t, h = input_batch.shape
            query = input_batch.reshape(b * a, t, h)
        else:
            assert f"Input dimension is neither 4 nor 5"
        
        # print(scene_enc.shape)

        _, _, num_p, _= scene_enc.shape
        # b, P*P, H= scene_ctx.shape

        key = scene_enc.reshape(b * a, num_p, h)
        
        attn_out,_ = self.attn_layer(
            query=query,
            key=key,
            value=key,
            # average_attn_weights=True
        )
        
        # attn_out shape: [b*a, t, h]
        if input_batch.dim()==5:
            attn_out = attn_out.reshape(b, a, t, n, h)

        elif input_batch.dim()==4:
            attn_out = attn_out.reshape(b, a, t, h)

        out = self.ff_layer(input_batch, attn_out)

        return out
    
class PatchContextAttnTransformerLayer(nn.Module):
    """Performs cross-attention between the occupancy grid and agents."""
    
    def __init__(
        self,
        params,
        hidden_size,
        expansion_factor=None,
        downsample_factor=None
    ):
        super().__init__()
        
        self.downsample_factor = downsample_factor
        if hidden_size % params.num_heads != 0:
            raise ValueError(
                f'hidden_size ({hidden_size}) must be an integer times'
                f' bigger than num_heads ({params.num_heads}).'
            )
        
        self.attn_layer = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=params.num_heads,
            dropout=params.drop_prob,
            batch_first=True
        )
        self.ff_layer = PostAttentionFFLayer(
            params=params, 
            hidden_size=hidden_size, 
            expansion_factor=expansion_factor
        )
        
    def forward(self, input_batch, scene_enc):
        b, a, t, h = input_batch.shape
        
        # Check if scene_enc is agent-specific (4D) or agent-invariant (3D)
        if scene_enc.dim() == 4:
            # Agent-specific: (b, ap, num_p, h)
            _, ap, num_p, _ = scene_enc.shape
            query = scene_enc.reshape(b, num_p * ap, h)
            is_agent_specific = True
        else:
            # Agent-invariant: (b, num_p, h)
            num_p = scene_enc.shape[1]
            query = scene_enc  # Use as is
            is_agent_specific = False
        
        # Key and value from input_batch
        key = input_batch.reshape(b, t * a, h)
        
        attn_out, attn_weights = self.attn_layer(
            query=query,
            key=key,
            value=key,
        )
        
        # Reshape attention output back if agent-specific
        if is_agent_specific:
            attn_out = attn_out.reshape(b, ap, num_p, h)
        
        # Feed-forward layer
        out = self.ff_layer(scene_enc, attn_out)
        
        # Optional downsampling based on attention importance
        if self.downsample_factor is not None:
            if is_agent_specific:
                # attn_weights: (b, num_p*ap, t*a) → (b, ap, num_p, t*a)
                attn_weights = attn_weights.reshape(b, ap, num_p, t * a)
                importance = attn_weights.mean(dim=-1)  # (b, ap, num_p)
                
                k = num_p // self.downsample_factor
                top_k_indices = importance.topk(k, dim=-1).indices  # (b, ap, k)
                
                out = torch.gather(
                    out,
                    dim=2,
                    index=top_k_indices.unsqueeze(-1).expand(-1, -1, -1, h)
                )  # (b, ap, k, h)
            else:
                # attn_weights: (b, num_p, t*a)
                importance = attn_weights.mean(dim=-1)  # (b, num_p)
                
                k = num_p // self.downsample_factor
                top_k_indices = importance.topk(k, dim=-1).indices  # (b, k)
                
                out = torch.gather(
                    out,
                    dim=1,
                    index=top_k_indices.unsqueeze(-1).expand(-1, -1, h)
                )  # (b, k, h)
        
        return out
    
class PatchSelfAttnLayer(nn.Module):
    """Performs cross-attention between the occupancy grid and agents."""
    
    def __init__(
        self,
        params,
        hidden_size,
        expansion_factor=None,
        up_factor=1
    ):
        super().__init__()
        

        if hidden_size % params.num_heads != 0:
            raise ValueError(
                f'hidden_size ({hidden_size}) must be an integer times'
                f' bigger than num_heads ({params.num_heads}).'
            )
        
        self.attn_layer = nn.MultiheadAttention(
            embed_dim=hidden_size*up_factor,
            num_heads=params.num_heads*1,
            dropout=params.drop_prob,
            batch_first=True
        )

        self.ff_layer=PostAttentionFFLayer(params=params, hidden_size=hidden_size*up_factor, expansion_factor=expansion_factor)
        # self.linear_down=nn.Linear(hidden_size*up_factor, hidden_size)
        # self.linear_up=nn.Linear(hidden_size, hidden_size*up_factor)

    def forward(self, scene_enc):
        
        # scene_enc=self.linear_up(scene_enc)
        if scene_enc.dim()==4:
            b, a, num_p, h = scene_enc.shape
            # scene_enc=scene_enc.reshape(b, a*num_p, h)
            query = scene_enc.reshape(b * a, num_p, h)
        else:
            b, num_p, h= scene_enc.shape
            query = scene_enc

        key = query

        attn_out, attn_weights = self.attn_layer(
            query=query,
            key=key,
            value=key,
        )

        if scene_enc.dim()==4:
            attn_out = attn_out.reshape(b, a, num_p, h)
            
        out = self.ff_layer(scene_enc, attn_out)
        # out=self.linear_down(out)

        return out