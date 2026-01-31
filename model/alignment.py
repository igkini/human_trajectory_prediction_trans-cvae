import torch
import torch.nn as nn
import torch.nn.functional as F
from .attention import PostAttentionFFLayer

class AgentSelfAlignmentLayer(nn.Module):
    def __init__(self,
                 params,
                 hidden_size):
        super().__init__()

        if hidden_size % params.num_heads != 0:
            raise ValueError(f'hidden_size ({hidden_size}) must be an integer '
                           f'times bigger than num_heads ({params.num_heads}).')
        
        self.attn_layer = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=params.num_heads,
            batch_first=True,
            dropout=params.drop_prob
        )
        
        self.ff_layer=PostAttentionFFLayer(params=params, hidden_size=hidden_size)
        
        # Learned query vector [1, 1, 1, h]
        self.learned_query_vec = nn.Parameter(
            torch.empty(1, 1, 1, hidden_size).uniform_(-1.0, 1.0)
        )
    
    def forward(self, input_batch):
        
        b, a, t, h = input_batch.shape

        # Build learned query [b, a, t, h]
        learned_query = self.learned_query_vec.repeat(b, a, t, 1)
        
        learned_query_reshaped = learned_query.view(b*a, t, h)
        key = input_batch.view(b*a, t, h)
        
        attn_out, _ = self.attn_layer(
            query=learned_query_reshaped,
            key=key,
            value=key,
            attn_mask=None,
        )
        
        attn_out = attn_out.view(b, a, t, h)
        
        out=self.ff_layer(input_batch, attn_out)

        return out