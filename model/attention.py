import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttnTransformerLayer(nn.Module):
    """Performs full self-attention across the agent and time dimensions."""
    
    def __init__(
        self,
        params,
        hidden_size,
        mask_type=None
    ):
        super().__init__()
        
        if hidden_size % params.num_heads != 0:
            raise ValueError(
                f'hidden_size ({hidden_size}) must be an integer '
                f'times bigger than num_heads ({params.num_heads}).'
            )
        
        self.mask_type=mask_type

        self.attn_layer = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=params.num_heads,
            dropout=params.drop_prob,
            batch_first=True
        )
        
        self.ff_layer=PostAttentionFFLayer(params=params, hidden_size=hidden_size)
    
    def forward(self, input_batch):
        
        # [b, a, t, h]
        b, a, t, h = input_batch.shape
        
        input_batch_reshaped = input_batch.reshape(b, a * t, h)
        
        if self.mask_type == 'causal':
            seq_len = a * t
            mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
            mask = mask.to(input_batch.device)
        else:
            mask = None

        attn_out, _ = self.attn_layer(
            input_batch_reshaped,
            input_batch_reshaped,
            input_batch_reshaped,
            attn_mask=mask,
        )

        attn_out = attn_out.reshape(b, a, t, h)
        
        out = self.ff_layer(input_batch, attn_out)
        
        return out

class CrossAttnTransformerLayer(nn.Module):
    """Performs full cross-attention across the temporal dimension only."""
    
    def __init__(
        self,
        params,
        hidden_size,
    ):
        super().__init__()
        
        if hidden_size % params.num_heads != 0:
            raise ValueError(
                f'hidden_size ({hidden_size}) must be an integer '
                f'times bigger than num_heads ({params.num_heads}).'
            )
        
        self.attn_layer = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=params.num_heads,
            dropout=params.drop_prob,
            batch_first=True
        )
        
        self.ff_layer=PostAttentionFFLayer(params=params, hidden_size=hidden_size)
    
    def forward(self, query_seq, key_seq):
        
        # [b, a, t, h]
        bq, aq, tq, hq = query_seq.shape

        bk, ak, tk, hk = key_seq.shape

        query_seq_reshaped = query_seq.reshape(bq*aq, tq, hq)
        key_seq_reshaped = key_seq.reshape(bk*ak, tk, hk)

        attn_out, _ = self.attn_layer(
            query_seq_reshaped,
            key_seq_reshaped,
            key_seq_reshaped,
        )

        attn_out = attn_out.reshape(bq, aq, tq, hq)
        
        out = self.ff_layer(query_seq, attn_out)
        
        return out

class AgentTypeCrossAttnTransformerLayer(nn.Module):
    """Performs full cross-attention across the agent and time dimensions."""
    
    def __init__(
        self,
        params,
        hidden_size,
    ):
        super().__init__()
        
        if hidden_size % params.num_heads != 0:
            raise ValueError(
                f'hidden_size ({hidden_size}) must be an integer '
                f'times bigger than num_heads ({params.num_heads}).'
            )
        
        self.attn_layer = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=params.num_heads,
            dropout=params.drop_prob,
            batch_first=True
        )
        
        self.ff_layer=PostAttentionFFLayer(params=params, hidden_size=hidden_size)
    
    def forward(self, query_seq, key_seq):
        
        # [b, a, t, h]
        bq, aq, tq, hq = query_seq.shape

        bk, ak, tk, hk = key_seq.shape

        query_seq_reshaped = query_seq.reshape(bq, tq*aq, hq)
        key_seq_reshaped = key_seq.reshape(bk, tk*ak, hk)

        attn_out, _ = self.attn_layer(
            query_seq_reshaped,
            key_seq_reshaped,
            key_seq_reshaped,
        )

        attn_out = attn_out.reshape(bq, aq, tq, hq)
        
        out = self.ff_layer(query_seq, attn_out)
        
        return out

class PostAttentionFFLayer(nn.Module):
    
    def __init__(self, 
                 params,
                 hidden_size,
                 expansion_factor=None):
        super().__init__()

        if expansion_factor is None:
            self.expansion_factor=params.ff_expansion_factor
        else:
            self.expansion_factor=expansion_factor

        self.attn_lnorm = nn.LayerNorm(hidden_size)
        self.ff_lnorm = nn.LayerNorm(hidden_size)
        
        ff_dim = int(hidden_size * self.expansion_factor)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, ff_dim),
            nn.GELU(),
            nn.Dropout(params.drop_prob),
            nn.Linear(ff_dim, hidden_size),
        )
        self.dropout = nn.Dropout(params.drop_prob)
    
    def forward(self, input, attn_out):
      
        x = self.attn_lnorm(input + attn_out)
        
        x = self.ff_lnorm(x + self.dropout(self.ffn(x)))
        
        return x