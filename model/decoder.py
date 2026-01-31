from . import agent_encoder, attention, scene_attn
from .attention import PostAttentionFFLayer
import torch, torch.nn as nn

class LearnedPredictionDecoder(nn.Module):
    def __init__(self, 
                 params, 
                 hidden_size):
        super().__init__()
        self.pred_len = params.pred_len
        
        self.learned_embedding = nn.Parameter(
            torch.empty(1, 1, self.pred_len, hidden_size)
        )
        nn.init.uniform_(self.learned_embedding, -1.0, 1.0)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=params.num_heads,
            batch_first=True,
            dropout=params.drop_prob
        )

        self.ff_layer=PostAttentionFFLayer(params=params, hidden_size=hidden_size)
        
    def forward(self, input_batch):
        
        b, a, t, h = input_batch.shape
        
        query = self.learned_embedding.expand(b, a, -1, -1)  # [b, a, pred_len, h]
        
        query_reshaped = query.reshape(b * a, self.pred_len, h)
        key = input_batch.reshape(b * a, t, h)

        attn_out, _ = self.cross_attn(
            query=query_reshaped,
            key=key,
            value=key,
        )

        attn_out=attn_out.reshape(b, a, self.pred_len, h)
        
        out=self.ff_layer(query, attn_out)
        
        return out  # [b, a, pred_len, h]

class LearnedAdditivePredictionDecoder(nn.Module):
    def __init__(self, 
                 params,
                 hidden_size):
        super().__init__()
        self.pred_len = params.pred_len
        
        self.learned_embedding = nn.Parameter(
            torch.empty(1, 1, self.pred_len, hidden_size)
        )
        nn.init.uniform_(self.learned_embedding, -1.0, 1.0)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=params.num_heads,
            batch_first=True,
            dropout=params.drop_prob
        )

        self.ff_layer=PostAttentionFFLayer(params=params, hidden_size=hidden_size)
        
    def forward(self, input_batch):
        
        b, a, t, h = input_batch.shape
        
        query = input_batch + self.learned_embedding.expand(b, a, self.pred_len, -1)  # [b, a, pred_len, h]
        
        query_reshaped = query.reshape(b * a, self.pred_len, h)
        key = input_batch.reshape(b * a, t, h)
        
        attn_out, _ = self.cross_attn(
            query=query_reshaped,
            key=key,
            value=key,
        )

        attn_out=attn_out.reshape(b, a, self.pred_len, h)
        
        out=self.ff_layer(query, attn_out)
        
        return out  # [b, a, pred_len, h]
    
    def __init__(self, params, hidden_size):
        super().__init__()

        self.pred_len=params.pred_len

        self.human_agent_encoding_layer = agent_encoder.FeatureAddAgentEncoderLayer(
            agent_type='human',
            hidden_size=128,
            output_size=hidden_size,
            params=params)
        
        self.masked_self_attn=attention.SelfAttnTransformerLayer(params=params, hidden_size=hidden_size,mask_type='causal')
        self.encoder_cross_attn_layer=attention.CrossAttnTransformerLayer(params=params, hidden_size=hidden_size)
        self.scene_cross_attn_layer=scene_attn.PatchSceneCrossAttnTransformerLayer(params=params, hidden_size=hidden_size)

    def forward(self, target, encoder_input, scene_encoder, training=True):
        
        b, a, _, h = encoder_input.shape
        
        if training:
            # Teacher forcing: use ground truth
            learned_sequence = self.human_agent_encoding_layer(target)
            learned_sequence = learned_sequence.view(b*a, self.pred_len, h)
        else:
            # Inference: autoregressive generation
            learned_sequence = self.human_agent_encoding_layer(target[:, :, -1:])  # Last position
            learned_sequence = learned_sequence.view(b*a, 1, h)
            
            for _ in range(self.pred_len - 1):
                out = self.masked_self_attn(learned_sequence)
                out = self.scene_cross_attn_layer(out, scene_encoder)
                out = self.encoder_cross_attn_layer(out, encoder_input)
                
                # Predict next position (need a prediction head here!)
                next_pos = self.prediction_head(out[:, -1:])  # [b*a, 1, 2]
                
                # Encode and append
                next_encoded = self.human_agent_encoding_layer(next_pos.view(b, a, 1, 2))
                learned_sequence = torch.cat([learned_sequence, next_encoded.view(b*a, 1, h)], dim=1)
        
        out = self.masked_self_attn(learned_sequence)
        out = self.scene_cross_attn_layer(out, scene_encoder)
        out = self.encoder_cross_attn_layer(out, encoder_input)
        
        return out