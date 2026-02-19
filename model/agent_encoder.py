from .embed import AgentTemporalEncoder
import torch
import torch.nn as nn
from typing import Dict

class FeatureConcatAgentEncoderLayer(nn.Module):

    def __init__(self,
                agent_type, 
                params,
                hidden_size,
                output_size
                ):
        super().__init__()
        
        agents_feature_config= getattr(params, f"{agent_type}_agents_feature_config")
        num_encoders = len(agents_feature_config) + 2
        input_size = num_encoders * (hidden_size)
        
        self.agent_feature_embedding_layers = nn.ModuleList()
        
        # Feature Embeddings
        for key, layer in agents_feature_config.items():
            self.agent_feature_embedding_layers.append(
                layer(key, params, hidden_size)
            )
        
        # Temporal Embedding
        self.agent_feature_embedding_layers.append(
            AgentTemporalEncoder(
                list(agents_feature_config.keys())[0],
                params,
                hidden_size
            )
        )

        self.ff_layer = nn.Linear(
            in_features=input_size,
            out_features=output_size,
            bias=True
        )

        self.lnorm=nn.LayerNorm(output_size)
        self.ff_dropout = nn.Dropout(params.drop_prob)

    def forward(self, input_batch: Dict[str, torch.Tensor]):
        
        layer_embeddings = []
        for layer in self.agent_feature_embedding_layers:
            layer_embedding = layer(input_batch)
            
            # flatten last two dimensions
            original_shape = layer_embedding.shape
            new_shape = original_shape[:-2] + (original_shape[-2] * original_shape[-1],)
            layer_embedding = layer_embedding.reshape(new_shape)

            layer_embeddings.append(layer_embedding)

        embedding = torch.cat(layer_embeddings, dim=-1)
        # Apply final feedforward layer
        out = self.ff_layer(embedding)
        out= self.lnorm(out)
        out = self.ff_dropout(out)
        
        return out



class FeatureAddAgentEncoderLayer(nn.Module):
    """
    MLP that connects all features
    """
    
    def __init__(self,
                 agent_type, 
                 params,
                 hidden_size,
                 output_size):
        super().__init__()
        
        self.hidden_size = hidden_size
        agents_feature_config = getattr(params, f"{agent_type}_agents_feature_config")
        
        self.num_encoders = len(agents_feature_config) + 1
        self.input_size = self.num_encoders * self.hidden_size
        
        self.agent_feature_embedding_layers = nn.ModuleList()
        
        # Feature Embeddings
        for key, layer in agents_feature_config.items():
            self.agent_feature_embedding_layers.append(
                layer(key, params, hidden_size)
            )
        
        # Temporal Embedding
        self.agent_temporal_embedding_layer = AgentTemporalEncoder(
            list(agents_feature_config.keys())[0],
            params,
            hidden_size
        )
        
        self.lnorm = nn.LayerNorm(self.input_size)

        self.ff_layer = nn.Sequential(
            nn.Linear(self.input_size, output_size),
            # nn.ReLU(),
            # nn.Linear(output_size*4, output_size*2),
            # nn.ReLU(),
            # nn.Linear(output_size*2, output_size),
        )

        self.ff_dropout = nn.Dropout(params.drop_prob)

    def _flatten_embedding(self, layer_embedding):
        original_shape = layer_embedding.shape
        new_shape = original_shape[:-2] + (original_shape[-2] * original_shape[-1],)
        flat_layer_embedding = layer_embedding.reshape(new_shape)
        return flat_layer_embedding

    def forward(self, input_batch: Dict[str, torch.Tensor]):
        layer_embeddings = []
        
        for layer in self.agent_feature_embedding_layers:
            layer_embedding = layer(input_batch)
            
            # flatten last two dimensions
            layer_embedding = self._flatten_embedding(layer_embedding)
            layer_embeddings.append(layer_embedding)
            
            # print(layer)
            # print(layer_embedding.shape)

        # [B, A, T, num_enc * H]
        features = torch.cat(layer_embeddings, dim=-1)
        
        temporal_embedding = self.agent_temporal_embedding_layer(input_batch)
        temporal_embedding = temporal_embedding.repeat(
            1, 1, 1, self.num_encoders
        )  # [B, A, T, num_enc * H]

        x = features + temporal_embedding
    
        out = self.ff_layer(self.lnorm(x))
        
        return out