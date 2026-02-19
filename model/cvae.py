import torch
import torch.nn as nn
from .head import ConvHead
from .decoder import LearnedPredictionDecoder
from .scene_attn import PatchSceneCrossAttnTransformerLayer, PatchContextAttnTransformerLayer, PatchSelfAttnLayer
from .attention import SelfAttnTransformerLayer, AgentTypeCrossAttnTransformerLayer
from .scene_encoder import SequenceFilterLayer, TransformerOccupancyGridEncoderLayer
from .model_params import ModelParams
from .agent_encoder import FeatureAddAgentEncoderLayer
from .preprocess import GridPreprocessLayer

class TransformerCVAE(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.hidden_size = 256
        self.latent_size = self.hidden_size // 8

        self.encoder = VAEEncoder(params, hidden_size=self.hidden_size)

        self.mu = nn.Sequential(
            nn.Linear(self.hidden_size, self.latent_size * 2),
            nn.ReLU(),
            nn.Linear(self.latent_size * 2, self.latent_size),
        )
        self.logvar = nn.Sequential(
            nn.Linear(self.hidden_size, self.latent_size * 2),
            nn.ReLU(),
            nn.Linear(self.latent_size * 2, self.latent_size),
        )

        self.decoder = VAEDecoder(params=params, hidden_size=self.hidden_size, latent_size=self.latent_size)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def get_mu_logvar(self, enc_out):
        mu = self.mu(enc_out)
        logvar = self.logvar(enc_out)
        return mu, logvar

    def forward(self, input_batch, future_batch):
        """
        Args:
            input_batch: past trajectory data
            future_batch: future trajectory data
        """

        enc_out, past_cond, scene_cond, station_cond = self.encoder(
            input_batch,
            future_batch
        )

        mu, logvar = self.get_mu_logvar(enc_out)

        z = self.reparameterize(mu, logvar)
        self.latent=z

        out = self.decoder(z, past_cond, scene_cond, station_cond)

        return out, mu, logvar

    def sample(self, input_batch, temperature=0.1):

        _, past_cond, scene_cond, station_cond = self.encoder(
            input_batch,
            future_batch=None
        )
        
        b, a, t, _= past_cond.shape
        device = past_cond.device
        
        # Sample z from prior N(0, I)
        z = torch.randn(b, a, t, self.latent_size, device=device) * temperature
        
        # Decode to get trajectory prediction
        out = self.decoder(z, past_cond, scene_cond, station_cond)
        
        return out
    
    def get_grid(self, input_batch):
        
        occ_grid, coord_grid = self.encoder.scene_condition.get_grid(input_batch)

        return occ_grid, coord_grid

class VAEEncoder(nn.Module):
    def __init__(self, params: ModelParams, hidden_size=256):
        super().__init__()

        self.hidden_size = hidden_size

        self.future_encoder=TrajectoryEncoder(params, hidden_size=self.hidden_size, future=True)
        self.past_encoder=TrajectoryEncoder(params, hidden_size=self.hidden_size, future=False)
        
        self.human_agent_encoding_layer = FeatureAddAgentEncoderLayer(
            agent_type='human',
            hidden_size=128,
            output_size=self.hidden_size,
            params=params
        )
        self.sequence_filter = SequenceFilterLayer(
            params=params,
            hidden_size=self.hidden_size,
            kernel_size=3,
            stride=1,
            padding='same',
            pool=False
        )
        self.sequence_filter2 = SequenceFilterLayer(
            params=params,
            hidden_size=self.hidden_size,
            kernel_size=3,
            stride=1,
            padding='same',
            pool=False
        )
        
        self.scene_condition = SceneCondition(params=params, hidden_size=self.hidden_size)
        
        self.station_agent_encoding_layer = FeatureAddAgentEncoderLayer(
                    agent_type='stations',
                    hidden_size=128,
                    output_size=self.hidden_size,
                    params=params
                )

        self.fusion_layer = nn.Sequential(
            nn.Linear(self.hidden_size*2, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
        )

    def forward(self, input_batch, future_batch=None):
        
        station_cond=self.station_agent_encoding_layer(input_batch).mean(dim=2).unsqueeze(-2)
        scene_cond = self.scene_condition(input_batch, station_cond)
        
        past_cond, scene_cond, station_cond = self.past_encoder(input_batch, scene_cond, station_cond)
        
        enc_out = past_cond
        
        if future_batch is not None:
            
            # Encode future trajectory
            # future_enc = self.future_encoder(future_batch, scene_cond)
            future_enc = self.human_agent_encoding_layer(future_batch)
            future_enc = self.sequence_filter(future_enc)
            future_enc = self.sequence_filter2(future_enc)

            # future_enc = self.sequence_filter(future_enc)

            # enc_out = future_enc + past_cond
            # Concatenate past and future encodings
            combined = torch.cat([past_cond, future_enc], dim=-1)  # (b, a, t, 2*h)
            # Fuse them
            enc_out = self.fusion_layer(combined)  # (b, a, t, h)

        return enc_out, past_cond, scene_cond, station_cond

class TrajectoryEncoder(nn.Module):
    def __init__(self, params: ModelParams, hidden_size=256, future=False):
        super().__init__()

        self.hidden_size = hidden_size
        self.future = future

        # Past trajectory encoding
        self.human_agent_encoding_layer = FeatureAddAgentEncoderLayer(
            agent_type='human',
            hidden_size=128,
            output_size=self.hidden_size,
            params=params
        )

        self.agent_type_cross_attention_layer = AgentTypeCrossAttnTransformerLayer(
            params=params, 
            hidden_size=self.hidden_size
        )

        self.human_self_attn_layer = SelfAttnTransformerLayer(
            params,
            hidden_size=self.hidden_size
        )

        self.sequence_filter = SequenceFilterLayer(
            params=params,
            hidden_size=self.hidden_size,
            kernel_size=3,
            stride=1,
            padding='same',
            pool=False
        )
        self.sequence_filter1 = SequenceFilterLayer(
            params=params,
            hidden_size=self.hidden_size,
            kernel_size=3,
            stride=1,
            padding='same',
            pool=False
        )

        self.sequence_down_filter = SequenceFilterLayer(
            params=params,
            hidden_size=self.hidden_size,
            kernel_size=3,
            stride=1,
            padding='same',
            pool=True
        )

        self.cross_attention_layer = AgentTypeCrossAttnTransformerLayer(
            params=params,
            hidden_size=self.hidden_size
        )

        self.scene_cross_attn_layer = PatchSceneCrossAttnTransformerLayer(
            params,
            hidden_size=self.hidden_size,
        )
        self.scene_cross_attn_layer1 = PatchSceneCrossAttnTransformerLayer(
            params,
            hidden_size=self.hidden_size,
        )
        self.scene_context_down_layer = PatchContextAttnTransformerLayer(
            params=params,
            hidden_size=self.hidden_size,
            downsample_factor=2,
        )
        self.scene_context_layer = PatchContextAttnTransformerLayer(
            params=params,
            hidden_size=self.hidden_size,
        )

        self.learned_token = nn.Parameter(
            torch.empty(1, 1, 1, hidden_size).uniform_(-1.0, 1.0)
        )

        self.patch_self_attn_layer = PatchSelfAttnLayer(
            params=params,
            hidden_size=self.hidden_size,
            expansion_factor=2
        )

    def forward(self, input_batch, scene_cond, station_cond):
    
        # Encode past trajectory
        human_emb = self.human_agent_encoding_layer(input_batch)
        b, a, _, h=human_emb.shape
        
        human_emb = self.scene_cross_attn_layer(human_emb, scene_cond)
        human_emb = self.sequence_filter(human_emb)
        scene_enc = self.scene_context_layer(human_emb, scene_cond)

        human_emb = self.scene_cross_attn_layer(human_emb, scene_enc)
        human_emb = self.sequence_filter(human_emb)
        scene_enc = self.scene_context_layer(human_emb, scene_enc)

        human_emb = self.scene_cross_attn_layer(human_emb, scene_enc)
        human_emb = self.sequence_filter(human_emb)
        scene_enc = self.scene_context_layer(human_emb, scene_enc)

        station_cond = self.cross_attention_layer(station_cond, human_emb)
        scene_enc = self.scene_context_layer(station_cond, scene_enc)

        # if self.future:
        #     return human_emb
        # else:

        return human_emb, scene_enc, station_cond

class SceneCondition(nn.Module):    
    def __init__(self, params: ModelParams, hidden_size=256):
        super().__init__()

        self.pred_len = params.pred_len
        self.hidden_size = hidden_size

        self.scene_preprocess_layer = GridPreprocessLayer(params, scale_coords=False)
        self.scene_encoder = TransformerOccupancyGridEncoderLayer(
            params,
            prefilter=False,
            hidden_size=self.hidden_size,
            position_encoding='origin'
        )

        self.scene_cross_attn_layer = PatchSceneCrossAttnTransformerLayer(
            params,
            hidden_size=self.hidden_size,
        )

        self.scene_context_down_layer = PatchContextAttnTransformerLayer(
            params=params,
            hidden_size=self.hidden_size,
            downsample_factor=2,
        )
        self.scene_context_layer = PatchContextAttnTransformerLayer(
            params=params,
            hidden_size=self.hidden_size,
        )

        self.patch_self_attn_layer = PatchSelfAttnLayer(
            params=params,
            hidden_size=self.hidden_size,
            expansion_factor=2
        )

        self.learned_token = nn.Parameter(
            torch.empty(1, 1, 1, hidden_size).uniform_(-1.0, 1.0)
        )

    def forward(self, input_batch, station_cond):
        """
        Args:
            input_batch: dict containing past trajectory data
            future_batch: dict containing future trajectory data (None during inference)
        """
        human_emb = input_batch['human_pos'] 
        
        b, a, _, _ = human_emb.shape
        
        # Scene encoding
        occ_grid, coord_grid = self.scene_preprocess_layer(input_batch)
        
        scene_enc = self.scene_encoder(input_batch, occ_grid, coord_grid)
        scene_enc = self.scene_context_layer(station_cond, scene_enc)
        scene_enc = self.patch_self_attn_layer(scene_enc)

        scene_enc=scene_enc.reshape(b,a,-1, self.hidden_size)

        scene_cond = scene_enc

        return scene_cond
    
    def get_grid(self, input_batch):
        
        occ_grid, coord_grid =  self.scene_preprocess_layer(input_batch)
        
        return occ_grid, coord_grid

class VAEDecoder(nn.Module):
    def __init__(self, params: ModelParams, hidden_size, latent_size):
        super().__init__()

        self.hidden_size = hidden_size
        self.latent_size = latent_size

        self.expand_latent = nn.Sequential(
            nn.Linear(self.latent_size, self.latent_size * 2),
            nn.ReLU(),
            nn.Linear(self.latent_size * 2, self.hidden_size),
        )

        self.trajectory_generator = LearnedPredictionDecoder(params, hidden_size=self.hidden_size)
        self.human_self_attn_layer = SelfAttnTransformerLayer(
            params,
            hidden_size=self.hidden_size
        )

        self.scene_attention_layer = PatchSceneCrossAttnTransformerLayer(
            params=params,
            hidden_size=self.hidden_size,
        )
        self.scene_attention_layer1 = PatchSceneCrossAttnTransformerLayer(
            params=params,
            hidden_size=self.hidden_size,
        )
        
        self.scene_context_layer = PatchContextAttnTransformerLayer(
            params=params,
            hidden_size=self.hidden_size,
        )
        self.scene_context_down_layer = PatchContextAttnTransformerLayer(
            params=params,
            hidden_size=self.hidden_size,
            downsample_factor=2
        )

        self.cross_attention_layer = AgentTypeCrossAttnTransformerLayer(
            params=params,
            hidden_size=self.hidden_size
        )

        self.sequence_filter = SequenceFilterLayer(
            params=params,
            hidden_size=self.hidden_size,
            kernel_size=3,
            stride=1,
            padding='same',
            pool=False
        )
        self.sequence_filter1 = SequenceFilterLayer(
            params=params,
            hidden_size=self.hidden_size,
            kernel_size=3,
            stride=1,
            padding='same',
            pool=False
        )

        self.sequence_filter_down = SequenceFilterLayer(
            params=params,
            hidden_size=self.hidden_size,
            kernel_size=3,
            stride=1,
            output_size=self.hidden_size//2,
            padding='same',
            pool=False
        )
        self.head = ConvHead(params=params, hidden_size=self.hidden_size)
        
        self.fusion_layer = nn.Sequential(
            nn.Linear(self.hidden_size+self.latent_size, (self.hidden_size+self.latent_size)//2),
            nn.ReLU(),
            nn.Linear((self.hidden_size+self.latent_size)//2, self.hidden_size),
        )

        self.ff_layer = nn.Sequential(
            nn.Linear(self.hidden_size*8, self.hidden_size*2),
            nn.ReLU(),
            nn.Linear(self.hidden_size*2, self.hidden_size),
        )

    def forward(self, latent, past_cond, scene_cond, station_cond):
        
        # past_cond = self.down_layer(past_cond)
        traj = torch.cat([past_cond, latent], dim=-1)  # (b, a, 1, h+latent_size)
        traj = self.fusion_layer(traj)
        
        b, a, t, h = traj.shape
        # traj = traj.reshape(b, a, 1, t * h)
        # traj = self.ff_layer(traj)
        traj = traj.mean(dim=2).unsqueeze(-2)

        traj = self.scene_attention_layer(traj, scene_cond)
        traj = self.cross_attention_layer(traj, station_cond)

        out = self.trajectory_generator(traj)
        out = self.sequence_filter(out)
        out = self.scene_attention_layer1(out, scene_cond)
        
        scene_cond = self.scene_context_layer(out, scene_cond)
        out = self.scene_attention_layer1(out, scene_cond)
        # out = self.sequence_filter(out)
        # out = self.scene_attention_layer1(out, scene_cond)


        out = self.head(out)

        return out