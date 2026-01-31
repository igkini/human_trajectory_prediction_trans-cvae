from . import embed
# from model import scene_encoders

class ModelParams(object):
  """This object configures the model."""

  def __init__(
      self,
      human_agents_feature_config=None,
      prediction_agents_feature_config=None,
      robot_agents_feature_config=None,
      stations_agents_feature_config= None,
      agents_position_key='human_pos',
      hidden_size=128,
      latent_dim=32,

      feature_embedding_size=128,
      sin_embedding_size=64,
      transformer_ff_dim=64,
      ff_dim=64,
      ff_expansion_factor=1/2,
      kpt_hidden=64,

      ln_eps=1e-6,
      num_heads=8,
      num_conv_filters=(32,16,8,3),
      patch_size=16,

      depth=13,
      scene_encoder=None,
      pred_len=8,
      seq_len=8,

      drop_prob=0.1,
      grid_output_size=(256,256),
      grid_coord_scale=9,

      # grid_output_size=(320,320),
      # grid_coord_scale=12,
      # grid_output_size=(640,640),
      # grid_coord_scale=21.5,
      # grid_output_size=(400,400),
      # grid_coord_scale=15
  ):

    if human_agents_feature_config is None:
      self.human_agents_feature_config = {
          'human_pos': embed.AgentPositionEncoder,
          # 'human_vel': embed.AgentScalarEncoder,
          # 'task_id': embed.AgentOneHotEncoder,
          # 'all_agents_pos': embed.StationPositionEncoder,
          # 'human_kpts': embed.AgentKeypointsEncoder
          }
    
    else:
      self.human_agents_feature_config = human_agents_feature_config

    if robot_agents_feature_config is None:
      self.robot_agents_feature_config = {
          'robot_pos': embed.AgentPositionEncoder,
          }
    else:
      self.robot_agents_feature_config = robot_agents_feature_config

    if stations_agents_feature_config is None:
      self.stations_agents_feature_config = {
          'stations_pos': embed.AgentPositionEncoder,
          }
    else:
      self.stations_agents_feature_config = stations_agents_feature_config

    if prediction_agents_feature_config is None:
      self.prediction_agents_feature_config = {
          'prediction_pos': embed.AgentPositionEncoder,
          # 'human_kpts': embed.AgentKeypointsEncoder
          }
    
    else:
      self.prediction_agents_feature_config = prediction_agents_feature_config



    self.kpt_hidden=kpt_hidden
    self.agents_position_key = agents_position_key
    self.hidden_size = hidden_size
    self.latent_dim=latent_dim
    self.ff_dim= ff_dim
    self.feature_embedding_size = feature_embedding_size
    self.sin_embedding_size = sin_embedding_size

    self.transformer_ff_dim = transformer_ff_dim
    self.ff_expansion_factor=ff_expansion_factor

    self.ln_eps = ln_eps
    self.num_heads = num_heads
    self.num_conv_filters = num_conv_filters
    self.patch_size=patch_size

    self.scene_encoder = scene_encoder
    self.drop_prob = drop_prob
    self.depth = depth
    self.pred_len=pred_len
    self.seq_len=seq_len

    self.grid_output_size=grid_output_size
    self.grid_coord_scale=grid_coord_scale