import torch.nn as nn

class ConvHead(nn.Module):
    def __init__(self, params, hidden_size):
        super().__init__()

        self.coord_scale = params.grid_coord_scale
        
        self.conv = nn.Sequential(
            nn.Conv1d(hidden_size, hidden_size, kernel_size=3,stride=1, padding=1),
            nn.GELU(),
            nn.Conv1d(hidden_size, hidden_size//2, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.Conv1d(hidden_size//2, 2, kernel_size=3, stride=1, padding=1),
        )
        
    def forward(self, x):
        b, a, pred_len, h = x.shape
        x_flat = x.reshape(b*a, pred_len, h).transpose(1, 2)  # [b*a, h, pred_len]
        out = self.conv(x_flat).transpose(1, 2)  # [b*a, pred_len, 2]
        return out.reshape(b, a, pred_len, 2) * self.coord_scale
    

class MLPPredictionHead(nn.Module):
    def __init__(self, params, hidden_size):
        super().__init__()
        self.coord_scale = params.grid_coord_scale
        
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size*2),
            nn.GELU(),
            nn.Linear(hidden_size*2, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 2)
        )
        
    def forward(self, x):
        return self.mlp(x) * self.coord_scale


class LstmHead(nn.Module):
    def __init__(self, 
                 params, 
                 hidden_size,
                 num_layers=10):

        super().__init__()
        self.pred_len = params.pred_len
        self.coord_scale=params.grid_coord_scale
        self.lstm = nn.LSTM(input_size=hidden_size,
                           hidden_size=hidden_size,
                           num_layers=num_layers,
                           batch_first=True)
        self.mlp = nn.Linear(hidden_size, 2)  # Output all predictions at once
        self.lnorm=nn.LayerNorm(hidden_size)
        
    def forward(self, input_batch):
        b, a, t, h = input_batch.shape
        x = input_batch.view(b*a, t, h)
        output, (hidden, _) = self.lstm(x)  # Only use final hidden state
        out = self.lnorm(output+x)
        # print(out.shape)
        out = self.mlp(out)  # (b*a, 2*pred_len)

        preds = out.view(b, a, self.pred_len, 2)
        
        return preds*self.coord_scale