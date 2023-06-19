import math
import torch
import torch.nn as nn
import torch.utils.checkpoint

from einops import rearrange

class LatentEdgePredictor(nn.Module):
    def __init__(self, input_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        
        # Following section 4.1
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=512),
            nn.Linear(512, 256),         
            nn.ReLU(),
            nn.BatchNorm1d(num_features=256),
            nn.Linear(256, 128),     
            nn.ReLU(),
            nn.BatchNorm1d(num_features=128),
            nn.Linear(128, 64),      
            nn.ReLU(),
            nn.BatchNorm1d(num_features=64),
            nn.Linear(64, output_dim)
        )
        
        # init using kaiming uniform
        for name, module in self.layers.named_modules():
            if module.__class__.__name__ == "Linear":
                nn.init.kaiming_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x, t):
        # x: b, (h w), c
        pos_elem = [torch.sin(2 * math.pi * t * (2 **-l)) for l in range(self.num_layers)]
        pos_encoding = torch.cat(pos_elem, dim=1)
        
        x = torch.cat((x, t, pos_encoding), dim=1)
        # x = rearrange(x, "b c h w -> (b w h) c").to(torch.float16)
        x = rearrange(x, "b c h w -> (b w h) c")
        
        return self.layers(x)