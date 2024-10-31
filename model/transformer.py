import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import math
from einops import rearrange, repeat
from torch.nn import Parameter
from einops import rearrange
from math import sqrt

device = "cuda" if torch.cuda.is_available() else "cpu"

class TokenEmbedding(nn.Module): 
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x

class PositionEmbedding(nn.Module):
    def __init__(self, config, is_sea, mean, std):
        super(PositionEmbedding, self).__init__()
        self.config = config
        self.d_model = config.hidden_dim
        self.mean = mean
        self.std = std
        self.is_sea = is_sea
        learnable_position_embedding = self.get_position_embeding()[:,is_sea.bool()]
        self.projection1 = nn.Linear(3, self.d_model)
        self.register_buffer("embedding", learnable_position_embedding)

    def forward(self):
        x = self.embedding.transpose(0,1)
        x = self.projection1(x)
        x = x.transpose(0,1)
        x = x.unsqueeze(0).unsqueeze(0)
        return x

    def get_position_embeding(self):
        height = self.config.height
        width = self.config.width
        mean = torch.zeros(height, width)
        std = torch.zeros(height, width)
        mean[self.is_sea.bool()] = self.mean.cpu()
        std[self.is_sea.bool()] = self.std.cpu()
        pos_w = torch.arange(0., width)/width
        pos_h = torch.arange(0., height)/height
        pos_w = pos_w.unsqueeze(0).expand(height, -1)
        pos_h = pos_h.unsqueeze(1).expand(-1, width)
        pe = torch.stack([mean, std, pos_h], 0)
        pe = pe.to(device)
        return pe

class Transformer(nn.Module):
    def __init__(self, config, adj, is_sea, mean, std):
        super(type(self), self).__init__()
        self.config = config 
        self.c_in = 1
        self.c_out= 1
        self.c_hid = config.hidden_dim
        self.out_len = config.out_len
        self.in_len = config.in_len

        self.value_embedding = TokenEmbedding(c_in=self.c_in, d_model=self.c_hid)
        self.position_embedding = PositionEmbedding(config, is_sea=is_sea, mean=mean, std=std)
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.c_hid, nhead=1, dim_feedforward=self.c_hid, activation="gelu")
        self.temporal_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)

        self.projection1 = nn.Linear(self.in_len, self.out_len)
        self.projection2 = nn.Linear(self.c_hid, self.c_out)
        self.adj = adj

    def forward(self, x_enc, mask=None):
        enc_out = self.value_embedding(x_enc)
        enc_out = rearrange(enc_out, 'b t c -> t b c')
        enc_out = self.temporal_encoder(enc_out, src_key_padding_mask=mask)
        enc_out = rearrange(enc_out, 't b c -> b t c')
        out = self.projection2(enc_out)
        out = F.silu(out)
        out = self.projection1(out.permute(0,2,1))
        return out

