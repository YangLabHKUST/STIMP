import numpy as np
import torch
from torch import nn
from linear_attention_transformer import LinearAttentionTransformer
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


class GCN(nn.Module):
    def __init__(self,
                 c_in, # dimensionality of input features
                 c_out, # dimensionality of output features
                 c_hid,
                 num_types,
                 temp=1, # temperature parameter
                 ):

        super().__init__()

        self.linear = nn.Linear(c_in, c_out, bias=False)
        self.num_types = num_types
        self.temp = temp

        # Initialization
        nn.init.uniform_(self.linear.weight.data, -np.sqrt(6 / (c_in + c_out)), np.sqrt(6 / (c_in + c_out)))
        self.weights_pool = nn.init.xavier_normal_(nn.Parameter(torch.FloatTensor(c_hid, c_in, c_out)))

    def forward(self,
                node_feats, # input node features
                adj_matrix, # adjacency matrix including self-connections
                position_embedding
                ):

        # Apply linear layer and sort nodes by head
        node_feats = torch.matmul(adj_matrix, node_feats)
        position_embedding = torch.matmul(adj_matrix, position_embedding)
        position_weights = torch.einsum('nd, dio-> nio', position_embedding, self.weights_pool)
        node_feats = torch.einsum('bni, nio->bno', node_feats, position_weights)
        return node_feats

class GraphTransformer(nn.Module):
    def __init__(self, config, adj, is_sea, mean, std):
        super(type(self), self).__init__()
        self.config = config 
        self.c_in = 1
        self.c_out= 1
        self.c_hid = config.hidden_dim
        self.out_len = config.out_len
        self.in_len = config.in_len
        self.norm1 = nn.LayerNorm(self.c_hid)
        self.norm2 = nn.LayerNorm(self.c_hid)
        self.gn = nn.GroupNorm(4, self.c_hid)

        self.value_embedding = TokenEmbedding(c_in=self.c_in, d_model=self.c_hid)
        self.position_embedding = PositionEmbedding(config, is_sea=is_sea, mean=mean, std=std)

        self.spatial_encoder = GCN(self.c_hid, self.c_hid, self.c_hid, 3)
        self.temporal_encoder = LinearAttentionTransformer(dim=self.c_hid, depth=1, heads=1, max_seq_len=100, n_local_attn_heads=0, local_attn_window_size=0)

        self.projection1 = nn.Linear(self.in_len, self.out_len)
        self.projection2 = nn.Linear(self.c_hid, self.c_out)
        self.adj = adj

    def forward(self, x_enc):
        B, L, D, N = x_enc.shape
        mean = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - mean
        x_enc = rearrange(x_enc, 'b l d n -> (b n) l d')
        enc_out = self.value_embedding(x_enc)
        position_embedding = self.position_embedding()
        enc_out = rearrange(enc_out, '(b n) l d -> b l d n', b=B, n=N)
        position_embedding = position_embedding.squeeze()

        D = enc_out.shape[-1]
        enc_out = rearrange(enc_out, 'b l d n -> (b n) l d', b=B, n=N)
        enc_out = self.temporal_encoder(enc_out)
        enc_out = rearrange(enc_out, '(b n) l d -> (b l) n d', b=B, n=N)

        enc_out = self.spatial_encoder(enc_out, self.adj, position_embedding.transpose(0,1))
        enc_out = self.gn(enc_out.transpose(1,2))
        enc_out = rearrange(enc_out, '(b l) d n -> (b n) l d', b=B, l=L)

        out = self.projection2(enc_out)
        out = F.silu(out)
        out = self.projection1(out.permute(0,2,1))
        out = rearrange(out, '(b n) d l -> b l d n', b=B, n=N)
        out = out + mean
        return out

