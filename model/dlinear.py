import torch
from torch import nn
import math
import copy
from timm.models.layers import DropPath, trunc_normal_
from linear_attention_transformer import LinearAttentionTransformer
from einops import rearrange
import numpy as np

class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """

    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x

class series_decomp(nn.Module):
    """
    Series decomposition block
    """

    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean
    
class Model(nn.Module):
    """
    Paper link: https://arxiv.org/pdf/2205.13504.pdf
    """

    def __init__(self, configs, individual=False):
        """
        individual: Bool, whether shared model among different variates.
        """
        super(Model, self).__init__()
        self.seq_len = configs['in_len']
        self.pred_len = configs['out_len']
        self.decompsition = series_decomp(5)
        self.channels = 1

        # self.input_projection = nn.Conv1d(1, self.channels, 1, bias=False)
        # nn.init.kaiming_normal_(self.input_projection.weight)
        # self.Linear_Seasonal = LinearAttentionTransformer(dim=self.channels, depth=1, heads=1, max_seq_len=46, n_local_attn_heads=0, local_attn_window_size=0)
        # self.Linear_Trend = LinearAttentionTransformer(dim=self.channels, depth=1, heads=1, max_seq_len=46, n_local_attn_heads=0, local_attn_window_size=0)
        self.Linear_Seasonal = nn.Linear(self.seq_len, self.pred_len)
        self.Linear_Trend = nn.Linear(self.seq_len, self.pred_len)

        self.Linear_Seasonal.weight = nn.Parameter(
            (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))
        self.Linear_Trend.weight = nn.Parameter(
            (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))
        
        self.spatial_encoding = GCN(self.channels, 1, 3)

    def encoder(self, x):
        seasonal_init, trend_init = self.decompsition(x)
        seasonal_init, trend_init = seasonal_init.permute(
            0, 2, 1), trend_init.permute(0, 2, 1)
        
        seasonal_output = self.Linear_Seasonal(seasonal_init)
        trend_output = self.Linear_Trend(trend_init)
        x = seasonal_output + trend_output
        return x.permute(0,2,1)

    def forward(self, x_enc, adj, node_type):
        B, L, D, N = x_enc.shape
        x_enc = rearrange(x_enc, 'B L D N->(B L) D N', B=B, L=L)
        # x_enc = self.input_projection(x_enc)
        # x_enc = nn.functional.gelu(x_enc)

        x_enc = rearrange(x_enc, '(B L) D N->(B N) L D', B=B, L=L)
        x_enc = self.encoder(x_enc)
        x_enc = nn.functional.gelu(x_enc)
        x_enc = rearrange(x_enc, '(B N) L D->(B L) N D', B=B, N=N)
        dec_out = self.spatial_encoding(x_enc, adj, node_type)
        dec_out = rearrange(dec_out, '(B L) N D->B L D N', B=B, L=L)
        return dec_out[:, -self.pred_len:, :, :]  # [B, L, D]


class GCN(nn.Module):
    def __init__(self,
                 c_in, # dimensionality of input features
                 c_out, # dimensionality of output features
                 num_types,
                 temp=1, # temperature parameter
                 ):

        super().__init__()

        self.num_types = num_types
        self.temp = temp

        self.weights_pool = nn.init.xavier_normal_(nn.Parameter(torch.FloatTensor(8, c_in, c_out)))
        self.node_embedding = nn.init.xavier_normal_(nn.Parameter(torch.FloatTensor(3,8)))

    def forward(self,
                node_feats, # input node features
                adj_matrix, # adjacency matrix including self-connections
                node_type,
                ):

        # Apply linear layer and sort nodes by head
        B = node_feats.shape[0]
        node_num = node_type.shape[0]
        node_type = node_type.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 8) # N, Hidden
        node_embedding = self.node_embedding.unsqueeze(0).expand(node_num, -1, -1)
        node_type_embedding = torch.gather(node_embedding, 1, node_type).squeeze()
        node_weights = torch.einsum('nd, dio->nio', node_type_embedding, self.weights_pool)
        node_feats = torch.matmul(adj_matrix, node_feats)
        node_feats = torch.einsum('bni, nio->bno', node_feats, node_weights)
        return node_feats


# class GCN(nn.Module):
#     def __init__(self,
#                  c_in, # dimensionality of input features
#                  c_out, # dimensionality of output features
#                  num_types,
#                  temp=1, # temperature parameter
#                  ):

#         super().__init__()

#         self.linear = nn.Linear(c_in, c_out, bias=False)
#         self.num_types = num_types
#         self.temp = temp

#         # Initialization
#         nn.init.uniform_(self.linear.weight.data, -np.sqrt(6 / (c_in + c_out)), np.sqrt(6 / (c_in + c_out)))

#     def forward(self,
#                 node_feats, # input node features
#                 adj_matrix, # adjacency matrix including self-connections
#                 node_type,
#                 ):

#         # Apply linear layer and sort nodes by head
#         node_feats = torch.matmul(adj_matrix, node_feats)
#         node_feats = self.linear(node_feats)
#         return node_feats
