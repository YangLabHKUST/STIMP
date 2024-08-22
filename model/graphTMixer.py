import torch
import torch.nn as nn
import torch.nn.functional as F
from linear_attention_transformer import LinearAttentionTransformer
from einops import rearrange, repeat
from torch.optim import Optimizer
import numpy as np

class GraphTMixer(nn.Module):
    def __init__(self, config, mean, std, max, min):
        super().__init__()
        self.seq_in = config['in_len']
        self.seq_out = config['out_len']
        self.channel_in = 1
        self.hidden_dim = config['hidden_dim']
        padding = 1 if torch.__version__ >= '1.5.0' else 2

        self.input_projection = Conv1d_with_init(1, self.hidden_dim, 1)
        self.time_encoding = ResBlock(config)
        self.spatial_encoding = GCN(self.hidden_dim, self.hidden_dim, 3)
        self.mid_projection = Conv1d_with_init(self.hidden_dim, 2*self.hidden_dim, 1)
        self.out_projection = Conv1d_with_init(self.hidden_dim, self.channel_in, 1)
        self.predictor= Conv1d_with_init(self.seq_in, self.seq_out, 1)
        self.gn = nn.GroupNorm(4, self.hidden_dim)

        self.mean = mean
        self.std = std
        self.max = max
        self.min = min
    def forward(self, x, adj, node_type):
        # x [b, t, c, n]
        B, T, _, N = x.shape
        # mean = torch.mean(x, dim=1, keepdim=True).detach()
        # x = x - mean
        x = rearrange(x, 'b t c n->(b t) c n')
        x = self.input_projection(x)
        C = x.shape[1]
        # x = F.relu(x)

        x_in = x
        x = rearrange(x, '(b t) c n-> (b t) n c', b=B, t=T)
        x = self.spatial_encoding(x, adj, node_type)
        x = rearrange(x, '(b t) n c-> (b t) c n', b=B, t=T)
        x = x + x_in
        x = self.gn(x)

        x = rearrange(x, '(b t) c n -> (b n) t c', b=B, t=T)
        x = self.time_encoding(x)
        # x = rearrange(x, '(b n) t c -> (b n) c t', b=B, n=N)

        # x = self.mid_projection(x)
        # gate, filter = torch.chunk(x, 2, dim=1)
        # y = torch.sigmoid(gate)*torch.tanh(filter)

        # y = rearrange(x, '(b n) c t -> (b n) t c', b=B, n=N)
        # y = self.predictor(y)
        y = rearrange(x, '(b n) t c -> (b n) c t', b=B, n=N)
        y = F.silu(y)
        y = self.out_projection(y)
        y = rearrange(y, '(b n) c t -> b t c n', b=B, n=N)
        # y = y + mean
        return y

    def normalize(self, x):
        mean = self.mean.reshape(1,1,1,-1)
        std = self.std.reshape(1,1,1,-1)
        normalized_x = (x - mean)/(std + 1e-6)
        return normalized_x

    def denormalize(self, x):
        mean = self.mean.reshape(1,1,1,-1)
        std = self.std.reshape(1,1,1,-1)
        denormalized_x = x*std + mean
        return denormalized_x
    
    def normalize_minmax(self, x):
        max = self.max.reshape(1,1,1,-1)
        min = self.min.reshape(1,1,1,-1)
        normalized_x = (2*x-max-min)/(max-min+1e-5)
        return normalized_x
        
    def denormalize_minmax(self, x):
        max = self.max.reshape(1,1,1,-1)
        min = self.min.reshape(1,1,1,-1)
        denormalized_x = (x*(max-min) + max + min)/2
        return denormalized_x
        

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

class GCN(nn.Module):
    def __init__(self,
                 c_in, # dimensionality of input features
                 c_out, # dimensionality of output features
                 num_types,
                 temp=1, # temperature parameter
                 ):

        super().__init__()

        self.linear = nn.Linear(c_in, c_out, bias=False)
        self.num_types = num_types
        self.temp = temp

        # Initialization
        nn.init.uniform_(self.linear.weight.data, -np.sqrt(6 / (c_in + c_out)), np.sqrt(6 / (c_in + c_out)))
        self.weights_pool = nn.init.xavier_normal_(nn.Parameter(torch.FloatTensor(8, c_in, c_out)))
        self.node_embedding = nn.Parameter(torch.FloatTensor(3,8))

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

class ResBlock(nn.Module):
    def __init__(self, configs):
        super(ResBlock, self).__init__()

        self.temporal = nn.Sequential(
            nn.Linear(configs['in_len'], configs['hidden_dim']),
            nn.ReLU(),
            nn.Linear(configs['hidden_dim'], configs['in_len']),
            # nn.Dropout(0.2)
        )

        # self.channel = nn.Sequential(
        #     nn.Linear(configs['hidden_dim'], configs['hidden_dim']),
        #     nn.ReLU(),
        #     nn.Linear(configs['hidden_dim'], configs['hidden_dim']),
        #     # nn.Dropout(0.2)
        # )

    def forward(self, x):
        # x: [B, L, D]
        x = x + self.temporal(x.transpose(1, 2)).transpose(1, 2)
        # x = x + self.channel(x)
        return x

def Conv1d_with_init(in_channels, out_channels, kernel_size):
    layer = nn.Conv1d(in_channels, out_channels, kernel_size)
    nn.init.kaiming_normal_(layer.weight)
    return layer
