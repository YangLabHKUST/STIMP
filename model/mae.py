import numpy as np
import torch
from torch import nn
import timm
import torch.nn.functional as F

from einops import repeat, rearrange
from einops.layers.torch import Rearrange

from timm.models.layers import trunc_normal_
from timm.models.vision_transformer import Block

class MaskedAutoEncoder(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.patch_size = 4
        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        self.cls_token = torch.nn.Parameter(torch.zeros(1, 1, self.config.embedding_size))
        self.input_projection = Conv1d_with_init(1, config.hidden_channels, 1)
        self.pos_embedding = torch.nn.Parameter(torch.zeros((self.config.height//self.patch_size)*(self.config.width//self.patch_size), 1, self.config.embedding_size))
        self.de_pos_embedding = torch.nn.Parameter(torch.zeros((self.config.height//self.patch_size)*(self.config.width//self.patch_size)+1, 1, self.config.embedding_size))
        self.masked_embedding = nn.Embedding(1, config.hidden_channels)
        self.patchify = torch.nn.Conv2d(self.config.hidden_channels, self.config.embedding_size, self.patch_size, self.patch_size)
        self.en_transformer = Block(self.config.embedding_size, self.config.num_heads)
        self.de_transformer = Block(self.config.embedding_size, self.config.num_heads)
        self.layer_norm = torch.nn.LayerNorm(self.config.embedding_size)
        self.head = torch.nn.Linear(self.config.embedding_size, self.patch_size**2)
        self.patch2img = Rearrange('(h w) (b t) (c p1 p2) -> b t c (h p1) (w p2)', t=self.config.in_len, p1=self.patch_size, p2=self.patch_size, h=self.config.height//self.patch_size, w=self.config.width//self.patch_size)
        self.init_weight()

    def init_weight(self):
        trunc_normal_(self.pos_embedding, std=.02) 
    
    def get_randmask(self, observed_mask):
        rand_for_mask = torch.rand_like(observed_mask) * observed_mask
        rand_for_mask = rand_for_mask.reshape(len(rand_for_mask), -1)
        for i in range(len(observed_mask)):
            # sample_ratio = 0.2 * np.random.rand()
            sample_ratio = self.config.missing_ratio
            num_observed = observed_mask[i].sum().item()
            num_masked = round(num_observed * sample_ratio)
            rand_for_mask[i][rand_for_mask[i].topk(num_masked).indices] = -1
        cond_mask = (rand_for_mask > 0).reshape(observed_mask.shape).float()
        return cond_mask
    
    def impute(self, x, cond_mask):
        x = x.unsqueeze(3)
        B, T, K, C, H, W = x.shape
        x = x.reshape(B*T*K, C, H*W)
        x = self.input_projection(x)
        x = x.reshape(B, T, K, self.config.hidden_channels, H, W)

        #fill masked feature
        masked_feature = self.masked_embedding.weight.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        masked_feature = masked_feature.expand(B, T, -1, self.config.hidden_channels, H, W)
        cond_mask = cond_mask.unsqueeze(3).expand(-1, -1, -1, self.config.hidden_channels, -1, -1)
        input = torch.where(cond_mask.bool(), x, masked_feature)
        input = input.reshape(B*T, K*self.config.hidden_channels, H, W)

        patches = self.patchify(input) # B 64 h w
        patches = rearrange(patches, 'b c h w -> (h w) b c')
        patches = patches + self.pos_embedding

        patches = torch.cat([self.cls_token.expand(-1, patches.shape[1], -1), patches], dim=0)
        patches = rearrange(patches, 't b c -> b t c')
        features = self.layer_norm(self.en_transformer(patches))

        #Decode
        features = rearrange(features, 'b t c -> t b c')
        features = features + self.de_pos_embedding
        features = rearrange(features, 't b c -> b t c')
        features = self.de_transformer(features)
        features = rearrange(features, 'b t c -> t b c')
        features = features[1:]
        patches = self.head(features)
        predicted = self.patch2img(patches)
        return predicted

    def trainstep(self, observed_data, observed_mask, observed_y, observed_y_mask, is_train, set_t=-1):

        cond_mask = self.get_randmask(observed_mask)
        cond_mask = cond_mask.to(self.device)

        predicted = self.impute(observed_data, cond_mask)

        target_mask = observed_mask - cond_mask
        residual = (observed_data - predicted) * target_mask
        num_eval = target_mask.sum()
        loss = (residual ** 2).sum() / (num_eval if num_eval > 0 else 1)

        return loss

def Conv1d_with_init(in_channels, out_channels, kernel_size):
    layer = nn.Conv1d(in_channels, out_channels, kernel_size)
    nn.init.kaiming_normal_(layer.weight)
    return layer
