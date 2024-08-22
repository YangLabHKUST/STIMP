import numpy as np
import torch
from torch import nn
import timm
import torch.nn.functional as F
from einops import repeat, rearrange
from einops.layers.torch import Rearrange

class NCF(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        self.x_embedding = nn.Embedding(config.height, 64)
        self.y_embedding = nn.Embedding(config.width, 64)

        self.projection = nn.Sequential(
                nn.Linear(64, 32),
                nn.SiLU(),
                nn.Linear(32,1)
                )

    def get_randmask(self, observed_mask):
        rand_for_mask = torch.rand_like(observed_mask) * observed_mask
        rand_for_mask = rand_for_mask.reshape(len(rand_for_mask), -1)
        for i in range(len(observed_mask)):
            # sample_ratio = 0.2 * np.random.rand()
            sample_ratio = 0.1 
            num_observed = observed_mask[i].sum().item()
            num_masked = round(num_observed * sample_ratio)
            rand_for_mask[i][rand_for_mask[i].topk(num_masked).indices] = -1
        cond_mask = (rand_for_mask > 0).reshape(observed_mask.shape).float()
        return cond_mask

    def impute(self, x, cond_mask):

        bs, t , _ , _, _ = x.shape
        x_embedding = self.x_embedding.weight.unsqueeze(1).expand(-1, self.config.width, -1)
        y_embedding = self.y_embedding.weight.unsqueeze(0).expand(self.config.height, -1, -1)

        inter = x_embedding*y_embedding
        prediction = self.projection(inter.reshape(self.config.height*self.config.width, -1))
        prediction = repeat(prediction, "(h w) c->b t h w c", h=self.config.height, w=self.config.width, b=bs, t=t)
        prediction = rearrange(prediction, "b t h w c->b t c h w")
        return prediction

    def trainstep(self, observed_data, observed_mask, observed_y, observed_y_mask, is_train, set_t=-1):

        cond_mask = self.get_randmask(observed_mask)
        cond_mask = cond_mask.to(self.device)

        predicted = self.impute(observed_data, cond_mask)

        target_mask = observed_mask - cond_mask
        residual = (observed_data - predicted) * target_mask
        num_eval = target_mask.sum()
        loss = (residual ** 2).sum() / (num_eval if num_eval > 0 else 1)

        return loss

