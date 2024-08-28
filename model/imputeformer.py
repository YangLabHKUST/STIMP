import torch
import torch.nn as nn
from einops import repeat,rearrange
import copy
import numpy as np

class ImputeFormer(nn.Module):
    """
    Spatiotemporal Imputation Transformer induced by low-rank factorization, KDD'24.  Note:
        This is a simplified implementation under the SAITS framework (ORT+MIT).
        The timestamp encoding is also removed for ease of implementation.
    """

    def __init__(
        self,
        config,
        dropout: float = 0.0,
        input_dim: int = 1,
        output_dim: int = 1,
    ):
        super().__init__()

        adj = np.load("/home/mafzhang/data/{}/8d/adj.npy".format(config.area))
        self.config = config
        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        self.n_nodes = adj.shape[0]
        self.in_steps = config.in_len
        self.out_steps = config.in_len
        self.input_dim = 1
        self.output_dim = 1
        self.input_embedding_dim = config.hidden_channels
        self.learnable_embedding_dim = config.hidden_channels
        self.model_dim = 2*config.hidden_channels

        self.n_temporal_heads = 1
        self.num_layers = 1
        self.input_proj = nn.Linear(self.input_dim, self.input_embedding_dim)
        self.d_proj = config.hidden_channels
        self.d_ffn = config.hidden_channels

        self.learnable_embedding = nn.init.xavier_uniform_(
            nn.Parameter(
                torch.empty(self.in_steps, self.n_nodes, self.learnable_embedding_dim)
            )
        )

        self.readout = MLP(self.model_dim, self.model_dim, self.output_dim, n_layers=2)

        self.attn_layers_t = nn.ModuleList(
            [
                ProjectedAttentionLayer(
                    self.n_nodes,
                    self.d_proj,
                    self.model_dim,
                    self.n_temporal_heads,
                    self.model_dim,
                    dropout,
                )
                for _ in range(self.num_layers)
            ]
        )

        self.attn_layers_s = nn.ModuleList(
            [
                EmbeddedAttentionLayer(
                    self.model_dim,
                    self.learnable_embedding_dim,
                    self.d_ffn,
                )
                for _ in range(self.num_layers)
            ]
        )

        # apply SAITS loss function to Transformer on the imputation task

    def get_randmask(self, observed_mask, sample_ratio):
        rand_for_mask = torch.rand_like(observed_mask) * observed_mask
        rand_for_mask = rand_for_mask.reshape(len(rand_for_mask), -1)
        for i in range(len(observed_mask)):
            # sample_ratio = 0.2 * np.random.rand()
            sample_ratio = sample_ratio
            num_observed = observed_mask[i].sum().item()
            num_masked = round(num_observed * sample_ratio)
            rand_for_mask[i][rand_for_mask[i].topk(num_masked).indices] = -1
        cond_mask = (rand_for_mask > 0).reshape(observed_mask.shape).float()
        return cond_mask

    def trainstep(self, inputs, ob_masks)->float:
        cond_mask = self.get_randmask(ob_masks, self.config.missing_ratio)
        cond_mask = cond_mask.to(self.device)

        imputed = self.impute(inputs, cond_mask)
        target_mask = ob_masks -cond_mask 
        residual = (inputs - imputed) * target_mask
        num_eval = target_mask.sum()
        loss = (residual ** 2).sum() / (num_eval if num_eval > 0 else 1)
        return loss

    def impute(self, inputs, masks) -> dict:
        x, missing_mask = copy.deepcopy(inputs), masks

        # x: (batch_size, in_steps, num_nodes)
        # Note that ImputeFormer is designed for Spatial-Temporal data that has the format [B, S, N, C],
        # where N is the number of nodes and C is an additional feature dimension,
        # We simply add an extra axis here for implementation.
        x = rearrange(x, 'b t c n -> b t n c')
        missing_mask = rearrange(missing_mask, 'b t c n -> b t n c')
        batch_size = x.shape[0]
        # Whiten missing values
        x = x * missing_mask
        x = self.input_proj(x)  # (batch_size, in_steps, num_nodes, input_embedding_dim)

        # Learnable node embedding
        node_emb = self.learnable_embedding.expand(
            batch_size, *self.learnable_embedding.shape
        )
        x = torch.cat(
            [x, node_emb], dim=-1
        )  # (batch_size, in_steps, num_nodes, model_dim)

        # Spatial and temporal processing with customized attention layers
        x = x.permute(0, 2, 1, 3)  # [b n s c]
        for att_t, att_s in zip(self.attn_layers_t, self.attn_layers_s):
            x = att_t(x)
            x = att_s(x, self.learnable_embedding, dim=1)

        # Readout
        x = x.permute(0, 2, 1, 3)  # [b s n c]
        reconstruction = self.readout(x)
        reconstruction = rearrange(reconstruction, "b t n c -> b t c n")

        # Below is the SAITS processing pipeline:
        # replace the observed part with values from X
        imputed_data = masks * inputs + (1 - masks) * reconstruction
        return imputed_data


class Dense(nn.Module):
    """A simple fully-connected layer."""

    def __init__(self, input_size, output_size, dropout=0.0, bias=True):
        super(Dense, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(input_size, output_size, bias=bias),
            nn.ReLU(),
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
        )

    def forward(self, x):
        return self.layer(x)

class MLP(nn.Module):
    """
    Simple Multi-layer Perceptron encoder with optional linear readout.
    """

    def __init__(
        self, input_size, hidden_size, output_size=None, n_layers=1, dropout=0.0
    ):
        super(MLP, self).__init__()

        layers = [
            Dense(
                input_size=input_size if i == 0 else hidden_size,
                output_size=hidden_size,
                dropout=dropout,
            )
            for i in range(n_layers)
        ]
        self.mlp = nn.Sequential(*layers)

        if output_size is not None:
            self.readout = nn.Linear(hidden_size, output_size)
        else:
            self.register_parameter("readout", None)

    def forward(self, x, u=None):
        """"""
        out = self.mlp(x)
        if self.readout is not None:
            return self.readout(out)
        return out

class AttentionLayer(nn.Module):
    """Perform attention across the -2 dim (the -1 dim is `model_dim`).

    Make sure the tensor is permuted to correct shape before attention.

    E.g.
    - Input shape (batch_size, in_steps, num_nodes, model_dim).
    - Then the attention will be performed across the nodes.

    Also, it supports different src and tgt length.

    But must `src length == K length == V length`.

    """

    def __init__(self, model_dim, num_heads=8, mask=False):
        super().__init__()

        self.model_dim = model_dim
        self.num_heads = num_heads
        self.mask = mask
        self.head_dim = model_dim // num_heads

        self.FC_Q = nn.Linear(model_dim, model_dim)
        self.FC_K = nn.Linear(model_dim, model_dim)
        self.FC_V = nn.Linear(model_dim, model_dim)

        self.out_proj = nn.Linear(model_dim, model_dim)

    def forward(self, query, key, value):
        # Q    (batch_size, ..., tgt_length, model_dim)
        # K, V (batch_size, ..., src_length, model_dim)
        batch_size = query.shape[0]
        tgt_length = query.shape[-2]
        src_length = key.shape[-2]

        query = self.FC_Q(query)
        key = self.FC_K(key)
        value = self.FC_V(value)

        # Qhead, Khead, Vhead (num_heads * batch_size, ..., length, head_dim)
        query = torch.cat(torch.split(query, self.head_dim, dim=-1), dim=0)
        key = torch.cat(torch.split(key, self.head_dim, dim=-1), dim=0)
        value = torch.cat(torch.split(value, self.head_dim, dim=-1), dim=0)

        key = key.transpose(
            -1, -2
        )  # (num_heads * batch_size, ..., head_dim, src_length)

        attn_score = (
            query @ key
        ) / self.head_dim**0.5  # (num_heads * batch_size, ..., tgt_length, src_length)

        if self.mask:
            mask = torch.ones(
                tgt_length, src_length, dtype=torch.bool, device=query.device
            ).tril()  # lower triangular part of the matrix
            attn_score.masked_fill_(~mask, -torch.inf)  # fill in-place

        attn_score = torch.softmax(attn_score, dim=-1)
        out = attn_score @ value  # (num_heads * batch_size, ..., tgt_length, head_dim)
        out = torch.cat(
            torch.split(out, batch_size, dim=0), dim=-1
        )  # (batch_size, ..., tgt_length, head_dim * num_heads = model_dim)

        out = self.out_proj(out)

        return out


class ProjectedAttentionLayer(nn.Module):
    """
    Temporal projected attention layer.
    A low-rank factorization is achieved in the temporal attention matrix.
    """

    def __init__(
        self,
        seq_len,
        dim_proj,
        d_model,
        n_heads,
        d_ff=None,
        dropout=0.1,
    ):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.out_attn = AttentionLayer(d_model, n_heads, mask=None)
        self.in_attn = AttentionLayer(d_model, n_heads, mask=None)
        self.projector = nn.Parameter(torch.randn(seq_len, dim_proj, d_model))
        # self.projector = nn.Parameter(torch.randn(dim_proj, d_model))

        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.MLP = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model)
        )
        self.seq_len = seq_len

    def forward(self, x):
        # x: [b s n d]
        batch = x.shape[0]
        projector = repeat(
            self.projector,
            "seq_len dim_proj d_model -> repeat seq_len dim_proj d_model",
            repeat=batch,
        )  # [b, s, c, d]
        # projector = repeat(self.projector, 'dim_proj d_model -> repeat seq_len dim_proj d_model',
        #                       repeat=batch, seq_len=self.seq_len)  # [b, s, c, d]

        message_out = self.out_attn(
            projector, x, x
        )  # [b, s, c, d] <-> [b s n d] -> [b s c d]
        message_in = self.in_attn(
            x, projector, message_out
        )  # [b s n d] <-> [b, s, c, d] -> [b s n d]
        message = x + self.dropout(message_in)
        message = self.norm1(message)
        message = message + self.dropout(self.MLP(message))
        message = self.norm2(message)

        return message


class EmbeddedAttention(nn.Module):
    """
    Spatial embedded attention layer.
    The node embedding serves as the query and key matrices for attentive aggregation on graphs.
    """

    def __init__(self, model_dim, node_embedding_dim):
        super().__init__()

        self.model_dim = model_dim
        self.FC_Q_K = nn.Linear(node_embedding_dim, model_dim)
        self.FC_V = nn.Linear(model_dim, model_dim)
        self.out_proj = nn.Linear(model_dim, model_dim)

    def forward(self, value, emb):
        # V (batch_size, ..., seq_length, model_dim)
        # emb (..., length, model_dim)
        batch_size = value.shape[0]
        query = self.FC_Q_K(emb)
        key = self.FC_Q_K(emb)
        value = self.FC_V(value)

        # Q, K (..., length, model_dim)
        # V (batch_size, ..., length, model_dim)
        key = key.transpose(-1, -2)  # (..., model_dim, src_length)
        # attn_score = query @ key  # (..., tgt_length, src_length)
        # attn_score = torch.softmax(attn_score, dim=-1)
        # attn_score = repeat(attn_score, 'n s1 s2 -> b n s1 s2', b=batch_size)

        # re-normalization
        query = torch.softmax(query, dim=-1)
        key = torch.softmax(key, dim=-1)
        query = repeat(query, "n s1 s2 -> b n s1 s2", b=batch_size)
        key = repeat(key, "n s2 s1 -> b n s2 s1", b=batch_size)

        # out = attn_score @ value  # (batch_size, ..., tgt_length, model_dim)
        out = key @ value  # (batch_size, ..., tgt_length, model_dim)
        out = query @ out  # (batch_size, ..., tgt_length, model_dim)

        return out


class EmbeddedAttentionLayer(nn.Module):
    def __init__(
        self,
        model_dim,
        node_embedding_dim,
        feed_forward_dim=2048,
        dropout=0,
    ):
        super().__init__()

        self.attn = EmbeddedAttention(model_dim, node_embedding_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(model_dim, feed_forward_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feed_forward_dim, model_dim),
        )
        self.ln1 = nn.LayerNorm(model_dim)
        self.ln2 = nn.LayerNorm(model_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, emb, dim=-2):
        x = x.transpose(dim, -2)
        # x: (batch_size, ..., length, model_dim)
        # emb: (..., length, model_dim)
        residual = x
        out = self.attn(x, emb)  # (batch_size, ..., length, model_dim)
        out = self.dropout1(out)
        out = self.ln1(residual + out)

        residual = out
        out = self.feed_forward(out)  # (batch_size, ..., length, model_dim)
        out = self.dropout2(out)
        out = self.ln2(residual + out)

        out = out.transpose(dim, -2)
        return out