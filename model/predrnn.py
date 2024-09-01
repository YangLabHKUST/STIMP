import torch
import torch.nn as nn
from einops import rearrange


class PredRNN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        config.patch_size = 4
        self.frame_channel = config.patch_size**2
        height = config.height//config.patch_size
        width = config.width//config.patch_size

        self.cell_list = nn.ModuleList()
        self.n_layers = 2
        for i in range(self.n_layers):
            in_channel = self.frame_channel if i==0 else config.hidden_dim
            self.cell_list.append(SpatioTemporalLSTMCell(in_channel, self.config.hidden_dim, height, width, 3, 1, True))
        
        self.conv_last = nn.Conv2d(config.hidden_dim, self.frame_channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    
    def forward(self, x):
        x = reshape_patch(x, self.config.patch_size)
        B, T, C, H, W = x.shape
        next_frames = []
        h_t = []
        c_t = []
        for i in range(self.n_layers):
            zeros = torch.zeros([B, self.config.hidden_dim, H, W]).to(self.device)
            h_t.append(zeros)
            c_t.append(zeros)
        
        memory = torch.zeros([B, self.config.hidden_dim, H, W]).to(self.device)
        for t in range(self.config.in_len + self.config.out_len - 1):
            net = x[:, t] if t < self.config.in_len else next_frames[t-self.config.in_len]
            h_t[0], c_t[0], memory = self.cell_list[0](net, h_t[0], c_t[0], memory)

            for i in range(1, self.n_layers):
                h_t[i], c_t[i], memory = self.cell_list[i](h_t[i-1], h_t[i], c_t[i], memory)

            x_gen = self.conv_last(h_t[-1])
            if t >= self.config.in_len-1:
                next_frames.append(x_gen)
        
        next_frames = torch.stack(next_frames, 1)
        next_frames = reshape_patch_back(next_frames, self.config.patch_size)
        return next_frames
            

class SpatioTemporalLSTMCell(nn.Module):
    def __init__(self, in_channel, num_hidden, height, width, filter_size, stride, layer_norm):
        super(SpatioTemporalLSTMCell, self).__init__()

        self.num_hidden = num_hidden
        self.padding = filter_size // 2
        self._forget_bias = 1.0
        if layer_norm:
            self.conv_x = nn.Sequential(
                nn.Conv2d(in_channel, num_hidden * 7, kernel_size=filter_size,
                          stride=stride, padding=self.padding, bias=False),
                nn.LayerNorm([num_hidden * 7, height, width])
            )
            self.conv_h = nn.Sequential(
                nn.Conv2d(num_hidden, num_hidden * 4, kernel_size=filter_size,
                          stride=stride, padding=self.padding, bias=False),
                nn.LayerNorm([num_hidden * 4, height, width])
            )
            self.conv_m = nn.Sequential(
                nn.Conv2d(num_hidden, num_hidden * 3, kernel_size=filter_size,
                          stride=stride, padding=self.padding, bias=False),
                nn.LayerNorm([num_hidden * 3, height, width])
            )
            self.conv_o = nn.Sequential(
                nn.Conv2d(num_hidden * 2, num_hidden, kernel_size=filter_size,
                          stride=stride, padding=self.padding, bias=False),
                nn.LayerNorm([num_hidden, height, width])
            )
        else:
            self.conv_x = nn.Sequential(
                nn.Conv2d(in_channel, num_hidden * 7, kernel_size=filter_size,
                          stride=stride, padding=self.padding, bias=False),
            )
            self.conv_h = nn.Sequential(
                nn.Conv2d(num_hidden, num_hidden * 4, kernel_size=filter_size,
                          stride=stride, padding=self.padding, bias=False),
            )
            self.conv_m = nn.Sequential(
                nn.Conv2d(num_hidden, num_hidden * 3, kernel_size=filter_size,
                          stride=stride, padding=self.padding, bias=False),
            )
            self.conv_o = nn.Sequential(
                nn.Conv2d(num_hidden * 2, num_hidden, kernel_size=filter_size,
                          stride=stride, padding=self.padding, bias=False),
            )
        self.conv_last = nn.Conv2d(num_hidden * 2, num_hidden, kernel_size=1,
                                   stride=1, padding=0, bias=False)

    def forward(self, x_t, h_t, c_t, m_t):
        x_concat = self.conv_x(x_t)
        h_concat = self.conv_h(h_t)
        m_concat = self.conv_m(m_t)
        i_x, f_x, g_x, i_x_prime, f_x_prime, g_x_prime, o_x = \
            torch.split(x_concat, self.num_hidden, dim=1)
        i_h, f_h, g_h, o_h = torch.split(h_concat, self.num_hidden, dim=1)
        i_m, f_m, g_m = torch.split(m_concat, self.num_hidden, dim=1)

        i_t = torch.sigmoid(i_x + i_h)
        f_t = torch.sigmoid(f_x + f_h + self._forget_bias)
        g_t = torch.tanh(g_x + g_h)

        c_new = f_t * c_t + i_t * g_t

        i_t_prime = torch.sigmoid(i_x_prime + i_m)
        f_t_prime = torch.sigmoid(f_x_prime + f_m + self._forget_bias)
        g_t_prime = torch.tanh(g_x_prime + g_m)

        m_new = f_t_prime * m_t + i_t_prime * g_t_prime

        mem = torch.cat((c_new, m_new), 1)
        o_t = torch.sigmoid(o_x + o_h + self.conv_o(mem))
        h_new = o_t * torch.tanh(self.conv_last(mem))

        return h_new, c_new, m_new

def reshape_patch(img_tensor, patch_size):
    assert 5 == img_tensor.ndim
    batch_size, seq_length, num_channels, img_height, img_width = img_tensor.shape
    patch_tensor = img_tensor.reshape(batch_size, seq_length, num_channels, 
                                img_height//patch_size, patch_size,
                                img_width//patch_size, patch_size)
    patch_tensor = rearrange(patch_tensor, 'b t c h p1 w p2 -> b t (c p1 p2) h w')
    return patch_tensor

def reshape_patch_back(patch_tensor, patch_size):
    batch_size, seq_length, channels, patch_height, patch_width = patch_tensor.shape
    img_channels = channels // (patch_size*patch_size)
    img_tensor = rearrange(patch_tensor, "b t (c p1 p2) h w->b t c (h p1) (w p2)", c=img_channels, p1=patch_size, p2=patch_size)
    return img_tensor