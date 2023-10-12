import torch
import torch.nn as nn
from einops import rearrange
import torch.functional as F
import torch.nn.functional as nF

from torch.distributions.normal import Normal

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, x_delta, x_theta, x_alpha, x_beta, x_gamma, x_upper, **kwargs):

        x_ff, x_delta_ff, x_theta_ff, x_alpha_ff, x_beta_ff, x_gamma_ff, x_upper_ff = \
            self.fn(x, x_delta, x_theta, x_alpha, x_beta, x_gamma, x_upper, **kwargs)

        return x_ff + x, x_delta_ff + x_delta, x_theta_ff + x_theta, x_alpha_ff + x_alpha, \
               x_beta_ff + x_beta, x_gamma_ff + x_gamma, x_upper_ff + x_upper

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, x_delta, x_theta, x_alpha, x_beta, x_gamma, x_upper, **kwargs):
        x, x_delta, x_theta, x_alpha, x_beta, x_gamma, x_upper = \
            self.fn(self.norm(x),
                    self.norm(x_delta), self.norm(x_theta),
                    self.norm(x_alpha), self.norm(x_beta),
                    self.norm(x_gamma), self.norm(x_upper),
                    **kwargs)

        return x, x_delta, x_theta, x_alpha, x_beta, x_gamma, x_upper

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, x_delta, x_theta, x_alpha, x_beta, x_gamma, x_upper):
        return self.net(x), \
               self.net(x_delta), \
               self.net(x_theta),\
               self.net(x_alpha), \
               self.net(x_beta), \
               self.net(x_gamma),\
               self.net(x_upper)

class Residual_FA(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, x_delta, x_theta, x_alpha, x_beta, x_gamma, x_upper, **kwargs):

        x_ff, x_delta_ff, x_theta_ff, x_alpha_ff, x_beta_ff, x_gamma_ff, x_upper_ff, L1, weight = \
            self.fn(x, x_delta, x_theta, x_alpha, x_beta, x_gamma, x_upper, **kwargs)

        return x_ff + x, x_delta_ff + x_delta, x_theta_ff + x_theta, x_alpha_ff + x_alpha, \
               x_beta_ff + x_beta, x_gamma_ff + x_gamma, x_upper_ff + x_upper,  L1, weight

class PreNorm_FA(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, x_delta, x_theta, x_alpha, x_beta, x_gamma, x_upper, **kwargs):
        x, x_delta, x_theta, x_alpha, x_beta, x_gamma, x_upper, L1, weight = \
            self.fn(self.norm(x),
                    self.norm(x_delta), self.norm(x_theta),
                    self.norm(x_alpha), self.norm(x_beta),
                    self.norm(x_gamma), self.norm(x_upper),
                    **kwargs)

        return x, x_delta, x_theta, x_alpha, x_beta, x_gamma, x_upper, L1, weight

class FeedForward_FA(nn.Module):
    def __init__(self, dim, hidden_dim, w_gate, dropout=0., num_experts=6):
        super().__init__()
        self.experts_num = num_experts
        self.unit_net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

        self.experts_net = nn.ModuleList([])
        for _ in range(self.experts_num):
            self.experts_net.append(nn.ModuleList([
                self.unit_net
            ]))
        self.w_gate = w_gate
        self.softmax = nn.Softmax(dim=0)


    def forward(self, x, x_delta, x_theta, x_alpha, x_beta, x_gamma, x_upper):

        delta = self.experts_net[0][0](x_delta)
        theta = self.experts_net[1][0](x_theta)
        alpha = self.experts_net[2][0](x_alpha)
        beta = self.experts_net[3][0](x_beta)
        gamma = self.experts_net[4][0](x_gamma)
        upper = self.experts_net[5][0](x_upper)#32,161,128
        # print(upper.size())

        expert_outputs = torch.stack([self.experts_net[i][0](x) for i in range(self.experts_num)], dim = 3)#32,161,128,6
        # print(expert_outputs.shape)
        # 32,161,128,6
        x = self.softmax(self.w_gate) * expert_outputs# torch.matmul(self.softmax(self.w_gate), expert_outputs)
        # print(x.size())

        x = torch.sum(x,dim=3)
        # print(x.size())
        #32, 161, 128
        L1 = self.w_gate.norm(1)

        return x, delta,theta,alpha, beta, gamma,upper, L1, self.w_gate


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dropout=0.):
        super().__init__()
        self.heads = heads
        self.scale = dim ** -0.5

        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.attn_drop = nn.Dropout(dropout)
        self.to_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, x_delta, x_theta, x_alpha, x_beta, x_gamma, x_upper, mask=None):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        qkv_delta = self.to_qkv(x_delta).chunk(3, dim=-1)
        qkv_theta = self.to_qkv(x_theta).chunk(3, dim=-1)
        qkv_alpha = self.to_qkv(x_alpha).chunk(3, dim=-1)
        qkv_beta = self.to_qkv(x_beta).chunk(3, dim=-1)
        qkv_gamma = self.to_qkv(x_gamma).chunk(3, dim=-1)
        qkv_upper = self.to_qkv(x_upper).chunk(3, dim=-1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        q_delta, k_delta, v_delta = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv_delta)
        q_theta, k_theta, v_theta = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv_theta)
        q_alpha, k_alpha, v_alpha = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv_alpha)
        q_beta, k_beta, v_beta = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv_beta)
        q_gamma, k_gamma, v_gamma = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv_gamma)
        q_upper, k_upper, v_upper = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv_upper)


        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        dots_delta = torch.einsum('bhid,bhjd->bhij', q_delta, k_delta) * self.scale
        dots_theta = torch.einsum('bhid,bhjd->bhij', q_theta, k_theta) * self.scale
        dots_alpha = torch.einsum('bhid,bhjd->bhij', q_alpha, k_alpha) * self.scale
        dots_beta = torch.einsum('bhid,bhjd->bhij', q_beta, k_beta) * self.scale
        dots_gamma = torch.einsum('bhid,bhjd->bhij', q_gamma, k_gamma) * self.scale
        dots_upper = torch.einsum('bhid,bhjd->bhij', q_upper, k_upper) * self.scale

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value=True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, float('-inf'))
            dots_delta.masked_fill_(~mask, float('-inf'))
            dots_theta.masked_fill_(~mask, float('-inf'))
            dots_alpha.masked_fill_(~mask, float('-inf'))
            dots_beta.masked_fill_(~mask, float('-inf'))
            dots_gamma.masked_fill_(~mask, float('-inf'))
            dots_upper.masked_fill_(~mask, float('-inf'))
            del mask

        attn = dots.softmax(dim=-1)
        attn_delta = dots_delta.softmax(dim=-1)
        attn_theta = dots_theta.softmax(dim=-1)
        attn_alpha = dots_alpha.softmax(dim=-1)
        attn_beta = dots_beta.softmax(dim=-1)
        attn_gamma = dots_gamma.softmax(dim=-1)
        attn_upper = dots_upper.softmax(dim=-1)

        attn = self.attn_drop(attn)
        attn_delta = self.attn_drop(attn_delta)
        attn_theta = self.attn_drop(attn_theta)
        attn_alpha = self.attn_drop(attn_alpha)
        attn_beta = self.attn_drop(attn_beta)
        attn_gamma = self.attn_drop(attn_gamma)
        attn_upper = self.attn_drop(attn_upper)


        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out_delta = torch.einsum('bhij,bhjd->bhid', attn_delta, v_delta)
        out_theta = torch.einsum('bhij,bhjd->bhid', attn_theta, v_theta)
        out_alpha = torch.einsum('bhij,bhjd->bhid', attn_alpha, v_alpha)
        out_beta = torch.einsum('bhij,bhjd->bhid', attn_beta, v_beta)
        out_gamma = torch.einsum('bhij,bhjd->bhid', attn_gamma, v_gamma)
        out_upper = torch.einsum('bhij,bhjd->bhid', attn_upper, v_upper)

        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        out_delta = rearrange(out_delta, 'b h n d -> b n (h d)')
        out_delta = self.to_out(out_delta)
        out_theta = rearrange(out_theta, 'b h n d -> b n (h d)')
        out_theta = self.to_out(out_theta)
        out_alpha = rearrange(out_alpha, 'b h n d -> b n (h d)')
        out_alpha = self.to_out(out_alpha)
        out_beta = rearrange(out_beta, 'b h n d -> b n (h d)')
        out_beta = self.to_out(out_beta)
        out_gamma = rearrange(out_gamma, 'b h n d -> b n (h d)')
        out_gamma = self.to_out(out_gamma)
        out_upper = rearrange(out_upper, 'b h n d -> b n (h d)')
        out_upper = self.to_out(out_upper)

        return out, out_delta, out_theta, out_alpha, out_beta, out_gamma, out_upper

class Multi_Layer_Waveout_Transformer_FA(nn.Module):
    def __init__(self, dim, depth, heads, mlp_ratio, dropout,  experts_num=6):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.experts_num = experts_num
        mlp_dim = dim * mlp_ratio
        self.depth = depth
        self.w_gate = nn.Parameter(torch.ones(self.experts_num), requires_grad=True)

        for d in range(depth):
            self.layers.append(nn.ModuleList([
                    Residual(PreNorm(dim, Attention(dim, heads=heads, dropout=dropout))),
                    Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))),
                    Residual(PreNorm(dim, Attention(dim, heads=heads, dropout=dropout))),
                    Residual_FA(PreNorm_FA(dim, FeedForward_FA(dim, mlp_dim,self.w_gate, dropout=dropout)))
                ]))

    def forward(self, x, x_delta, x_theta, x_alpha, x_beta, x_gamma, x_upper, mask=None):
        my_x = []
        my_x_delta = []
        my_x_theta = []
        my_x_alpha = []
        my_x_beta = []
        my_x_gamma = []
        my_x_upper = []
        for d, (attn, ff, attn2, ff2) in enumerate(self.layers):

            x, x_delta, x_theta, x_alpha, x_beta, x_gamma, x_upper = attn(x, x_delta, x_theta, x_alpha, x_beta, x_gamma, x_upper, mask=mask)
            x, x_delta, x_theta, x_alpha, x_beta, x_gamma, x_upper = ff(x, x_delta, x_theta, x_alpha, x_beta, x_gamma, x_upper)
            x, x_delta, x_theta, x_alpha, x_beta, x_gamma, x_upper = attn2(x, x_delta, x_theta, x_alpha, x_beta, x_gamma, x_upper, mask=mask)
            x, x_delta, x_theta, x_alpha, x_beta, x_gamma, x_upper, L1, weight = ff2(x, x_delta, x_theta, x_alpha, x_beta, x_gamma, x_upper)


            my_x.append(x)
            my_x_delta.append(x_delta)
            my_x_theta.append(x_theta)
            my_x_alpha.append(x_alpha)
            my_x_beta.append(x_beta)
            my_x_gamma.append(x_gamma)
            my_x_upper.append(x_upper)

        return my_x, my_x_delta, my_x_theta, my_x_alpha, my_x_beta, my_x_gamma, my_x_upper,\
               x, x_delta, x_theta, x_alpha, x_beta, x_gamma, x_upper, L1, weight