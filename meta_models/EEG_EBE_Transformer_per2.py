from random import random

import torch
import torch.nn as nn

import my_config
from base_models.MultiOut_transformer_EBE_per2 import Multi_Layer_Waveout_Transformer_FA

from utils.utils import get_1d_sincos_pos_embed
from einops.layers.torch import Rearrange

def uptri_random_signal_decomposed(data_eeg, delta_eeg, theta_eeg, alpha_eeg, beta_eeg, gamma_eeg, upper_eeg):
    batch = data_eeg.shape[0]
    n = 6
    matrix0 = torch.tensor([[1 if x <= y else 0 for x in range(n)] for y in range(n)]).cuda(my_config.Config.GPU_id) #左下1
    matrix1 = torch.tensor([[1 if x > y else 0 for x in range(n)] for y in range(n)]).cuda(my_config.Config.GPU_id) #左下0

    random_column =  torch.randint(n-1, [batch,]).cuda(my_config.Config.GPU_id)

    weight0 = torch.reshape(matrix0[random_column],[1, -1]).cuda(my_config.Config.GPU_id)#32,6
    weight1 = torch.reshape(matrix1[random_column],[1, -1]).cuda(my_config.Config.GPU_id)#32,6
    # print(data_eeg.shape)
    # print(delta_eeg.shape)
    # print(weight1.shape)
    # print(weight0.shape)

    data_matrix = torch.stack((delta_eeg, theta_eeg, alpha_eeg, beta_eeg, gamma_eeg, upper_eeg), dim=1).cuda(my_config.Config.GPU_id)
    # print(data_matrix.shape)
    data_matrix_puppet = torch.reshape(data_matrix, ([data_matrix.shape[2],data_matrix.shape[3],data_matrix.shape[4], -1])).cuda(my_config.Config.GPU_id)

    lower_matrix = torch.mul(data_matrix_puppet, weight0).cuda(my_config.Config.GPU_id)
    lower_matrix = torch.reshape(lower_matrix, ([batch, 6, data_matrix.shape[2],data_matrix.shape[3],data_matrix.shape[4]])).cuda(my_config.Config.GPU_id)
    lower = torch.sum(lower_matrix, dim = 1).cuda(my_config.Config.GPU_id)

    higher_matrix = torch.mul(data_matrix_puppet, weight1).cuda(my_config.Config.GPU_id)
    higher_matrix = torch.reshape(higher_matrix, ([batch, 6, data_matrix.shape[2],data_matrix.shape[3],data_matrix.shape[4]])).cuda(my_config.Config.GPU_id)
    higher = torch.sum(higher_matrix, dim = 1).cuda(my_config.Config.GPU_id)
    # print((data_eeg - lower - higher).sum())
    return data_eeg, lower, higher
def uptri_random_signal_decomposed_combine(part_num , data_eeg, delta_eeg, theta_eeg, alpha_eeg, beta_eeg, gamma_eeg, upper_eeg):
    part_length = int(data_eeg.shape[3] / part_num)
    batch = data_eeg.shape[0]
    alpha_eeg = 0 * alpha_eeg
    theta_eeg = 0 * theta_eeg
    delta_eeg = 0 * delta_eeg
    beta_eeg = 0 * beta_eeg
    n=2
    # n = 6
    matrix0 = torch.tensor([[1 if x <= y else 0 for x in range(n)] for y in range(n)]).cuda(my_config.Config.GPU_id) #左下1
    matrix1 = torch.tensor([[1 if x > y else 0 for x in range(n)] for y in range(n)]).cuda(my_config.Config.GPU_id) #左下0
    part_list_low = []
    part_list_high = []


    for ptn in range(part_num):
        # current_delta = delta_eeg[:, :, :, ptn * part_length : (ptn + 1) * part_length]
        # current_theta = theta_eeg[:, :, :, ptn * part_length : (ptn + 1) * part_length]
        # current_alpha = alpha_eeg[:, :, :, ptn * part_length : (ptn + 1) * part_length]
        # current_beta = beta_eeg[:, :, :, ptn * part_length : (ptn + 1) * part_length]
        current_gamma = gamma_eeg[:, :, :, ptn * part_length : (ptn + 1) * part_length]
        current_upper = upper_eeg[:, :, :, ptn * part_length : (ptn + 1) * part_length]

        random_column =  torch.randint(n-1, [batch,]).cuda(my_config.Config.GPU_id)

        weight0 = torch.reshape(matrix0[random_column],[1, -1]).cuda(my_config.Config.GPU_id)#32,6
        weight1 = torch.reshape(matrix1[random_column],[1, -1]).cuda(my_config.Config.GPU_id)#32,6

        data_matrix = torch.stack((#current_delta, current_theta, current_alpha, current_beta,
                                   current_gamma, current_upper), dim=1).cuda(my_config.Config.GPU_id)
        # print(data_matrix.shape)
        data_matrix_puppet = torch.reshape(data_matrix, ([data_matrix.shape[2],data_matrix.shape[3],data_matrix.shape[4], -1])).cuda(my_config.Config.GPU_id)

        lower_matrix = torch.mul(data_matrix_puppet, weight0).cuda(my_config.Config.GPU_id)
        lower_matrix = torch.reshape(lower_matrix, ([batch, n, data_matrix.shape[2],data_matrix.shape[3],data_matrix.shape[4]])).cuda(my_config.Config.GPU_id)
        lower = torch.sum(lower_matrix, dim = 1).cuda(my_config.Config.GPU_id)
        # print(lower.shape)
        higher_matrix = torch.mul(data_matrix_puppet, weight1).cuda(my_config.Config.GPU_id)
        higher_matrix = torch.reshape(higher_matrix, ([batch, n, data_matrix.shape[2],data_matrix.shape[3],data_matrix.shape[4]])).cuda(my_config.Config.GPU_id)
        higher = torch.sum(higher_matrix, dim = 1).cuda(my_config.Config.GPU_id)

        part_list_low.append(lower)
        part_list_high.append(higher)

    all_low = torch.stack(part_list_low, dim=3).cuda(my_config.Config.GPU_id)
    all_high = torch.stack(part_list_high, dim=3).cuda(my_config.Config.GPU_id)

    all_low = torch.reshape(all_low,([all_low.shape[0], all_low.shape[1], all_low.shape[2], -1])).cuda(
        my_config.Config.GPU_id)
    all_high = torch.reshape(all_high,([all_low.shape[0], all_high.shape[1], all_high.shape[2], -1])).cuda(
        my_config.Config.GPU_id)

    return data_eeg, all_low, all_high
def random_wave_decomposed_combine(part_num , data_eeg, delta_eeg, theta_eeg, alpha_eeg, beta_eeg, gamma_eeg, upper_eeg):
    part_length = int(data_eeg.shape[3] / part_num)
    batch = data_eeg.shape[0]
    n = 6

    part_list_low = []
    part_list_high = []

    # print((data_eeg-delta_eeg-theta_eeg-alpha_eeg-beta_eeg-gamma_eeg-upper_eeg).sum())

    for ptn in range(part_num):
        current_delta = delta_eeg[:, :, :, ptn * part_length: (ptn + 1) * part_length]
        current_theta = theta_eeg[:, :, :, ptn * part_length: (ptn + 1) * part_length]
        current_alpha = alpha_eeg[:, :, :, ptn * part_length: (ptn + 1) * part_length]
        current_beta = beta_eeg[:, :, :, ptn * part_length: (ptn + 1) * part_length]
        current_gamma = gamma_eeg[:, :, :, ptn * part_length: (ptn + 1) * part_length]
        current_upper = upper_eeg[:, :, :, ptn * part_length: (ptn + 1) * part_length]

        weight0 = torch.zeros((batch, 6)).cuda(my_config.Config.GPU_id)
        sum = torch.ones((batch, 6)).cuda(my_config.Config.GPU_id)
        for i in range(batch):
            row_indices = torch.randperm(6)[:1].cuda(my_config.Config.GPU_id)  # Randomly choose one index to set as '1'
            weight0[i][row_indices] = 1

        for i in range(batch):
            num_fives = torch.randint(0, 5, (1,)).cuda(my_config.Config.GPU_id)  # Randomly choose the number of '5's in each row
            if num_fives > 0:
                row_indices = torch.randperm(6)[:num_fives].cuda(my_config.Config.GPU_id)  # Randomly choose indices to set as '5'
                weight0[i][row_indices] = 1

        weight1 = (sum - weight0).cuda(my_config.Config.GPU_id)
        weight0 = torch.reshape(weight0, [1, -1]).cuda(my_config.Config.GPU_id)  # 32,6
        weight1 = torch.reshape(weight1, [1, -1]).cuda(my_config.Config.GPU_id)  # 32,6
        # print(weight1.shape)
        # print(weight0.shape)

        # random_column = torch.randint(n - 1, [batch, ]).cuda(my_config.Config.GPU_id)
        #
        # weight0 = torch.reshape(matrix0[random_column], [1, -1]).cuda(my_config.Config.GPU_id)  # 32,6
        # weight1 = torch.reshape(matrix1[random_column], [1, -1]).cuda(my_config.Config.GPU_id)  # 32,6

        data_matrix = torch.stack((current_delta, current_theta, current_alpha,
                                   current_beta, current_gamma, current_upper), dim=1).cuda(my_config.Config.GPU_id)
        # print(data_matrix.shape)
        data_matrix_puppet = torch.reshape(data_matrix, (
        [data_matrix.shape[2], data_matrix.shape[3], data_matrix.shape[4], -1])).cuda(my_config.Config.GPU_id)
        # print(data_matrix_puppet.shape)
        lower_matrix = torch.mul(data_matrix_puppet, weight0).cuda(my_config.Config.GPU_id)
        # print(lower_matrix.shape)

        lower_matrix = torch.reshape(lower_matrix, (
        [batch, 6, data_matrix.shape[2], data_matrix.shape[3], data_matrix.shape[4]])).cuda(my_config.Config.GPU_id)
        # print(lower_matrix.shape)

        lower = torch.sum(lower_matrix, dim=1).cuda(my_config.Config.GPU_id)
        # print(lower_matrix.shape)

        # print(lower.shape)

        higher_matrix = torch.mul(data_matrix_puppet, weight1).cuda(my_config.Config.GPU_id)
        higher_matrix = torch.reshape(higher_matrix, (
        [batch, 6, data_matrix.shape[2], data_matrix.shape[3], data_matrix.shape[4]])).cuda(my_config.Config.GPU_id)
        higher = torch.sum(higher_matrix, dim=1).cuda(my_config.Config.GPU_id)

        part_list_low.append(lower)
        part_list_high.append(higher)

    all_low = torch.stack(part_list_low, dim=3).cuda(my_config.Config.GPU_id)
    all_high = torch.stack(part_list_high, dim=3).cuda(my_config.Config.GPU_id)

    all_low = torch.reshape(all_low, ([all_low.shape[0], all_low.shape[1], all_low.shape[2], -1])).cuda(
        my_config.Config.GPU_id)
    all_high = torch.reshape(all_high, ([all_low.shape[0], all_high.shape[1], all_high.shape[2], -1])).cuda(
        my_config.Config.GPU_id)

    # print((data_eeg - all_low - all_high).sum())

    return data_eeg, all_low, all_high

class PatchEmbed(nn.Module):
    def __init__(self, patch_size, in_chans, embed_dim):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv1d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=False),
            Rearrange('b c l -> b l c')
        )

    def forward(self, x):
        return self.proj(x)

class Multi_TimeTransformer(nn.Module):
    def __init__(self, part_num=32, org_signal_length=512, org_channel=20, patch_size=80, pool='mean', in_chans=1, embed_dim=128,
                 depth=2, num_heads=4, mlp_ratio=4, dropout=0.2, num_classes=4, scale = 1):
        super().__init__()
        signal_length = org_signal_length * org_channel
        assert signal_length % patch_size == 0
        self.num_patches = signal_length // patch_size

        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        self.pool = pool
        self.data_rebuild = Rearrange('b c h w -> b h (c w)')

        self.patch_embed_delta = PatchEmbed(int(patch_size / scale), in_chans, embed_dim)
        self.patch_embed_theta = PatchEmbed(int(patch_size * scale), in_chans, embed_dim)
        self.patch_embed_alpha = PatchEmbed(int(patch_size / scale), in_chans, embed_dim)
        self.patch_embed_beta = PatchEmbed(int(patch_size * scale), in_chans, embed_dim)
        self.patch_embed_gamma = PatchEmbed(int(patch_size / scale), in_chans, embed_dim)
        self.patch_embed_upper = PatchEmbed(int(patch_size * scale), in_chans, embed_dim)
        self.patch_embed = PatchEmbed(patch_size, in_chans, embed_dim)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.cls_token_delta = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.cls_token_theta = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.cls_token_alpha = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.cls_token_beta = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.cls_token_gamma = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.cls_token_upper = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim), requires_grad=False)
        self.pos_embed_delta = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim), requires_grad=False)
        self.pos_embed_theta = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim), requires_grad=False)
        self.pos_embed_alpha = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim), requires_grad=False)
        self.pos_embed_beta = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim), requires_grad=False)
        self.pos_embed_gamma = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim), requires_grad=False)
        self.pos_embed_upper = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim), requires_grad=False)

        self.dropout = nn.Dropout(dropout)
        # self.blocks = Multiout_Transformer(embed_dim, depth, num_heads, mlp_ratio, dropout)
        self.blocks = Multi_Layer_Waveout_Transformer_FA(embed_dim, depth, num_heads, mlp_ratio, dropout)
        self.classifier = nn.Linear(embed_dim, num_classes)

        self.initialize_weights()

    def initialize_weights(self):
        pos_embed = get_1d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.num_patches), cls_token=True)
        pos_embed_delta = get_1d_sincos_pos_embed(self.pos_embed_delta.shape[-1], int(self.num_patches), cls_token=True)
        pos_embed_theta = get_1d_sincos_pos_embed(self.pos_embed_theta.shape[-1], int(self.num_patches), cls_token=True)
        pos_embed_alpha = get_1d_sincos_pos_embed(self.pos_embed_alpha.shape[-1], int(self.num_patches), cls_token=True)
        pos_embed_beta = get_1d_sincos_pos_embed(self.pos_embed_beta.shape[-1], int(self.num_patches), cls_token=True)
        pos_embed_gamma = get_1d_sincos_pos_embed(self.pos_embed_gamma.shape[-1], int(self.num_patches), cls_token=True)
        pos_embed_upper = get_1d_sincos_pos_embed(self.pos_embed_upper.shape[-1], int(self.num_patches), cls_token=True)

        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        self.pos_embed_delta.data.copy_(torch.from_numpy(pos_embed_delta).float().unsqueeze(0))
        self.pos_embed_theta.data.copy_(torch.from_numpy(pos_embed_theta).float().unsqueeze(0))
        self.pos_embed_alpha.data.copy_(torch.from_numpy(pos_embed_alpha).float().unsqueeze(0))
        self.pos_embed_beta.data.copy_(torch.from_numpy(pos_embed_beta).float().unsqueeze(0))
        self.pos_embed_gamma.data.copy_(torch.from_numpy(pos_embed_gamma).float().unsqueeze(0))
        self.pos_embed_upper.data.copy_(torch.from_numpy(pos_embed_upper).float().unsqueeze(0))


    def forward(self, data_eeg, delta_eeg, theta_eeg, alpha_eeg, beta_eeg, gamma_eeg, upper_eeg):
        # divide x with high and low frequency
        x = self.data_rebuild(data_eeg)
        x_delta = self.data_rebuild(delta_eeg)
        x_theta = self.data_rebuild(theta_eeg)
        x_alpha = self.data_rebuild(alpha_eeg)
        x_beta = self.data_rebuild(beta_eeg)
        x_gamma = self.data_rebuild(gamma_eeg)
        x_upper = self.data_rebuild(upper_eeg)

        # embed patches
        x = self.patch_embed(x)
        x_delta = self.patch_embed_delta(x_delta)
        x_theta = self.patch_embed_theta(x_theta)
        x_alpha = self.patch_embed_alpha(x_alpha)
        x_beta = self.patch_embed_beta(x_beta)
        x_gamma = self.patch_embed_gamma(x_gamma)
        x_upper = self.patch_embed_upper(x_upper)

        # add pos embed w/o cls toke=
        x = x + self.pos_embed[:, 1:, :]
        x_delta = x_delta + self.pos_embed_delta[:, 1:, :]
        x_theta = x_theta + self.pos_embed_theta[:, 1:, :]
        x_alpha = x_alpha + self.pos_embed_alpha[:, 1:, :]
        x_beta = x_beta + self.pos_embed_beta[:, 1:, :]
        x_gamma = x_gamma + self.pos_embed_gamma[:, 1:, :]
        x_upper = x_upper + self.pos_embed_upper[:, 1:, :]

        # append cls token
        x_cls_token = self.cls_token + self.pos_embed[:, :1, :]
        x_cls_tokens = x_cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((x_cls_tokens, x), dim=1)

        delta_cls_token = self.cls_token_delta + self.pos_embed_delta[:, :1, :]
        delta_cls_tokens = delta_cls_token.expand(x_delta.shape[0], -1, -1)
        x_delta = torch.cat((delta_cls_tokens, x_delta), dim=1)

        theta_cls_token = self.cls_token_theta + self.pos_embed_theta[:, :1, :]
        theta_cls_tokens = theta_cls_token.expand(x_theta.shape[0], -1, -1)
        x_theta = torch.cat((theta_cls_tokens, x_theta), dim=1)

        alpha_cls_token = self.cls_token_alpha + self.pos_embed_alpha[:, :1, :]
        alpha_cls_tokens = alpha_cls_token.expand(x_alpha.shape[0], -1, -1)
        x_alpha = torch.cat((alpha_cls_tokens, x_alpha), dim=1)

        beta_cls_token = self.cls_token_beta + self.pos_embed_beta[:, :1, :]
        beta_cls_tokens = beta_cls_token.expand(x_beta.shape[0], -1, -1)
        x_beta = torch.cat((beta_cls_tokens, x_beta), dim=1)

        gamma_cls_token = self.cls_token_gamma + self.pos_embed_gamma[:, :1, :]
        gamma_cls_tokens = gamma_cls_token.expand(x_gamma.shape[0], -1, -1)
        x_gamma = torch.cat((gamma_cls_tokens, x_gamma), dim=1)

        upper_cls_token = self.cls_token_upper + self.pos_embed_upper[:, :1, :]
        upper_cls_tokens = upper_cls_token.expand(x_upper.shape[0], -1, -1)
        x_upper = torch.cat((upper_cls_tokens, x_upper), dim=1)

        # apply Transformer blocks
        x = self.dropout(x)
        x_delta = self.dropout(x_delta)
        x_theta = self.dropout(x_theta)
        x_alpha = self.dropout(x_alpha)
        x_beta = self.dropout(x_beta)
        x_gamma = self.dropout(x_gamma)
        x_upper = self.dropout(x_upper)

        x_list, delta_list, theta_list, \
        alpha_list, beta_list, gamma_list, upper_list, \
        x, x_delta, x_theta, x_alpha, x_beta, x_gamma, x_upper, L1, trained_weight\
            = self.blocks(x, x_delta, x_theta, x_alpha, x_beta, x_gamma, x_upper)

        list_feature_map_x, list_feature_map_delta, list_feature_map_theta = [], [], []
        list_feature_map_alpha, list_feature_map_beta, list_feature_map_gamma = [], [], []
        list_feature_map_upper = []

        for i in range(len(x_list)):
            cur_x, cur_delta, cur_theta, cur_alpha, cur_beta, cur_gamma, cur_upper = \
                x_list[i], delta_list[i], theta_list[i], alpha_list[i], beta_list[i], gamma_list[i], upper_list[i]

            # using token feature as classification head or avgerage pooling for all feature
            list_feature_map_x.append(cur_x.mean(dim=1) if self.pool == 'mean' else cur_x[:, 0])
            list_feature_map_delta.append(cur_delta.mean(dim=1) if self.pool == 'mean' else cur_delta[:, 0])
            list_feature_map_theta.append(cur_theta.mean(dim=1) if self.pool == 'mean' else cur_theta[:, 0])
            list_feature_map_alpha.append(cur_alpha.mean(dim=1) if self.pool == 'mean' else cur_alpha[:, 0])
            list_feature_map_beta.append(cur_beta.mean(dim=1) if self.pool == 'mean' else cur_beta[:, 0])
            list_feature_map_gamma.append(cur_gamma.mean(dim=1) if self.pool == 'mean' else cur_gamma[:, 0])
            list_feature_map_upper.append(cur_upper.mean(dim=1) if self.pool == 'mean' else cur_upper[:, 0])

        # classify
        feature_map_x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]
        feature_map_delta = x_delta.mean(dim=1) if self.pool == 'mean' else x_delta[:, 0]
        feature_map_theta = x_theta.mean(dim=1) if self.pool == 'mean' else x_theta[:, 0]
        feature_map_alpha = x_alpha.mean(dim=1) if self.pool == 'mean' else x_alpha[:, 0]
        feature_map_beta = x_beta.mean(dim=1) if self.pool == 'mean' else x_beta[:, 0]
        feature_map_gamma = x_gamma.mean(dim=1) if self.pool == 'mean' else x_gamma[:, 0]
        feature_map_upper = x_upper.mean(dim=1) if self.pool == 'mean' else x_upper[:, 0]

        pred_x = self.classifier(feature_map_x)
        pred_delta = self.classifier(feature_map_delta)
        pred_theta = self.classifier(feature_map_theta)
        pred_alpha = self.classifier(feature_map_alpha)
        pred_beta = self.classifier(feature_map_beta)
        pred_gamma = self.classifier(feature_map_gamma)
        pred_upper = self.classifier(feature_map_upper)

        return pred_x, pred_delta, pred_theta, pred_alpha, pred_beta, pred_gamma, pred_upper, \
               feature_map_x, feature_map_delta, feature_map_theta, feature_map_alpha, feature_map_beta, feature_map_gamma, feature_map_upper, \
               list_feature_map_x, list_feature_map_delta, list_feature_map_theta, list_feature_map_alpha, \
               list_feature_map_beta, list_feature_map_gamma, list_feature_map_upper, L1, trained_weight

