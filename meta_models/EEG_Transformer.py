import torch
import torch.nn as nn
from base_models.base_transformer import Transformer
from utils.utils import get_1d_sincos_pos_embed
from einops.layers.torch import Rearrange

class PatchEmbed(nn.Module):
    def __init__(self, patch_size, in_chans, embed_dim):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv1d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=False),
            Rearrange('b c l -> b l c')
        )

    def forward(self, x):
        return self.proj(x)


class TimeTransformer(nn.Module):
    def __init__(self, org_signal_length=512, org_channel=20, patch_size=80, pool='mean', in_chans=1, embed_dim=128,
                 depth=4, num_heads=4, mlp_ratio=4, dropout=0.2, num_classes=4):
        super().__init__()
        signal_length = org_signal_length * org_channel
        assert signal_length % patch_size == 0
        self.num_patches = signal_length // patch_size
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        self.pool = pool
        self.data_rebuild = Rearrange('b c h w -> b h (c w)')
        self.patch_embed = PatchEmbed(patch_size, in_chans, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim), requires_grad=False)
        self.dropout = nn.Dropout(dropout)
        self.blocks = Transformer(embed_dim, depth, num_heads, mlp_ratio, dropout)
        # self.classifier = nn.Sequential(
        #     nn.Linear(embed_dim, num_classes),
        #     nn.BatchNorm1d(num_classes))
        self.classifier = nn.Linear(embed_dim, num_classes)

        self.initialize_weights()

    def initialize_weights(self):
        pos_embed = get_1d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.num_patches), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

    def forward(self, x):
        x = self.data_rebuild(x)
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        x = self.dropout(x)
        x = self.blocks(x)

        # using token feature as classification head or avgerage pooling for all feature
        feature_map = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]

        # classify
        pred = self.classifier(feature_map)

        return x, feature_map, pred
