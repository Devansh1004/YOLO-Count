from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from yolo_count.models.module import ConvModule, make_divisible


class ContrastiveHead(nn.Module):
    def __init__(self, embed_dims: int, use_einsum: bool = True) -> None:
        super().__init__()

        # Add a projection layer to match dimensions
        self.proj = nn.Conv2d(
            embed_dims, 512, kernel_size=1
        )  # Map any number of channels to 512 dims
        # Initialize as identity when input dims are 512
        if embed_dims == 512:
            nn.init.eye_(self.proj.weight.squeeze())
            nn.init.zeros_(self.proj.bias)

        self.bias = nn.Parameter(torch.zeros([]))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.use_einsum = use_einsum

    def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """Forward function

        Args:
            x (torch.Tensor): image features, shape [B, C, H, W]
            w (torch.Tensor): text features, shape [B, K, 512]

        Returns:
            torch.Tensor: similarity scores, shape [B, K, H, W]
        """
        # Project to the same dimension
        x = self.proj(x)

        # L2 normalize
        x = F.normalize(x, dim=1, p=2)
        w = F.normalize(w, dim=-1, p=2)

        if self.use_einsum:
            # Compute similarity using einsum
            x = torch.einsum("bchw,bkc->bkhw", x, w)
        else:
            # Compute similarity using matrix multiplication
            batch, channel, height, width = x.shape
            _, k, _ = w.shape
            x = x.permute(0, 2, 3, 1)  # [B,C,H,W] -> [B,H,W,C]
            x = x.reshape(batch, -1, channel)  # [B,H,W,C] -> [B,HW,C]
            w = w.permute(0, 2, 1)  # [B,K,C] -> [B,C,K]
            x = torch.matmul(x, w)  # [B,HW,K]
            x = x.reshape(batch, height, width, k)
            x = x.permute(0, 3, 1, 2)  # [B,H,W,K] -> [B,K,H,W]

        # Apply temperature scale and bias
        x = x * self.logit_scale.exp() + self.bias
        return x


class ProportionCountingHead(nn.Module):

    def __init__(
        self,
        in_channels: List[int],
        embed_dims: int,
        widen_factor: float = 1.0,
        norm_cfg: dict = dict(type="BN", momentum=0.03, eps=0.001),
        act_cfg: dict = dict(type="SiLU", inplace=True),
        freeze_all: bool = False,
    ):
        super().__init__()
        self.embed_dims = make_divisible(embed_dims, widen_factor)
        self.freeze_all = freeze_all
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg

        # Feature extraction for classification branch
        self.conv_80_cls = ConvModule(
            make_divisible(in_channels[0], widen_factor),
            self.embed_dims,
            kernel_size=3,
            padding=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
        )
        self.conv_40_cls = ConvModule(
            make_divisible(in_channels[1], widen_factor),
            self.embed_dims,
            kernel_size=3,
            padding=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
        )
        self.conv_20_cls = ConvModule(
            make_divisible(in_channels[2], widen_factor),
            self.embed_dims,
            kernel_size=3,
            padding=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
        )

        # Semantic branch
        self.semantic_branch = nn.Sequential(
            ConvModule(
                self.embed_dims * 3,
                self.embed_dims,
                kernel_size=1,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
            ),
            ConvModule(
                self.embed_dims,
                self.embed_dims,
                kernel_size=3,
                padding=1,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
            ),
        )

        # Contrastive head
        self.cls_contrast = ContrastiveHead(self.embed_dims)

        # Density prediction branch
        self.density_branch = nn.ModuleList(
            [
                nn.Sequential(
                    ConvModule(
                        make_divisible(in_channels[0], widen_factor),
                        self.embed_dims,
                        kernel_size=3,
                        padding=1,
                        norm_cfg=norm_cfg,
                        act_cfg=act_cfg,
                    ),
                    ConvModule(
                        self.embed_dims,
                        self.embed_dims // 2,
                        kernel_size=3,
                        padding=1,
                        norm_cfg=norm_cfg,
                        act_cfg=act_cfg,
                    ),
                ),
                nn.Sequential(
                    ConvModule(
                        make_divisible(in_channels[1], widen_factor),
                        self.embed_dims // 2,
                        kernel_size=3,
                        padding=1,
                        norm_cfg=norm_cfg,
                        act_cfg=act_cfg,
                    ),
                    ConvModule(
                        self.embed_dims // 2,
                        self.embed_dims // 2,
                        kernel_size=3,
                        padding=1,
                        norm_cfg=norm_cfg,
                        act_cfg=act_cfg,
                    ),
                ),
                nn.Sequential(
                    ConvModule(
                        make_divisible(in_channels[2], widen_factor),
                        self.embed_dims // 2,
                        kernel_size=3,
                        padding=1,
                        norm_cfg=norm_cfg,
                        act_cfg=act_cfg,
                    ),
                    ConvModule(
                        self.embed_dims // 2,
                        self.embed_dims // 2,
                        kernel_size=3,
                        padding=1,
                        norm_cfg=norm_cfg,
                        act_cfg=act_cfg,
                    ),
                ),
            ]
        )

        # Density map prediction head
        self.density_head = nn.Sequential(
            ConvModule(
                self.embed_dims * 3 // 2,
                self.embed_dims // 2,
                kernel_size=3,
                padding=1,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
            ),
            ConvModule(
                self.embed_dims // 2,
                self.embed_dims // 4,
                kernel_size=3,
                padding=1,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
            ),
            ConvModule(
                self.embed_dims // 4,
                self.embed_dims // 8,
                kernel_size=3,
                padding=1,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
            ),
            nn.Conv2d(self.embed_dims // 8, 1, 1),
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize network weights"""
        # ConvModule already applies Kaiming initialization
        # Only need to specially handle the last layer of the density head
        nn.init.constant_(self.density_head[-1].weight, 0)
        nn.init.constant_(self.density_head[-1].bias, 0.01)

    def train(self, mode: bool = True):

        super().train(mode)
        if self.freeze_all:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def forward(
        self,
        img_feats_1: Tuple[torch.Tensor],
        img_feats_2: Tuple[torch.Tensor],
        txt_feats: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        feat_80 = self.conv_80_cls(img_feats_1[0])
        feat_40 = self.conv_40_cls(img_feats_1[1])
        feat_20 = self.conv_20_cls(img_feats_1[2])

        feat_40_up = F.interpolate(
            feat_40, size=feat_80.shape[2:], mode="bilinear", align_corners=False
        )
        feat_20_up = F.interpolate(
            feat_20, size=feat_80.shape[2:], mode="bilinear", align_corners=False
        )

        semantic_feat = torch.cat([feat_80, feat_40_up, feat_20_up], dim=1)
        semantic_feat = self.semantic_branch(semantic_feat)
        cls_logit = self.cls_contrast(semantic_feat, txt_feats)

        density_feat_80 = self.density_branch[0](img_feats_2[0])
        density_feat_40 = F.interpolate(
            self.density_branch[1](img_feats_2[1]),
            size=density_feat_80.shape[2:],
            mode="bilinear",
            align_corners=False,
        )
        density_feat_20 = F.interpolate(
            self.density_branch[2](img_feats_2[2]),
            size=density_feat_80.shape[2:],
            mode="bilinear",
            align_corners=False,
        )

        density_feat = torch.cat(
            [density_feat_80, density_feat_40, density_feat_20], dim=1
        )

        density_pred = F.relu(self.density_head(density_feat))

        return cls_logit, density_pred
